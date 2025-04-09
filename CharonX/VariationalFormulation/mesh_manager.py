from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import SpatialCoordinate, Measure
from numpy import hstack, argsort, finfo, full_like, array, zeros, where, unique
from dolfinx.fem import functionspace, Function

class MeshManager:
    """
    Gestionnaire de maillage et de mesures d'intégration.
    
    Cette classe encapsule toutes les opérations liées au maillage,
    incluant la création de sous-maillages pour les facettes, le marquage des frontières,
    et la définition des mesures d'intégration.
    """
    def __init__(self, mesh, name):
        """
        Initialise le gestionnaire de maillage.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Maillage principal du problème.
        name : str
            Nom du problème (ex: "Axisymmetric", "PlaneStrain", etc.)
        """
        self.mesh = mesh
        self.name = name
        self.dim = mesh.topology.dim
        self.fdim = self.dim - 1
        self.h = self.calculate_mesh_size()
        
    def mark_boundary(self, flag_list, coord_list, localisation_list, tol=finfo(float).eps):
        """
        Marque les frontières du maillage.
        
        Parameters
        ----------
        flag_list : List[int]
            Drapeaux associés aux domaines.
        coord_list : List[str]
            Variables permettant de repérer les domaines.
        localisation_list : List[float]
            Valeurs au voisinage desquelles les points sont récupérés.
        tol : float, optional
            Tolérance pour la récupération des points. Par défaut: finfo(float).eps.
        """
        # Récupérer les facettes et leurs drapeaux
        facets, full_flag = self._set_facet_flags(flag_list, coord_list, localisation_list, tol)
        
        # Ajouter des drapeaux personnalisés si nécessaire
        facets, full_flag = self.set_custom_facet_flags(facets, full_flag)
        
        # Assembler et trier les facettes marquées
        marked_facets = hstack(facets)
        marked_values = hstack(full_flag)
        sorted_facets = argsort(marked_facets)
        
        # Créer les tags de maillage
        self.facet_tag = meshtags(
            self.mesh, 
            self.fdim, 
            marked_facets[sorted_facets], 
            marked_values[sorted_facets]
        )
        
    def set_custom_facet_flags(self, facets, full_flag):
        """
        Méthode à surcharger pour ajouter des drapeaux de facette personnalisés.
        
        Parameters
        ----------
        facets : List
            Liste des facettes.
        full_flag : List
            Liste des drapeaux.
            
        Returns
        -------
        Tuple
            (facets, full_flag) éventuellement modifiés.
        """
        return facets, full_flag
        
    def _set_facet_flags(self, flag_list, coord_list, localisation_list, tol):
        """
        Définit les drapeaux de facette en fonction des coordonnées et valeurs données.
        
        Parameters
        ----------
        flag_list : List[int]
            Drapeaux à attribuer.
        coord_list : List[str]
            Coordonnées à vérifier.
        localisation_list : List[float]
            Valeurs de référence.
        tol : float
            Tolérance.
            
        Returns
        -------
        Tuple
            (facets, full_flag) contenant les facettes marquées et leurs drapeaux.
        """
        facets = []
        full_flag = []
        
        for flag, coord, loc in zip(flag_list, coord_list, localisation_list):
            # Fonction de filtrage pour identifier les facettes
            def boundary_filter(x):
                return abs(x[self.index_coord(coord)] - loc) < tol
                
            # Localiser les entités correspondantes
            found_facets = locate_entities_boundary(self.mesh, self.fdim, boundary_filter)
            facets.append(found_facets)
            full_flag.append(full_like(found_facets, flag))
            
        return facets, full_flag
        
    def index_coord(self, coord):
        """
        Renvoie l'index correspondant à la variable spatiale.
        
        Parameters
        ----------
        coord : str
            Coordonnée parmi: "x", "y", "z", "r"
            
        Returns
        -------
        int
            Index de la coordonnée dans le système de coordonnées.
        """
        if coord in ("x", "r"):
            return 0
        elif coord == "y":
            return 1
        elif coord == "z":
            if self.name == "Axisymmetric":
                return 1
            else:
                return 2
        else:
            raise ValueError(f"Invalid coordinate: {coord}")
    
    def get_boundary_element_size(self, flag):
        """
        Calcule les surfaces des éléments connectés aux facettes marquées par un drapeau.
    
        Parameters
        ----------
        flag : int
            Drapeau des facettes d'intérêt.
    
        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray)
            Tableaux contenant les surfaces et centroïdes des éléments.
        """
        # Trouver les facettes marquées par le drapeau
        marked_facets = where(self.facet_tag.values == flag)[0]
        facet_indices = self.facet_tag.indices[marked_facets]
    
        # Connexions facette -> cellule
        facet_to_cell_map = self.mesh.topology.connectivity(1, 2)
    
        # Trouver les cellules connectées aux facettes marquées
        connected_cells = []
        for facet in facet_indices:
            if facet_to_cell_map.offsets[facet] != facet_to_cell_map.offsets[facet + 1]:
                connected_cells.extend(facet_to_cell_map.links(facet))
    
        connected_cells = unique(connected_cells)  # Supprimer les doublons
    
        # Calculer la surface des cellules connectées
        areas = []
        centroids_x = []
        
        for cell in connected_cells:
            # Récupérer les sommets de la cellule
            cell_vertices = self.mesh.geometry.x[self.mesh.topology.connectivity(2, 0).links(cell)]
            
            # Calculer la surface et le centroïde selon le type d'élément
            area, centroid_x = self._calculate_cell_properties(cell_vertices)
            
            areas.append(area)
            centroids_x.append(centroid_x)
    
        return array(areas), array(centroids_x)
    
    def _calculate_cell_properties(self, vertices):
        """
        Calcule la surface et le centroïde d'une cellule.
        
        Parameters
        ----------
        vertices : numpy.ndarray
            Sommets de la cellule.
            
        Returns
        -------
        tuple (float, float)
            Surface et coordonnée x du centroïde.
        """
        if len(vertices) == 3:  # Triangle
            x1, y1 = vertices[0][0], vertices[0][1]
            x2, y2 = vertices[1][0], vertices[1][1]
            x3, y3 = vertices[2][0], vertices[2][1]
            
            # Calcul de la surface du triangle
            area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            centroid_x = (x1 + x2 + x3) / 3
            
        elif len(vertices) == 4:  # Quadrilatère
            x1, y1 = vertices[0][0], vertices[0][1]
            x2, y2 = vertices[1][0], vertices[1][1]
            x3, y3 = vertices[2][0], vertices[2][1]
            x4, y4 = vertices[3][0], vertices[3][1]
            
            # Calcul de la surface du quadrilatère (somme de deux triangles)
            area = (0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) +
                    0.5 * abs(x1 * (y3 - y4) + x3 * (y4 - y1) + x4 * (y1 - y3)))
            centroid_x = (x1 + x2 + x3 + x4) / 4
            
        else:
            raise ValueError(f"Nombre de sommets non pris en charge: {len(vertices)}")
            
        return area, centroid_x
    
    def set_measures(self, quadrature):
        """
        Définit les mesures d'intégration pour le problème.
        
        Parameters
        ----------
        quadrature : Quadrature
            Schéma de quadrature à utiliser.
        """
        # Coordonnée radiale pour les modèles axisymétriques
        if self.name in ["Axisymmetric", "CylindricalUD", "SphericalUD"]:
            self.r = SpatialCoordinate(self.mesh)[0]
        else: 
            self.r = None
            
        # Définir les mesures d'intégration
        self.dx = Measure("dx", domain=self.mesh, metadata=quadrature.metadata)
        self.dx_l = Measure("dx", domain=self.mesh, metadata=quadrature.lumped_metadata)
        self.ds = Measure('ds')(subdomain_data=self.facet_tag)
        self.dS = Measure('dS')(subdomain_data=self.facet_tag)
        
    def calculate_mesh_size(self):
        """
        Calcule la taille locale des éléments du maillage.
        
        Returns
        -------
        Function
            Fonction contenant la taille locale des éléments.
        """
        # Créer un espace fonctionnel pour stocker la taille
        h_loc = Function(functionspace(self.mesh, ("DG", 0)), name="MeshSize")
        
        # Calculer la taille pour chaque cellule
        num_cells = self.mesh.topology.index_map(self.dim).size_local
        h_local = zeros(num_cells)
        
        for i in range(num_cells):
            h_local[i] = self.mesh.h(self.dim, array([i]))
        
        # Affecter les valeurs calculées
        h_loc.x.array[:] = h_local
        
        return h_loc