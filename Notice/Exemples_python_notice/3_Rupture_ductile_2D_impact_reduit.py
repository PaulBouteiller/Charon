from CharonX import *
###### Modèle élastique ######
E = 210e3
nu = 0.3
rho = 7.8e-3
eos_type = "U8"
dico_eos = {"kappa" : 175e3, "alpha" : 12e-6}
devia_type = "NeoHook"
dico_devia = {"E": E, "nu" : nu}
Acier = Material(rho, 500, eos_type, devia_type, dico_eos, dico_devia)

###### Temps simulation ######
TFin = 3e-3 #Simulation de 3 micro-seconde
pas_de_temps = 1.25e-6 #Pas de temps 1.25 nano
t_plateau = 2e-4
t_load = 5e-6
###### Paramètres géométriques ######
elancement = 6
x_infg = 3e-6
y_infg = 0
x_supd = 100
y_supd = x_supd/elancement
Nx = 200
Ny = int(3 * Nx /elancement)

class Cylindre_axi(Axisymetric):
    def __init__(self, material):
        Axisymetric.__init__(self, material, damage = "Johnson", plastic = "HPP_Plasticity")

    def prefix(self):
        return "Rupture_ecaillage_2D_Johnson"

    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(x_infg, y_infg), \
                               (x_supd, y_supd)], [Nx, Ny], CellType.quadrilateral)
            
    def fem_parameters(self):
        self.u_deg =2
        self.schema = "reduit"
        
    def set_damping(self):
        #Pour cet essai, il est conseillé d'ajouter une pseudo viscosité élevée
        damp = {}
        damp.update({"damping" : True})
        damp.update({"linear_coeff" : 0.3})
        damp.update({"quad_coeff" : 2})
        damp.update({"correction" : True})
        return damp
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["r", "r"], [x_infg, x_supd])
        
    def set_boundary_condition(self):
        size, centroid = self.get_boundary_element_size(1)
        V = 2 * 3.14 * size * centroid
        m = rho  * V
        quarter_m = m / 4
        self.bcs.add_axi(region=1, value = quarter_m)
        # self.bcs.add_axi(region=1)
        self.bcs.add_Ur(region=1)
        
    def set_custom_facet_flags(self, facets, full_flag):
        def impact_zone(x):
            bool_array_1 = abs(x[1] - y_infg) < np.finfo(float).eps
            bool_array_2 = x[0] < 95
            return np.logical_and(bool_array_1, bool_array_2)
        facets.append(locate_entities_boundary(self.mesh, self.fdim, impact_zone))
        full_flag.append(np.full_like(facets[-1], 3))
        return facets, full_flag  
        
    def set_loading(self):
        x = SpatialCoordinate(self.mesh)
        ufl_condition = 1 - x[0] / x_supd
        V = self.V.sub(0).collapse()[0]
        expr = Expression(ufl_condition, V.element.interpolation_points())
        func_chargement = Function(V)
        func_chargement.interpolate(expr)
        magnitude = 3e4
        chargement = MyConstant(self.mesh, t_load, t_plateau, magnitude, Type = "SmoothCreneau")
        # chargement.function = func_chargement
        self.loading.add_Fz(chargement, self.u_, self.ds(3))
        
    def set_plastic(self):
        sig_Y=1e3
        H=1e2
        self.constitutive.plastic.set_plastic(sigY = sig_Y, H = H)
        
    def set_damage(self):
        self.constitutive.damage.set_damage(self.mesh, eta=1e-3, sigma_0=500)
        condition = SpatialCoordinate(self.mesh)[1] < 6.
        self.constitutive.damage.set_unbreakable_zone(condition)

    def set_output(self):
        return {'U':True,  "d":True, "Pressure" : True}
        
pb = Cylindre_axi(Acier)
Solve(pb, compteur = 20, TFin=TFin, scheme = "fixed", dt = pas_de_temps)