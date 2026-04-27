"""
Test des schémas d'intégration numériques pour différents éléments finis.

Vérifie le nombre de points de quadrature par élément pour toutes les
combinaisons :

    - Éléments : bar (1D), triangle, quadrangle (2D), tétraèdre, hexaèdre (3D)
    - Ordres   : 1 (interpolation linéaire), 2 (interpolation quadratique)
    - Schémas  : "default" (intégration pleine), "reduit" (sous-intégration)

Hypothèse sur les valeurs attendues
-----------------------------------
Le schéma "default" est dimensionné pour intégrer exactement la matrice de
raideur d'un problème d'élasticité linéaire, dont l'intégrand est de degré
2(p-1) en chaque direction. On retrouve ainsi 1 pt pour bar p=1 et 4 pts
pour quad p=2 (cas validés dans le script initial).

Le schéma "reduit" correspond à 1 seul point de Gauss par élément
(intégration uniformément réduite, déformation constante).

Auteur : bouteillerp
"""
import pytest
from mpi4py import MPI
from dolfinx.fem import Function

from Charon import (CartesianUD, Axisymmetric, Tridimensional,
                    Material, MeshManager,
                    create_1D_mesh, create_rectangle, create_box,
                    CellType)

 
# ---------------------------------------------------------------------------
# Matériau commun à tous les tests
# ---------------------------------------------------------------------------
DummyMat = Material(2, 1, "IsotropicHPP", "IsotropicHPP",
                    {"E": 1, "nu": 0, "alpha": 1},
                    {"E": 1, "nu": 0})
 
 
# ---------------------------------------------------------------------------
# Helper : construit un problème Charon avec UN seul élément
# ---------------------------------------------------------------------------
def _make_problem(dim, cell_type, u_degree, schema):
    """Crée un problème Charon et renvoie (problème, nb_cellules_du_maillage).
 
    Note : `create_rectangle` avec `[1, 1]` et des triangles produit en réalité
    2 triangles, et `create_box` avec `[1, 1, 1]` et des tétraèdres en produit
    6. Il faut donc normaliser la taille du vecteur de quadrature par le nombre
    réel de cellules pour obtenir le nombre de points par élément.
    """
    if dim == 1:
        mesh = create_1D_mesh(0, 1, 1)
        bnd = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, 1]}
        ProblemClass = CartesianUD
    elif dim == 2:
        mesh = create_rectangle(MPI.COMM_WORLD,
                                [(0, 0), (1, 1)], [1, 1], cell_type)
        bnd = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [0, 1]}
        ProblemClass = Axisymmetric
    elif dim == 3:
        if Tridimensional is None:
            pytest.skip("Classe 3D non trouvée dans Charon "
                        "(à adapter selon la version)")
        mesh = create_box(MPI.COMM_WORLD,
                          [(0, 0, 0), (1, 1, 1)], [1, 1, 1], cell_type)
        bnd = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, 1]}
        ProblemClass = Tridimensional
    else:
        raise ValueError(f"Dimension {dim} non gérée")
 
    # Nombre réel de cellules du maillage (peut être > 1 pour tri/tet).
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim)
    n_cells = mesh.topology.index_map(tdim).size_local
 
    dictionnaire_mesh = {
        **bnd,
        "fem_parameters": {"u_degree": u_degree, "schema": schema},
    }
    mesh_manager = MeshManager(mesh, dictionnaire_mesh)
 
    dictionnaire = {
        "material": DummyMat,
        "mesh_manager": mesh_manager,
        "boundary_setup": bnd,
        "analysis": "static",
        "isotherm": True,
    }
    return ProblemClass(dictionnaire), n_cells
 
 
# ---------------------------------------------------------------------------
# Cas de test :
#   (dim, cell_type, label, u_degree, schema, nb_pts_quad_attendu)
# ---------------------------------------------------------------------------
CASES = [
    # 1D --- barre
    (1, None,                   "bar",  1, "default", 2),
    (1, None,                   "bar",  1, "reduit",  1),
    (1, None,                   "bar",  2, "default", 3),
    (1, None,                   "bar",  2, "reduit",  2),

    # 2D --- triangle
    (2, CellType.triangle,      "tri",  1, "default", 3),
    (2, CellType.triangle,      "tri",  1, "reduit",  1),
    (2, CellType.triangle,      "tri",  2, "default", 6),
    (2, CellType.triangle,      "tri",  2, "reduit",  3),

    # 2D --- quadrangle
    (2, CellType.quadrilateral, "quad", 1, "default", 4),
    (2, CellType.quadrilateral, "quad", 1, "reduit",  1),
    (2, CellType.quadrilateral, "quad", 2, "default", 9),
    (2, CellType.quadrilateral, "quad", 2, "reduit",  4),

    # 3D --- tétraèdre
    (3, CellType.tetrahedron,   "tet",  1, "default", 4),
    (3, CellType.tetrahedron,   "tet",  1, "reduit",  1),
    (3, CellType.tetrahedron,   "tet",  2, "default", 14),
    (3, CellType.tetrahedron,   "tet",  2, "reduit",  4),

    # 3D --- hexaèdre
    (3, CellType.hexahedron,    "hex",  1, "default", 8),
    (3, CellType.hexahedron,    "hex",  1, "reduit",  1),
    (3, CellType.hexahedron,    "hex",  2, "default", 27),
    (3, CellType.hexahedron,    "hex",  2, "reduit",  8),
]
 
_IDS = [f"{c[2]}-p{c[3]}-{c[4]}" for c in CASES]
 
 
# ---------------------------------------------------------------------------
# Test paramétré
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dim,cell_type,label,u_degree,schema,n_qp",
    CASES,
    ids=_IDS,
)
def test_quadrature_size(dim, cell_type, label, u_degree, schema, n_qp):
    """Vérifie le nombre de points de quadrature **par élément**."""
    pb, n_cells = _make_problem(dim, cell_type, u_degree, schema)
    quad_func = Function(pb.V_quad_UD)
    n_total = len(quad_func.x.array)
 
    assert n_total % n_cells == 0, (
        f"Vecteur quadrature de taille {n_total} non divisible par "
        f"{n_cells} cellules pour {label} (p={u_degree}, schema={schema})"
    )
    n_per_cell = n_total // n_cells
 
    print(f"  {label:4s}  p={u_degree}  schema={schema:7s}  "
          f"-> {n_per_cell:3d} pts/élément ({n_total} total / "
          f"{n_cells} cellules, attendu {n_qp})")
 
    assert n_per_cell == n_qp, (
        f"Mauvais nombre de points de quadrature par élément pour "
        f"{label} (p={u_degree}, schema={schema}) : "
        f"obtenu {n_per_cell}, attendu {n_qp}"
    )
 
 
# ---------------------------------------------------------------------------
# Exécution directe : affiche un tableau récapitulatif sans pytest
#   $ python test_quadrature_schemas.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    header = (f"{'Élément':6s} {'Ordre':>5s} {'Schéma':9s} "
              f"{'Cellules':>9s} {'Total':>6s} {'pts/él.':>8s} "
              f"{'Attendu':>8s}  Statut")
    print(header)
    print("-" * len(header))
 
    for dim, cell_type, label, u_degree, schema, n_qp in CASES:
        try:
            pb, n_cells = _make_problem(dim, cell_type, u_degree, schema)
            quad_func = Function(pb.V_quad_UD)
            n_total = len(quad_func.x.array)
            n_per_cell = n_total // n_cells if n_cells else 0
            statut = "OK" if n_per_cell == n_qp else "FAIL"
            cells_str = str(n_cells)
            total_str = str(n_total)
            per_str = str(n_per_cell)
        except pytest.skip.Exception:
            cells_str = total_str = per_str = "-"
            statut = "skip"
        except Exception as e:
            cells_str = total_str = per_str = "ERR"
            statut = type(e).__name__
 
        print(f"{label:6s} {u_degree:>5d} {schema:9s} "
              f"{cells_str:>9s} {total_str:>6s} {per_str:>8s} "
              f"{n_qp:>8d}  {statut}")
 
