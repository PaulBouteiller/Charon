# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
try:
    import gmsh
except Exception:
    print("Gmsh has not been loaded therefore cannot be used")
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from mpi4py.MPI import COMM_WORLD
from dolfinx.mesh import create_interval, create_unit_interval, create_rectangle, CellType
from numpy import array, linspace
from ..utils.default_parameters import default_fem_degree

def create_1D_mesh(x_left, x_right, N_el):
    return create_interval(COMM_WORLD, N_el, [array(x_left), array(x_right)])

def create_2D_rectangle(x_bl, y_bl, x_tr, y_tr, Nx, Ny):
    return create_rectangle(COMM_WORLD, [(x_bl, y_bl), (x_tr, y_tr)], [Nx, Ny], CellType.quadrilateral)

def init_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    gdim = 2
    rank = 0
    gmsh.model.add("Model")
    geom = gmsh.model.geo
    return gdim, rank, geom

def return_mesh(model, comm, rank, gdim, quad, write):
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    if quad:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(default_fem_degree())
    gmsh.model.mesh.optimize("Netgen")
    domain, meshtags, facets = model_to_mesh(model, comm, rank, gdim=gdim)
    gmsh.finalize()
    if write:
        with XDMFFile(COMM_WORLD, "mesh.xdmf", "w") as infile:
            infile.write_mesh(domain)
    return domain, meshtags, facets


def generate_perforated_plate(W, H, R, mesh_size):
    gdim, model_rank, geom = init_gmsh()
    if COMM_WORLD.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, W, H, tag=1)
        hole = gmsh.model.occ.addDisk(W / 2, H / 2, 0, R, R, zAxis=[0, 0, 1],
            xAxis=[0.0, 1.0, 0.0],)
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, hole)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(gdim, 1, "Plate")

        try:
            field_tag = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field_tag, "VIn", min(mesh_size))
            gmsh.model.mesh.field.setNumber(field_tag, "VOut", max(mesh_size))
            gmsh.model.mesh.field.setNumber(field_tag, "XMin", 0)
            gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
            gmsh.model.mesh.field.setNumber(field_tag, "YMin", H / 2 - 1.2 * R)
            gmsh.model.mesh.field.setNumber(field_tag, "YMax", H / 2 + 1.2 * R)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        except:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim)
    
def axi_sphere(Ri, Re, N_theta, N_radius, tol_dyn = 0, quad = True, write = False):
    gdim, model_rank, geom = init_gmsh()
    center = geom.add_point(tol_dyn, 0, 0)
    p1 = geom.add_point(Ri, 0, 0)
    p2 = geom.add_point(Re, 0, 0)
    p3 = geom.add_point(tol_dyn, Re, 0)
    p4 = geom.add_point(tol_dyn, Ri, 0)
    
    x_radius = geom.add_line(p1, p2)
    outer_circ = geom.add_circle_arc(p2, center, p3)
    y_radius = geom.add_line(p3, p4)
    inner_circ = geom.add_circle_arc(p4, center, p1)
    boundary = geom.add_curve_loop([x_radius, outer_circ, y_radius, inner_circ])
    surf = geom.add_plane_surface([boundary])

    geom.synchronize()  
    
    for curve in [outer_circ, inner_circ]:
        gmsh.model.mesh.setTransfiniteCurve(curve, N_theta)
    for curve in [x_radius, y_radius]:
        gmsh.model.mesh.setTransfiniteCurve(curve, N_radius)   
    gmsh.model.mesh.setTransfiniteSurface(surf)

    gmsh.model.addPhysicalGroup(gdim, [surf], 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [x_radius], 1, name="bottom")
    print("La partie où z = 0 se voit affecter le drapeau", 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [y_radius], 2, name="left")
    print("La partie où r = 0 se voit affecter le drapeau", 2)
    gmsh.model.addPhysicalGroup(gdim - 1, [outer_circ], 3, name="outer")
    print("La surface extérieure se voit affecter le drapeau", 3)
    gmsh.model.addPhysicalGroup(gdim - 1, [inner_circ], 4, name="inner")
    print("La surface intérieure se voit affecter le drapeau", 4)
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim, quad, write)

def axi_double_coquille(Ri, Re, R_mid, N_theta, Nr_int, Nr_out, tol_dyn = 0, quad = True):
    gdim, model_rank, geom = init_gmsh()
    center = geom.add_point(tol_dyn, 0, 0)
    p1 = geom.add_point(Ri, 0, 0)
    p2 = geom.add_point(Re, 0, 0)
    p3 = geom.add_point(tol_dyn, Re, 0)
    p4 = geom.add_point(tol_dyn, Ri, 0)
    p5 = geom.add_point(R_mid, 0, 0)
    p6 = geom.add_point(tol_dyn, R_mid, 0)

    radius_int_x = geom.add_line(p1, p5)
    mid_circ = geom.add_circle_arc(p5, center, p6)
    radius_int_y = geom.add_line(p6, p4)
    inner_circ = geom.add_circle_arc(p4, center, p1)
    boundary_int = geom.add_curve_loop([radius_int_x, mid_circ, radius_int_y, inner_circ])
    surf_int = geom.add_plane_surface([boundary_int])
    
    radius_out_x = geom.add_line(p2, p5)
    radius_out_y = geom.add_line(p6, p3)
    outer_circ = geom.add_circle_arc(p3, center, p2)

    
    boundary_out = geom.add_curve_loop([radius_out_x, mid_circ, radius_out_y, outer_circ])
    surf_out = geom.add_plane_surface([boundary_out])
    

    geom.synchronize()

    for curve in [outer_circ, inner_circ, mid_circ]:
        gmsh.model.mesh.setTransfiniteCurve(curve, N_theta)
    for curve in [radius_int_x, radius_int_y]:
        gmsh.model.mesh.setTransfiniteCurve(curve, Nr_int)
    for curve in [radius_out_x, radius_out_y]:
        gmsh.model.mesh.setTransfiniteCurve(curve, Nr_out)   
    gmsh.model.mesh.setTransfiniteSurface(surf_int)
    gmsh.model.mesh.setTransfiniteSurface(surf_out)

    gmsh.model.addPhysicalGroup(gdim, [surf_int], 1)
    gmsh.model.addPhysicalGroup(gdim, [surf_out], 2)
    gmsh.model.addPhysicalGroup(gdim - 1, [radius_int_x, radius_out_x], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [radius_int_y, radius_out_y], 2, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [outer_circ], 3, name="outer")
    gmsh.model.addPhysicalGroup(gdim - 1, [inner_circ], 4, name="inner")
    # gmsh.model.addPhysicalGroup(gdim - 1, [mid_circ], 5, name="mid")
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim, quad)

def double_rectangle(xbl, ybl, h1, h2, L, Nx, Nh1, Nh2, quad = True):
    gdim, model_rank, geom = init_gmsh()
    p1 = geom.add_point(xbl, ybl, 0)
    p2 = geom.add_point(xbl+h1, ybl, 0)
    p3 = geom.add_point(xbl+h1, ybl+L, 0)
    p4 = geom.add_point(xbl, ybl+L, 0)
    p5 = geom.add_point(xbl+h1+h2, ybl, 0)
    p6 = geom.add_point(xbl+h1+h2, ybl+L, 0)
    

    bottom_left = geom.add_line(p1, p2)
    mid = geom.add_line(p2, p3)
    top_left = geom.add_line(p3, p4)
    left = geom.add_line(p4, p1)
    
    bottom_right = geom.add_line(p5, p2)
    top_right = geom.add_line(p3, p6)
    right = geom.add_line(p6, p5)
    
    boundary_1 = geom.add_curve_loop([bottom_left, mid, top_left, left])
    surf_1 = geom.add_plane_surface([boundary_1])
    
    boundary_2 = geom.add_curve_loop([bottom_right, mid, top_right, right])
    surf_2 = geom.add_plane_surface([boundary_2])

    geom.synchronize()

    gmsh.model.addPhysicalGroup(gdim, [surf_1], 1)
    gmsh.model.addPhysicalGroup(gdim, [surf_2], 2)
    gmsh.model.addPhysicalGroup(gdim - 1, [left], 1, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [bottom_left], 2, name="left_bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [bottom_right], 3, name="left_top")
    
    for curve in [left, mid, right]:
        gmsh.model.mesh.setTransfiniteCurve(curve, Nx)
        
    for curve in [bottom_left, top_left]:
        gmsh.model.mesh.setTransfiniteCurve(curve, Nh1)
    for curve in [bottom_right, top_right]:
        gmsh.model.mesh.setTransfiniteCurve(curve, Nh2)
    gmsh.model.mesh.setTransfiniteSurface(surf_1)    
    gmsh.model.mesh.setTransfiniteSurface(surf_2)    
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim, quad)


def quarter_perforated_plate(width, heigth, radius, hsize, quad = True):
    gdim, model_rank, geom = init_gmsh()
    center = geom.add_point(0, 0, 0)
    p1 = geom.add_point(radius, 0, 0)
    p2 = geom.add_point(width, 0, 0)
    p3 = geom.add_point(width, heigth, 0)
    p4 = geom.add_point(0, heigth, 0)
    p5 = geom.add_point(0, radius, 0)

    bottom = geom.add_line(p1, p2)
    right = geom.add_line(p2, p3)
    top = geom.add_line(p3, p4)
    left = geom.add_line(p4, p5)
    inner_circ = geom.add_circle_arc(p5, center, p1)

    boundary = geom.add_curve_loop([bottom, right, top, left, inner_circ])
    surf = geom.add_plane_surface([boundary])

    geom.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    gmsh.model.addPhysicalGroup(gdim, [surf], 1)
    # gmsh.model.addPhysicalGroup(gdim - 1, [bottom], 1, name="bottom")
    # gmsh.model.addPhysicalGroup(gdim - 1, [left], 2, name="left")
    # gmsh.model.addPhysicalGroup(gdim - 1, [top], 3, name="outer")
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim, quad)


def raffinement_maillage(h, Largeur, ratio):
    boxField = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(boxField, "VIn", h / ratio)  # Taille à l'intérieur de la boîte
    gmsh.model.mesh.field.setNumber(boxField, "VOut", h)  # Taille à l'extérieur de la boîte
    gmsh.model.mesh.field.setNumber(boxField, "XMin", Largeur / 2.2)
    gmsh.model.mesh.field.setNumber(boxField, "XMax", Largeur)
    gmsh.model.mesh.field.setNumber(boxField, "YMin", Largeur * (1 / 2 - 0.1))
    gmsh.model.mesh.field.setNumber(boxField, "YMax", Largeur * (1 / 2 + 0.1))
    gmsh.model.mesh.field.setNumber(boxField, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(boxField, "ZMax", 0)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(boxField)
    
    
def broken_unit_square(N_largeur, Nh, Largeur = 1, tol = None, raff = 0, quad = True):
    if tol == None:
        tol = Largeur / 1e3
    gdim, model_rank, geom = init_gmsh() 
    h_size = Largeur/N_largeur
    #paramètre de l'ellipse
    pb1 = geom.add_point(0, 0, 0)
    pb2 = geom.add_point(Largeur/2, 0, 0)
    pb3 = geom.add_point(Largeur, 0, 0)
    
    pt1 = geom.add_point(0, Largeur, 0)
    pt2 = geom.add_point(Largeur/2, Largeur, 0)
    pt3 = geom.add_point(Largeur, Largeur, 0)
    
    pm1 = geom.add_point(0, Largeur/2 - tol, 0)
    pm1bis = geom.add_point(0, Largeur/2 + tol, 0)
    pm2 = geom.add_point(Largeur/2, Largeur/2, 0)
    pm3 = geom.add_point(Largeur, Largeur/2, 0)
    

    left_bottom = geom.add_line(pm1, pb1)
    bottom_left = geom.add_line(pb1, pb2)
    bottom_right = geom.add_line(pb2, pb3)
    right_bottom = geom.add_line(pb3, pm3)
    right_mid = geom.add_line(pm3, pm2)
    bottom_lip = geom.add_line(pm2, pm1)
    
    left_top = geom.add_line(pm1bis, pt1)
    top_left = geom.add_line(pt1, pt2)
    top_right = geom.add_line(pt2, pt3)
    right_top = geom.add_line(pt3, pm3)
    top_lip = geom.add_line(pm2, pm1bis)
    
    boundary_1 = geom.add_curve_loop([left_bottom, bottom_left, bottom_right, right_bottom, right_mid, bottom_lip])
    surf_1 = geom.add_plane_surface([boundary_1])
    
    boundary_2 = geom.add_curve_loop([left_top, top_left, top_right, right_top, right_mid, top_lip])
    surf_2 = geom.add_plane_surface([boundary_2])

    geom.synchronize()

    gmsh.model.addPhysicalGroup(gdim, [surf_1], 1)
    gmsh.model.addPhysicalGroup(gdim, [surf_2], 2)
    gmsh.model.addPhysicalGroup(gdim - 1, [bottom_left, bottom_right], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [left_bottom, left_top], 2, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [top_left, top_right], 3, name="top")
    
    if raff==0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_size)

    else:
        raffinement_maillage(h_size, Largeur, raff)
    
    geom.synchronize()
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim, quad)


def hotspot_plate(W, H, R, mesh_size):
    gdim, model_rank, geom = init_gmsh()
    if COMM_WORLD.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, W, H, tag=1)
        hole = gmsh.model.occ.addDisk(W / 2, H / 2, 0, R, R, zAxis=[0, 0, 1],
            xAxis=[0.0, 1.0, 0.0],)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(gdim, 1, "Plate")

        try:
            field_tag = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field_tag, "VIn", min(mesh_size))
            gmsh.model.mesh.field.setNumber(field_tag, "VOut", max(mesh_size))
            gmsh.model.mesh.field.setNumber(field_tag, "XMin", 0)
            gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
            gmsh.model.mesh.field.setNumber(field_tag, "YMin", H / 2 - 1.2 * R)
            gmsh.model.mesh.field.setNumber(field_tag, "YMax", H / 2 + 1.2 * R)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        except:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    return return_mesh(gmsh.model, COMM_WORLD, model_rank, gdim)