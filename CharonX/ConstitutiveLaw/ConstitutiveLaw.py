"""
Created on Mon Sep 26 18:06:31 2022

@author: bouteillerp
ConstitutiveLaw est le fichier principal gérant la loi de comportement dans CHARON
il appelle les sous fichiers elastic.py, plastic.py et damage.py pour éventuellement
définir un comportement plus complexe.
"""
from .eos import EOS
from .deviator import Deviator
from .plastic import HPPPlastic, FiniteStrainPlastic, JAXJ2Plasticity, JAXGursonPlasticity
from .damage import PhaseField, StaticJohnson, DynamicJohnson, InertialJohnson

from ufl import dot, Identity, dev, sym, tr, ln
from numpy import array, zeros
from dolfinx.fem import functionspace, Function
from ..utils.generic_functions import ppart, npart

class ConstitutiveLaw:
    def __init__(self, u, material, plastic_model, damage_model, multiphase, 
                 name, kinematic, quadrature, damping, is_hypo, relative_rho_0):
        """
        Parameters
        ----------
        u : Function, champ de déplacement.
        material : Objet de la classe Material.
        plastic_model : String, modèle de plasticité (HPP_Plasticity, Finite_Plasticity ou None)
        damage_model : String, modèle d'endommagement
        multiphase : Objet de la classe multiphase.
        name : String, nom du modèle mécanique.
        kinematic : Objet de la classe Kinematic.
        quad :  Objet de la classe quadrature.
        damping : Dic : dictionnaire contenant les paramètres d'amortissement
        relative_rho_0 : champ des masses volumiques initiales relatives
        """
        self.material = material
        self.eos = EOS(kinematic, quadrature)
        self.mesh = u.function_space.mesh
        self.deviator = Deviator(kinematic, name, quadrature, is_hypo)
        self.plastic_model = plastic_model
        self.damage_model = damage_model
        self.multiphase = multiphase
        self.kinematic = kinematic
        self.set_damping(damping)

        self.name = name
        self.relative_rho_0 = relative_rho_0
        if self.damage_model != None:
            self.damage = self.damage_class()(self.mesh, quadrature)
        if self.plastic_model != None:
            self.plastic = self.plastic_class(name)(u, material.devia.mu, name, kinematic, quadrature, self.plastic_model)

    def set_damping(self, damping):
        """
        Initialise les paramètres de la pseudo-viscosité
        """
        self.is_damping = damping["damping"]
        self.Klin = damping["linear_coeff"]
        self.Kquad = damping["quad_coeff"]
        self.correction = damping["correction"]
    
    def stress_3D(self, u, v, T, T0, J):
        """
        Calcul la contrainte tridimensionnelle totale, il s'agit d'une 
        moyenne pondérée des contraintes existant dans chacune des sous-phases.
        
        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ des vitesses.
        T : Function, champ de température actuelle
        T0 : Function, champ de température initiale
        J : Function, Jacobien de la transformation
        
        Returns
        -------
        sigma_3D : Function, contrainte tridimensionnelle réelle, inclus l'élasticité
        le multiphase et la plasticité mais pas l'endommagement.
        """

        if isinstance(self.material, list):
            self.p_list = []
            self.pseudo_p_list = []
            self.s_list = []
            n_mat = len(self.material)
            for i, mat in enumerate(self.material):
                p, pseudo_p, s = self.partial_stress_3D(u, v, T, T0, J, mat, self.relative_rho_0[i])
                self.p_list.append(p)
                self.pseudo_p_list.append(pseudo_p)
                self.s_list.append(s)
            self.p = sum(self.multiphase.c[i] * self.p_list[i] for i in range(n_mat))
            self.pseudo_p = sum(self.multiphase.c[i] * self.pseudo_p_list[i] for i in range(n_mat))
            self.s = sum(self.multiphase.c[i] * self.s_list[i] for i in range(n_mat))
        else:
            self.p, self.pseudo_p, self.s = self.partial_stress_3D(u, v, T, T0, J, self.material)
        return -(self.p + self.pseudo_p) * Identity(3) + self.s
    
    def partial_stress_3D(self, u, v, T, T0, J, mat, relative_rho_0 = 1):
        """
        Renvoie la contrainte tridimensionnelle de Cauchy associée au matériau 
        mat. Si le milieu est monophasique il s'agit de la vraie contrainte de Cauchy
        En cas de milieu multiphasique il s'agit de la contrainte dans le milieu mat.
        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ des vitesses.
        T : Function, champ de température actuelle
        T0 : Function, champ de température initiale
        J : Function, Jacobien de la transformation
        
        Returns
        -------
        sigma_3D : Function, contrainte tridimensionnelle dans le matériau mat
        """

        p = self.eos.set_eos(v, J * relative_rho_0, T, T0, mat)
        if mat.dev_type == "Hypoelastic":
            s = self.deviator.set_deviator(u, v, J, mat.devia.mu)
        elif self.plastic_model == "Finite_Plasticity":
            s = mat.devia.mu / J**(5./3) * dev(self.plastic.Be_trial())
        elif self.plastic_model =="J2_JAX":
            s = mat.devia.mu / J * dev(self.plastic.Be_bar_old_3D)
        else:
            s = self.deviator.set_elastic_dev(u, v, J, T, T0, mat)
        if self.plastic_model == "HPP_Plasticity":
            s -= self.plastic.plastic_correction(mat.devia.mu)
        if self.is_damping:
            pseudo_p = self.pseudo_pressure(v, mat, J)
        else:
            pseudo_p = 0
        return p, pseudo_p, s

    def pseudo_pressure(self, v, mat, J):
        """
        Construit la pseudo-viscosité pour la stabilisation

        Parameters
        ----------
        v : Function, champ des vitesses.
        mat : Objet de la classe material, matériau à l'étude.
        """
        V = functionspace(self.mesh, ("DG", 0))
        h_loc = Function(V)                
        tdim = self.mesh.topology.dim
        num_cells = self.mesh.topology.index_map(tdim).size_local
        h_local = zeros(num_cells)
        for i in range(num_cells):
            h_local[i] = self.mesh.h(tdim, array([i]))
        h_loc.x.array[:] = h_local
        
        div_v  = self.kinematic.div(v)
        lin_Q = self.Klin * mat.rho_0 * mat.celerity * h_loc * npart(div_v)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]: 
            quad_Q = self.Kquad * mat.rho_0 * h_loc**2 * npart(div_v) * div_v 
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
            quad_Q = self.Kquad * mat.rho_0 * h_loc**2 * dot(npart(div_v), div_v)
        if self.correction :
            lin_Q *= 1/J
            quad_Q *= 1/J
        return quad_Q - lin_Q

    def plastic_class(self, name):
        """
        Renvoie le nom de la classe plastique à appeler
        Parameters
        ----------
        name : String, nom du modèle mécanique.
        """
        if self.plastic_model == "HPP_Plasticity":
            return HPPPlastic
        elif self.plastic_model == "Finite_Plasticity":
            return FiniteStrainPlastic
        elif self.plastic_model == "J2_JAX":
            return JAXJ2Plasticity
        elif self.plastic_model == "JAX_Gurson":
            return JAXGursonPlasticity
        else:
            raise ValueError("This model do not exist, did you mean \
                             HPP_Plasticity or Finite_Plasticity ?")
    def damage_class(self):
        """
        Renvoie le nom de la classe endommagement à appeler
        """
        if self.damage_model == "PhaseField":
            return PhaseField
        elif self.damage_model == "Johnson":
            return StaticJohnson   
        elif self.damage_model == "Johnson_dyn":
            return DynamicJohnson
        elif self.damage_model == "Johnson_inertiel":
            return InertialJohnson
        else:
            raise ValueError("Unknown damage model")
         
    def set_plastic_driving(self):
        """
        Calcule la force motrice plastique en appelant la méthode de l'objet plastic.
        Si l'étude est élasto-plastique endommageable, cette force motrice est pondérée
        par la variable d'endommagement.
        """
        if self.plastic_model == "HPP_Plasticity":
            self.plastic.plastic_driving_force(self.s)
            if self.damage_model !=None:
                self.plastic.A *= self.damage.g_d
                
        elif self.plastic_model == "Finite_Plasticity":
            self.plastic.set_expressions()
            
    def set_damage_driving(self, u, J):
        """
        Initialise l'évolution de l'endommagement

        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ de vitesse.
        T : Function, champ de température actuelle.
        T0 : Function, champ de température initiale.
        J : Expression, jacobien de la transformation.
        """
        if self.damage_model in ["Johnson", "Johnson_dyn", "Johnson_inertiel"]:
            self.damage.set_p_mot(self.p)
        else:
            self.eHelm = self.Helmholtz_energy(u, J, self.material)
            self.damage.set_NL_energy(self.eHelm) 
            
    def Helmholtz_energy(self, u, J, mat):
        """
        Renvoie l'énergie libre volumique de Helmholtz

        Parameters
        ----------
        u : Function, champ de déplacement.
        J : Expression, jacobien de la transformation.
        mat : Objet de la classe material, matériau à l'étude.
        """
        if mat.eos_type == "IsotropicHPP":
            eps = sym(self.kinematic.grad_3D(u))
            E1 = tr(eps)
            psi_vol = mat.eos.kappa / 2 * E1 * ppart(E1)
        elif mat.eos_type == "U5":
            psi_vol = mat.eos.kappa * (J * ln(J) - J + 1)
        elif mat.eos_type == "U8":
            psi_vol = mat.eos.kappa / 2 * ln(J) * ppart(J-1)
        else:
            raise ValueError("Phase field analysis has not been implemented for this eos")
        if mat.dev_type == "IsotropicHPP": 
            psi_iso_vol = self.psi_isovol_HPP(u, mat.devia.mu)
        elif mat.dev_type == "NeoHook": 
            psi_iso_vol = self.psi_isovol_NeoHook(u, mat.devia.mu)
        elif mat.dev_type == None:
            psi_iso_vol = 0        
        else:
            raise ValueError("Phase field analysis has not been implemented for this deviatoric law")
            
        return psi_vol + psi_iso_vol
            
    def psi_isovol_HPP(self, u, mu):
        """
        Renvoie l'énergie libre isovolume de Helmholtz du modèle élastique
        linéaire dans l'hypothèse des petites perturbations.
        Parameters
        ----------
        u : Function, champ de déplacement.
        mu : Float, cooefficient de cisaillement
        """
        dev_eps = dev(sym(self.kinematic.grad_3D(u)))
        return 0
        # return  mu * tr(dot(dev_eps, dev_eps))
    
    def psi_isovol_NeoHook(self, u, mu):
        """
        Renvoie l'énergie libre isovolume de Helmholtz du modèle hyper-élastique
        Néo-Hookéen
        Parameters
        ----------
        u : Function, champ de déplacement.
        mu : Float, cooefficient de cisaillement
        """
        return mu * (self.kinematic.BBarI(u) - 3)