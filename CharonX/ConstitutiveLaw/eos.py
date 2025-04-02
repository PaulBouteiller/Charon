"""
Created on Thu Jun 23 10:47:33 2022

@author: bouteillerp
"""
from ..utils.generic_functions import ppart, npart
try:
    from jax.numpy import array
except Exception:
    print("JAX has not been loaded therefore tabulated law cannot be used")

from dolfinx.fem import Function, Expression
from ufl import (tr, sym, ln, exp)


class EOS:
    def __init__(self, kinematic, quadrature):
        """
        Classe contenant toutes les équations d'état disponible du code CHARON.
        Le type d'équation d'état est entièrement déterminé lors de la création
        d'un objet de la classe Material.

        Parameters
        ----------
        kinematic : Objet de la classe Kinematic
        """
        self.kin = kinematic
        self.quad = quadrature
    
    def set_eos(self, v, J, T, T0, mat):
        """
        Renvoie l'expression de la pression
        Parameters
        ----------
        v : Function, champ des vitesses.
        J : Jacobien de la transformation.
        mat : Objet de la classe material.
        T : champ de température actuelle
        T0 : champ de température initiale

        Returns
        -------
        p : Pression du modèle
        """
        eos_param = mat.eos
        if mat.eos_type == "IsotropicHPP":
            p = -eos_param.kappa * (J - 1 - 3 * eos_param.alpha * (T-T0))
        elif mat.eos_type == "U1":
            p = -eos_param.kappa * (J - 1 - 3 * eos_param.alpha * (T-T0))
        elif mat.eos_type == "U5":
            p = -eos_param.kappa * (ln(J) - ln(1 + 3 * eos_param.alpha * (T - T0)))
        elif mat.eos_type == "U8":
            p = -eos_param.kappa/2 * (ln(J) - 1 / J - ln(1 + 3 * eos_param.alpha * (T - T0)) + 1 / (1 + eos_param.alpha * (T - T0)))
        elif mat.eos_type == "MG":
            p = self.MG_pressure(J, T, T0, mat)
        elif mat.eos_type == "xMG":
            p = self.xMG_pressure(J, T, T0, mat)
        elif mat.eos_type == "GP":
            p = self.GP_pressure(J, T, mat)
        elif mat.eos_type == "NewtonianFluid":
            p = self.fluid_pressure(v, J, T, T0, eos_param)
        elif mat.eos_type == "Vinet":
            p = self.Vinet_pressure(J, T, T0, eos_param)
        elif mat.eos_type == "JWL":
            p = self.JWL_pressure(J, T, mat)
        elif mat.eos_type == "MACAW":
            p = self.MACAW_pressure(J, T, mat)
        elif mat.eos_type == "Tabulated":
            V = self.quad.quadrature_space(["Scalar"])
            self.J_func = Function(V)
            self.J_expr = Expression(J, V.element.interpolation_points())
            self.T = T
            p = Function(V)
            p.x.array[:] = self.tabulated_pressure(eos_param)
        else:
            raise ValueError("Unknwon eos")
        return p
    
    def GP_pressure(self, J, T, mat):
        """
        Définition de la pression dans le cas du gaz parfait
        """
        return (mat.eos.gamma - 1) * mat.rho_0 / J * mat.C_mass * T
    
    def fluid_pressure(self, v, J, T, T0, mat):
        """
        Définition de la pression totale dans le cas du fluide Newtonien
        """  
        chiT = mat.chiT 
        alpha = mat.alpha
        thermo_p = -1 / chiT * ln(J) + alpha / chiT * (T-T0)
        return  -mat.k * tr(sym(self.kin.grad_3D(v))) + thermo_p
    
    def Vinet_pressure(self, J, T, T0, eos):
        """
        Définition de la pression pour le modèe de Vinet
        """
        K0 = eos.iso_T_K0 + eos.T_dep_K0 * (T - T0)
        K1 = eos.iso_T_K1 + eos.T_dep_K1 * (T - T0)
        return 3 * K0 * J**(-2./3) * (1 - J**(1./3)) * exp(3./2 * (K1 - 1) * (1 - J**(1./3)))

    def JWL_pressure(self, J, T, mat):
        """
        Définition de la pression pour le modèe de JWL
        """
        return mat.eos.A * exp(- mat.eos.R1 * J) + mat.eos.B * exp(- mat.eos.R2 * J) + mat.eos.w * mat.rho_0 / J * mat.C_mass * T
    
    def MACAW_pressure(self, J, T, mat):
        """
        Définition de la pression pour le modèe de Macaw
        """
        def thetav(theta0, v, vinf, gamma0, gammainf, m):
            eta = v/vinf
            beta = (gammainf - gamma0) / m
            theta = theta0*(eta)**(-gamma0)*((eta)**(-m) + 1)**(beta)
            return theta

        def dthetadv(theta0, v, vinf, gamma0, gammainf, m):
            eta=v/vinf
            th=thetav(theta0, v, vinf, gamma0, gammainf, m)
            dtheta=-(th/eta)*(gamma0+(gammainf-gamma0)/(1.+(eta**m)))
            return dtheta/vinf

        def av(a0,vinf,v,n):
            eta=v/vinf
            a=a0/(1.+(eta)**(-n))
            return a

        def dadv(a0,vinf,v,n):
            eta=v/vinf
            da=(n/eta)*a0*(eta**n)/((1.+eta**n)**2)
            return da/vinf

        def thermal_curve(rho,T,cvinf,a0,vinf,n,theta0,gamma0,gammainf,m):
            v=1./rho
            thetavloc=thetav(theta0,v,vinf,gamma0,gammainf,m)
            dthetadvloc=dthetadv(theta0,v,vinf,gamma0,gammainf,m)
            avloc=av(a0,vinf,v,n)
            dadvloc=dadv(a0,vinf,v,n)
    
            q0=dadvloc/3.-dthetadvloc/thetavloc
            q1=5*thetavloc*dadvloc/6.-avloc*dthetadvloc/6.
            q2=(thetavloc**2)*dadvloc/2.-avloc*thetavloc*dthetadvloc/2.
            pref=cvinf/((T+thetavloc)**3)
            thermal=pref*(q0*(T**4)+q1*(T**3)+q2*(T**2))
            return thermal

        def cold_curve(rho0,A,B,C,rho):
            eta=rho0/rho
            cold_a=A*eta**(-(B+1.))
            cold_b=exp(2.*C*(1.-eta**(3./2.))/3.)
            cold_c=C*eta**(3./2.)+B
            cold_d=A*(B+C)
            cold=cold_a*cold_b*cold_c-cold_d
            return cold

        rholoc = mat.eos.rho0/J
        pc = cold_curve(mat.eos.rho0,mat.eos.A,mat.eos.B,mat.eos.C,rholoc)
        pth=thermal_curve(rholoc,T,mat.eos.cvinf,mat.eos.a0,mat.eos.vinf,mat.eos.n,mat.eos.theta0,mat.eos.Gamma0,mat.eos.Gammainf,mat.eos.m)
        return pc+pth

    
    def MG_pressure(self, J, T, T0, mat):
        """
        Définition du terme de pression dans le modèle de Mie-Gruneisen
        """
        eos = mat.eos
        mu = 1/J - 1
        MG_pressure = eos.C * mu + eos.D * mu**2 + eos.S * mu**3 + mat.rho_0 / J * eos.gamma0 * (T - T0)
        return MG_pressure
    
    def xMG_pressure(self, J, T, T0, mat):
        """
        Définition du terme de pression dans le modèle de Mie-Gruneisen
        """
        eos = mat.eos
        mu = 1 / J - 1
        # Loi complète
        numerateur_pos = mat.rho_0 * eos.c0**2 * ppart(mu) * (1 + (1 - eos.gamma0 / 2) * mu - eos.b / 2 * mu**2)
        denominateur_pos = 1 - (eos.s1 - 1) * mu - eos.s2 * mu ** 2 * J - eos.s3 * mu**3 * J**2
        # thermique =  mat.rho_0 / J * (mat.gamma0 + mat.b * ppart(mu)) * T        
        #Loi réduite
        partie_neg = mat.rho_0 * eos.c0**2 * npart(mu)
        thermique =  mat.rho_0 / J * eos.gamma0 * T
        return  numerateur_pos/denominateur_pos + partie_neg + thermique     
    
    def tabulated_pressure(self, mat_eos):
        self.J_func.interpolate(self.J_expr)
        p = mat_eos.tabulated_interpolator(array(self.T.x.array), array(self.J_func.x.array))
        return p  
        # return p_analytique 