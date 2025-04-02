"""
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""
from math import sqrt, exp
try:
    from jax.numpy import searchsorted, clip, array
    from jax import vmap, jit
except Exception:
    print("JAX has not been loaded therefore tabulated law cannot be used")
    
def Dataframe_to_array(df):
    T_list = array(df.index.values)
    J_list = array(df.columns.astype(float).values)
    P_list = array(df.values)
    return T_list, J_list, P_list
    

class Material:
    def __init__(self, rho_0, C_mass, eos_type, dev_type, dico_eos, dico_devia, **kwargs):
        """
        Création du matériau à l'étude.
        Parameters
        ----------
        rho_0 : Float or Expression, masse volumique initiale M.L^{-3}
        C_mass : Function ou Float, capacité thermique massique en J.K^{-1}.kg^{-1} (= M.L^{2}.T^{-2}.K^{-1})
        eos_type : String, type d'équation d'état souhaitée.
        dev_type : String, type d'équation déviatorique souhaitée.
        dico_eos : Dictionnaire, dico contenant les paramètres nécessaires
                                    à la création du modèle d'équation d'état.
        dico_devia : Dictionnaire, dico contenant les paramètres nécessaires
                                    à la création du modèle de comportement déviatorique.
        **kwargs : Paramètres optionnels supplémentaires utilisé pour la détonique
        """
        self.rho_0 = rho_0
        self.C_mass = C_mass
        self.eos_type = eos_type      
        self.dev_type = dev_type
        self.eos = self.eos_selection(self.eos_type)(dico_eos)
        self.devia = self.deviatoric_selection(self.dev_type)(dico_devia)
        self.celerity = self.eos.celerity(rho_0)
       
        self.e_activation = kwargs.get("e_activation", None)
        self.kin_pref = kwargs.get("kin_pref", None)
        
        print("La capacité thermique vaut", self.C_mass)        
        print("La masse volumique vaut", self.rho_0)
        print("La célérité des ondes élastique est", self.celerity)
        
    def eos_selection(self, eos_type):
        """
        Retourne le nom de la classe associée au modèle d'EOS choisi.
        Parameters
        ----------
        dev_type : String, nom du modèle d'équation d'état.
        Raises
        ------
        ValueError, erreur si un comportement déviatorique inconnu est demandé.
        """
        if eos_type == "IsotropicHPP":
            return IsotropicHPP_EOS
        elif eos_type in ["U1", "U2", "U3", "U4", "U5", "U7", "U8"]:
            return U_EOS
        elif eos_type == "Vinet":        
            return Vinet_EOS
        elif eos_type == "JWL":        
            return JWL_EOS
        elif eos_type == "MACAW":        
            return MACAW_EOS
        elif eos_type == "MG":
            return MG_EOS
        elif eos_type == "xMG":
            return xMG_EOS
        elif eos_type == "PMG":
            return PMG_EOS
        elif eos_type == "GP":
            return GP_EOS
        elif eos_type == "NewtonianFluid":
            return NewtonianFluid_EOS
        elif eos_type == "Tabulated":
            return Tabulated_EOS
        else:
            raise ValueError("Equation d'état inconnue")
        
    def deviatoric_selection(self, dev_type):
        """
        Retourne le nom de la classe associée au modèle déviatorique retenu.
        Parameters
        ----------
        dev_type : String, nom du modèle déviatorique parmi:
                            None, IsotropicHPP, NeoHook, MooneyRivlin.
        Raises
        ------
        ValueError, erreur si un comportement déviatorique inconnu est demandé.
        """
        if dev_type == None:
            return None_deviatoric
        elif dev_type in["IsotropicHPP", "NeoHook", "Hypoelastic"]:
            return IsotropicHPP_deviatoric
        elif dev_type == "MooneyRivlin":        
            return MooneyRivlin_deviatoric
        elif dev_type == "NeoHook_Transverse":
            return NeoHook_Transverse_deviatoric
        elif dev_type == "Lu_Transverse":
            return Lu_Transverse_deviatoric
        elif dev_type == "Anisotropic":        
            return Anisotropic_deviatoric
        else:
            raise ValueError("Comportement déviatorique inconnu")        
        
class IsotropicHPP_EOS:
    def __init__(self, dico):
        """
        Défini un objet possédant une équation d'état élastique isotrope HPP
        Parameters
        ----------
        Dictionnaire contenant:
        E : Float, module de Young (en Pa ou MPa).
        nu : Float, coefficient de Poisson
        alpha : Float, coefficient de dilatation thermique en K^{-1}
        """
        try:
            self.E = dico["E"]
            self.nu = dico["nu"]
            self.alpha = dico["alpha"]
        except KeyError:
            raise ValueError("Le matériau n'est pas correctement défini")
            
        self.kappa = self.E / 3. / (1 - 2 * self.nu)
        print("Le coefficient de poisson est", self.nu)
        print("Le module de compressibilité du matériau est", self.kappa)
        print("Le module de Young du matériau est", self.E)
        print("Le coefficient d'expansion thermique vaut", self.alpha)
        
    def celerity(self, rho_0):
        """
        Renvoie la célérité des ondes élastique dans un milieu élastique linéaire       
        """
        return sqrt(self.E / rho_0)
    
class U_EOS:
    def __init__(self, dico):
        """
        Défini un objet possédant une équation d'état hyper-élastique isotrope 
        à un coefficient.

        Parameters
        ----------
        kappa : Float, module de compressibilité
        alpha : Float, coefficient de dilatation thermique en K^{-1}
        """
        try:
            self.kappa = dico["kappa"]
            self.alpha = dico["alpha"]
        except KeyError:
            raise ValueError("Le matériau n'est pas correctement défini")
            
        print("Le module de compressibilité du matériau est", self.kappa)
        print("Le coefficient d'expansion thermique vaut", self.alpha)
        
    def celerity(self, rho_0):
        """
        Renvoie une estimation de la célérité des ondes élastique 
        dans un milieu hyper-élastique.       
        """
        return sqrt(self.kappa / rho_0)
    
class Vinet_EOS:
    def __init__(self, dico):
        """
        Défini un matériau suivant une loi d'état de Vinet.
        Parameters
        ----------
        iso_T_K0 : Float, rigidité isotherme dans le modèle de Vinet.
        T_dep_K0 : Float, rigidité pondérant T dans le modèle de Vinet.
        iso_T_K1 : Float, coefficient isotherme dans le modèle de Vinet.
        T_dep_K1 : Float, coefficient pondérant T dans le modèle de Vinet.
        """
        try:
            self.iso_T_K0 = dico["iso_T_K0"]
            self.T_dep_K0 = dico["T_dep_K0"]
            self.iso_T_K1 = dico["iso_T_K1"]
            self.T_dep_K1 = dico["T_dep_K1"]
        except KeyError:
            raise ValueError("La loi d'état de Vinet n'est pas correctement définie")
            
        print("La rigidité isotherme K0 est", self.iso_T_K0)
        print("Le coefficient isotherme K1 est", self.iso_T_K1)
        
    def celerity(self, rho_0):
        """
        Renvoie la célérité des ondes élastique dans le milieu
        """
        return (self.iso_T_K0 / rho_0)**(1./2)
    
class JWL_EOS:
    def __init__(self, dico):
        """
        Défini un matériau suivant une loi d'état JWL.

        Parameters
        ----------
        A : TYPE
            DESCRIPTION.
        R1 : TYPE
            DESCRIPTION.
        B : TYPE
            DESCRIPTION.
        R2 : TYPE
            DESCRIPTION.
        w : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        try:
            self.A = dico["A"]
            self.R1 = dico["R1"]
            self.B = dico["B"]
            self.R2 = dico["R2"]
            self.w = dico["w"]
        except KeyError:
            raise ValueError("La loi d'état JWL n'est pas correctement définie")
                
        print("Le coefficient A vaut", self.A)
        print("Le coefficient R1 vaut", self.R1)
        print("Le coefficient B vaut", self.B)
        print("Le coefficient R2 vaut", self.R2)
        print("Le coefficient w vaut", self.w)
        
    def celerity(self, rho_0):
        return sqrt((self.A * self.R1 * exp(-self.R1) + self.B * self.R2 * exp(-self.R2)) / rho_0)
    
class MACAW_EOS:
    def __init__(self, dico):
        try:
            #Pour la partie froide
            self.A = dico["A"]
            self.B = dico["B"]
            self.C = dico["C"]
            #Pour la partie chaude
            self.eta = dico["eta"]
            self.vinf = dico["vinf"]
            self.rho0 = dico["rho0"]
            self.theta0 = dico["theta0"]
            self.a0 = dico["a0"]
            self.m = dico["m"]
            self.n = dico["n"]
            self.Gammainf = dico["Gammainf"]
            self.Gamma0 = dico["Gamma0"]
            self.cvinf = dico["cvinf"]
        except KeyError:
            raise ValueError("La loi d'état MACAW n'est pas correctement définie")
                
        print("Le coefficient A vaut", self.A)
        print("Le coefficient B vaut", self.B)
        print("Le coefficient C vaut", self.C)
        print("Le coefficient eta vaut", self.eta)
        print("Le coefficient theta0 vaut", self.theta0)
        print("Le coefficient a0 vaut", self.a0)
        print("Le coefficient m vaut", self.m)
        print("Le coefficient n vaut", self.n)
        print("Le coefficient Gammainf vaut", self.Gammainf)
        print("Le coefficient Gamma0 vaut", self.Gamma0)
        
    def celerity(self, rho_0):
        kappa = self.A * (self.B - 1./2 * self.C + (self.B + self.C)**2)
        print("Le module de compressibilité froid vaut", kappa)
        return sqrt( kappa/ rho_0)
    
        
class MG_EOS:
    def __init__(self, dico):
        """
        Défini un matériau suivant une loi d'état de Mie-Gurneisen identique
        à ESTHER

        Parameters
        ----------
        C : Float, Coefficient linéaire.
        D : Float, coefficient quadratique.
        S : Float, coefficient cubique.
        gamma0 : Float, coefficient de Gruneisen.
        """
        try:
            self.C = dico["C"]
            self.D = dico["D"]
            self.S = dico["S"]
            self.gamma0 = dico["gamma0"]
        except KeyError:
            raise ValueError("La loi d'état de Mie-Gruneisen n'est pas correctement définie")
        print("La coefficient linéaire", self.C)
        print("La coefficient quadratique vaut", self.D)
        print("La coefficient cubique vaut", self.S)
        print("Le coefficient Gamma0", self.gamma0)
        
        
    def celerity(self, rho_0):
        return sqrt(self.C / rho_0)
            
class xMG_EOS:
    def __init__(self, dico):
        """
        Défini un matériau suivant une loi d'état de Mie-Gurneisen.
        Parameters
        ----------
        c0 : TYPE
            DESCRIPTION.
        gamma0 : TYPE
            DESCRIPTION.
        s1 : TYPE
            DESCRIPTION.
        s2 : TYPE
            DESCRIPTION.
        s3 : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.
        """
        try:
            self.c0 = dico["c0"]
            self.gamma0 = dico["gamma0"]
            self.s1 = dico["s1"]
            self.s2 = dico["s2"]
            self.s3 = dico["s3"]
            self.b = dico["b"]
        except KeyError:
            raise ValueError("La loi d'état de Mie-Gruneisen n'est pas correctement définie")

        print("Le coefficient gamma0 est", self.gamma0)
        print("Le coefficient s1", self.s1)
        print("Le coefficient s2", self.s2)
        print("Le coefficient s3", self.s3)
        print("Le coefficient b", self.b)
        print("Une estimation de la célérité des ondes élastique est", self.c0)
        
    def celerity(self, rho_0):
        return self.c0
        
class PMG_EOS:
    def __init__(self, dico):
        """
        Défini un matériau suivant une loi d'état de Mie-Gurneisen.
        Parameters
        ----------
        Pa : TYPE
            DESCRIPTION.
        Gamma0 : TYPE
            DESCRIPTION.
        D : TYPE
            DESCRIPTION.
        S : TYPE
            DESCRIPTION.
        H : TYPE
            DESCRIPTION.
        """
        try:
            self.Pa = dico["Pa"]
            self.Gamma0 = dico["Gamma0"]
            self.D = dico["D"]
            self.S = dico["S"]
            self.H = dico["H"]
            self.c0 = dico["c0"]
        except KeyError:
            raise ValueError("La loi d'état de Mie-Gruneisen n'est pas correctement définie")
        
        print("La pression atmosphérique est", self.Pa)
        print("Le coefficient Gamma0", self.Gamma0)
        print("Le coefficient D", self.D)
        print("Le coefficient S", self.S)
        print("Le coefficient H", self.H)
        print("Le coefficient c0", self.c0)
        
class GP_EOS:
    def __init__(self, dico):
        """
        Défini les paramètres d'un gaz suivant la loi des gaz parfaits. 
        Parameters
        ----------
        gamma : Float, coefficient polytropique du gaz.
        e_max : fFoat, estimation de l'énergie interne massique maximale (ne sert que pour estimation CFL)
        """
        try:
            self.gamma = dico["gamma"]
            self.e_max = dico["e_max"]
        except KeyError:
            raise ValueError("La loi d'état du gaz parfait n'est pas correctement définie")

        print("Le coefficient polytropique vaut", self.gamma)
        print("Une estimation de la température maximale est", self.e_max)
        
    def celerity(self, rho_0):
        """
        Renvoie une estimation de la célérité des ondes accoustiques
        """
        return sqrt(self.gamma * (self.gamma - 1) * self.e_max)
    
class NewtonianFluid_EOS:
    def __init__(self, dico):
        """
        Défini un objet possédant des 
        caractéristiques mécanique d'un fluide Newtobien

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        chiT : TYPE
            DESCRIPTION.
        """
        try:
            self.k = dico["k"]
            self.alpha = dico["alpha"]
            self.chiT = dico["chiT"]
        except KeyError:
            raise ValueError("La loi d'état du fluide Newtonien n'est pas correctement définie")
        
        print("La viscosité de volumique est", self.k)
        print("Le coefficient de conductivité thermique vaut", self.alpha)
        print("La compréssibilité à température constante", self.chiT)
        
    def celerity(self, rho_0):
        return sqrt(1/(self.chiT * rho_0))
    
class Tabulated_EOS:
    def __init__(self, dico):
        """
        Défini un objet régit par une eos tabulée
        Parameters
        ----------
        c0 : Float, estimation de la célérité des ondes élastique
        """
        self.c0 = dico.get("c0")
        if "Dataframe" in dico:
            self.T_list, self.J_list, self.P_list = Dataframe_to_array(dico.get("Dataframe"))
        else:
            try:
                self.T_list = dico.get("T")
                self.J_list = dico.get("J")
                self.P_list = dico.get("P")
            except KeyError:
                raise ValueError("La loi d'état tabulée n'est pas correctement définie")
        
        self.set_tabulated_interpolator()

    def set_tabulated_interpolator(self):
        def find_index(x, xs):
            return clip(searchsorted(xs, x, side='right') - 1, 0, len(xs) - 2)

        def interpoler_2d(x, y, x_grid, y_grid, values):
            i = find_index(x, x_grid)
            j = find_index(y, y_grid)
            
            x1, x2 = x_grid[i], x_grid[i+1]
            y1, y2 = y_grid[j], y_grid[j+1]
            
            fx = (x - x1) / (x2 - x1)
            fy = (y - y1) / (y2 - y1)
            
            v11, v12 = values[i, j], values[i, j+1]
            v21, v22 = values[i+1, j], values[i+1, j+1]
            
            return (1-fx)*(1-fy)*v11 + fx*(1-fy)*v21 + (1-fx)*fy*v12 + fx*fy*v22

        def interpoler_jax(T, J):
            return interpoler_2d(T, J, self.T_list, self.J_list, self.P_list)
        
        self.tabulated_interpolator = jit(vmap(interpoler_jax, in_axes=(0, 0)))
    
    
    def celerity(self, rho_0):
        return self.c0   

class None_deviatoric:
    def __init__(self, dico):
        """
        Défini un comportement déviatorique nul(simulation purement hydro)
        """
        pass

class NewtonianFluid_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique pour un fluide
        Newtonien.
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        try:
            self.mu = dico["mu"]
        except KeyError:
            raise ValueError("Le comportement déviatorique d'un fluide Newtonien \
                             n'est pas correctement défini")

        
class IsotropicHPP_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique élastique isotrope HPP.
        Les paramètres sont identiques pour une loi déviatorique de type Néo-Hookéenne.
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        self.mu = dico.get("mu", None)
        if self.mu == None:
            try:
                self.E = dico["E"]
                self.nu = dico["nu"]
            except KeyError:
                raise ValueError("Le comportement déviatorique n'est pas correctement défini")
            print("Le module de Young du matériau est", self.E)
            print("Le coefficient de poisson est", self.nu)
            self.mu = self.E / 2. / (1 + self.nu)
        print("Le module de cisaillement vaut", self.mu)
        
class MooneyRivlin_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique de MooneyRivlin
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        try:
            self.mu = dico["mu"]
            self.mu_quad = dico["mu_quad"]
        except KeyError:
            raise ValueError("Le comportement déviatorique de Mooney-Rivlin n'est pas correctement défini")
        print("Le module de cisaillement vaut", self.mu)
        print("Le module de cisaillement quadratique", self.mu_quad)

class NeoHook_Transverse_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique isotrope transverse
        Néo-Hookée
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        try:
            self.mu = dico["mu"]
            self.mu_T = dico["mu_T"]
        except KeyError:
            raise ValueError("Le comportement déviatorique isotrope transverse \
                                 Néo-Hookéen n'est pas correctement définie")
        print("Le module de cisaillement vaut", self.mu)
        print("Le module de cisaillement transverse", self.mu_T)

        
class Lu_Transverse_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique isotrope transverse
        de Lu
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        try:
            self.k2 = dico["k2"]
            self.k3 = dico["k3"]
            self.k4 = dico["k4"]
            self.c = dico["c"]
        except KeyError:
            raise ValueError("Le comportement déviatorique de Lu n'est pas correctement défini")
        print("Le coefficient de dilatation linéique vaut", self.k2)
        print("Le module de cisaillement transverse vaut", self.k3)
        print("Le module de cisaillement dans le plan transverse vaut", self.k4)
        
class Anisotropic_deviatoric:
    def __init__(self, dico):
        """
        Défini les paramètres pour un comportement déviatorique anisotrope quelconque
        Parameters
        ----------
        dico : Dictionnaire, dico contenant les paramètres nécessaires
        """
        try:
            self.C = dico["C"]
            self.f_func_coeffs = dico.get("f_func", None)
            # self.init_kappa = dico["kappa_0"]
        except KeyError:
        #     try:
        #         compliance = dico["param"]
        #         anisotropy = dico["anisotropie"]
        #     except KeyError:
            raise ValueError("Le comportement anisotrope n'est pas correctement défini")
    