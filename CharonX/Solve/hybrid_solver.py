"""
hybrid_solver.py - Implémentation d'un solveur linéaire hybride pour FEniCSx

Ce module contient les classes nécessaires pour implémenter un solveur hybride
qui alterne entre résolution directe et itérative. L'idée est de préconditionner
le solveur itératif avec la factorisation de la matrice obtenue avec le solveur direct.

@author: bouteillerp, 
"""

from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from time import time

class SolverStatistics:
    """
    Classe pour collecter et afficher les statistiques du solveur hybride.
    """
    def __init__(self):
        # Compteurs généraux
        self.total_solves = 0
        self.direct_solves = 0  # Nombre de fois où une résolution directe a été utilisée
        self.iterative_solves = 0  # Nombre de fois où une résolution itérative a été utilisée
        self.precond_reuses = 0  # Nombre de fois où le préconditionneur a été réutilisé
        self.precond_updates = 0  # Nombre de fois où le préconditionneur a été mis à jour
        self.failures = 0  # Nombre d'échecs de résolution

        # Historique des itérations et des temps
        self.iteration_history = []  # Nombre d'itérations pour chaque résolution
        self.time_history = []  # Temps pour chaque résolution
        self.direct_time_history = []  # Temps pour les résolutions directes
        self.iterative_time_history = []  # Temps pour les résolutions itératives
        
        # Statistiques de convergence
        self.convergence_reasons = {}  # Décompte des raisons de convergence/divergence
        
        # Statistiques temporelles globales
        self.total_time = 0
        self.direct_time = 0
        self.iterative_time = 0
    
    def record_solve(self, is_direct, iterations, solve_time, converged_reason):
        """
        Enregistre les statistiques d'une résolution
        
        Parameters
        ----------
        is_direct : bool True si une résolution directe a été utilisée
        iterations : int Nombre d'itérations effectuées
        solve_time : float Temps de résolution en secondes
        converged_reason : int Code de raison de convergence/divergence PETSc
        """
        self.total_solves += 1
        self.iteration_history.append(iterations)
        self.time_history.append(solve_time)
        
        # Mettre à jour le compteur de raisons de convergence
        reason_str = f"{converged_reason}"
        if reason_str in self.convergence_reasons:
            self.convergence_reasons[reason_str] += 1
        else:
            self.convergence_reasons[reason_str] = 1
        
        if is_direct:
            self.direct_solves += 1
            self.direct_time_history.append(solve_time)
            self.direct_time += solve_time
            self.precond_updates += 1
        else:
            self.iterative_solves += 1
            self.iterative_time_history.append(solve_time)
            self.iterative_time += solve_time
            
        self.total_time += solve_time
    
    def record_precond_reuse(self):
        """
        Enregistre la réutilisation d'un préconditionneur
        """
        self.precond_reuses += 1
    
    def record_precond_update(self):
        """
        Enregistre la mise à jour d'un préconditionneur
        """
        self.precond_updates += 1
    
    def record_failure(self):
        """
        Enregistre un échec de résolution
        """
        self.failures += 1
    
    def get_summary(self):
        """
        Retourne un résumé des statistiques sous forme de dictionnaire
        
        Returns
        -------
        dict Résumé des statistiques
        """
        avg_iterations = sum(self.iteration_history) / len(self.iteration_history) if self.iteration_history else 0
        avg_time = self.total_time / self.total_solves if self.total_solves > 0 else 0
        avg_direct_time = self.direct_time / self.direct_solves if self.direct_solves > 0 else 0
        avg_iterative_time = self.iterative_time / self.iterative_solves if self.iterative_solves > 0 else 0        
        return {
            "total_solves": self.total_solves,
            "direct_solves": self.direct_solves,
            "iterative_solves": self.iterative_solves,
            "precond_reuses": self.precond_reuses,
            "precond_updates": self.precond_updates,
            "failures": self.failures,
            "avg_iterations": avg_iterations,
            "avg_time": avg_time,
            "avg_direct_time": avg_direct_time,
            "avg_iterative_time": avg_iterative_time,
            "total_time": self.total_time
        }
    
    def print_summary(self, detailed=False):
        """
        Affiche un résumé des statistiques
        
        Parameters
        ----------
        detailed : bool, optional
            Si True, affiche des statistiques détaillées
        """
        summary = self.get_summary()
        
        print("\n===== HYBRID SOLVER STATISTICS =====")
        print(f"Total solves: {summary['total_solves']}")
        print(f"  - Direct solves: {summary['direct_solves']} ({summary['direct_solves']/summary['total_solves']*100:.1f}%)")
        print(f"  - Iterative solves: {summary['iterative_solves']} ({summary['iterative_solves']/summary['total_solves']*100:.1f}%)")
        print(f"Preconditioner reuses: {summary['precond_reuses']}")
        print(f"Preconditioner updates: {summary['precond_updates']}")
        if summary['failures'] > 0:
            print(f"Failures: {summary['failures']}")
        
        print("\nTime statistics:")
        print(f"  - Total time: {summary['total_time']:.3f} seconds")
        print(f"  - Average time per solve: {summary['avg_time']:.3f} seconds")
        if summary['direct_solves'] > 0:
            print(f"  - Average direct solve time: {summary['avg_direct_time']:.3f} seconds")
        if summary['iterative_solves'] > 0:
            print(f"  - Average iterative solve time: {summary['avg_iterative_time']:.3f} seconds")
        
        if detailed:
            print("\nDetailed statistics:")
            print(f"  - Average iterations: {summary['avg_iterations']:.1f}")
            print("\nConvergence reasons:")
            for reason, count in self.convergence_reasons.items():
                print(f"  - {reason}: {count} times ({count/self.total_solves*100:.1f}%)")
            
            if len(self.iteration_history) > 0:
                print("\nIteration distribution:")
                max_iter = max(self.iteration_history)
                for i in range(max_iter + 1):
                    count = self.iteration_history.count(i)
                    if count > 0:
                        print(f"  - {i} iterations: {count} times ({count/len(self.iteration_history)*100:.1f}%)")


# Paramètres par défaut pour les solveurs
_direct_solver_default = {"solver": "mumps", "type": "cholesky", "blr": True}
_iterative_solver_default = {"solver": "cg", "maximum_iterations": 100}
_hybrid_parameters_default = {"iteration_switch": 10, "user_switch": True}


class HybridSolver:
    """
    Classe générique implémentant un solveur hybride.

    Un solveur direct est utilisé pour une première factorisation de matrice.
    Les résolutions suivantes utilisent un solveur itératif avec la matrice
    précédemment factorisée comme préconditionneur. Ce préconditionneur est
    conservé même si l'opérateur change, sauf si le nombre d'itérations ksp
    dépasse le seuil défini `iteration_switch`. Dans ce cas, la prochaine 
    résolution mettra à jour l'opérateur et effectuera une nouvelle factorisation.
    """

    def __init__(self, parameters, direct_solver, iterative_solver, 
                 log, timings, collect_stats=True, comm = MPI.COMM_WORLD):
        """
        Initialise le solveur hybride.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional Le communicateur MPI à utiliser (par défaut MPI.COMM_WORLD)
        parameters : dict, optional Paramètres pour le comportement hybride
        direct_solver : dict, optional Configuration du solveur direct
        iterative_solver : dict, optional Configuration du solveur itératif
        log : bool, optional Activer le logging
        timings : bool, optional Activer le suivi des temps d'exécution
        collect_stats : bool, optional Activer la collecte de statistiques
        """
        # Initialisation des paramètres
        self.comm = comm
        self.parameters = parameters if parameters is not None else _hybrid_parameters_default.copy()
        self.direct_solver = direct_solver if direct_solver is not None else _direct_solver_default.copy()
        self.iterative_solver = iterative_solver if iterative_solver is not None else _iterative_solver_default.copy()
        
        # Initialisation du solveur KSP
        self.ksp = self.get_ksp()
        self.pc = self.ksp.getPC()
        
        # Configuration des options PETSc
        self._set_petsc_options()
        
        # État du solveur
        self.reuse_preconditioner = False
        self.log = log
        self.timings = timings
        # self.tic = None
        
        # Forcer la première résolution à être directe
        self.is_direct_solve = True
        # Configurer pour une résolution directe initiale
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        self.ksp.setFromOptions()

        
        # Statistiques
        self.collect_stats = collect_stats
        if collect_stats:
            self.stats = SolverStatistics()
            # Définir un moniteur pour collecter les résidus à chaque itération
            self.residual_history = []
            self.ksp.setMonitor(self._ksp_monitor)
    
    def _ksp_monitor(self, ksp, iteration, residual_norm):
        """
        Fonction de suivi pour KSP qui enregistre les normes résiduelles
        
        Parameters
        ----------
        ksp : PETSc.KSP L'objet KSP
        iteration : int Le numéro d'itération actuel
        residual_norm : float La norme du résidu
        """
        if hasattr(self, 'residual_history'):
            self.residual_history.append(residual_norm)

    def _set_petsc_options(self):
        """Configure les options PETSc pour le solveur"""
        # Options du solveur itératif
        self.ksp.setType(self.iterative_solver["solver"])
        
        # Définir le nombre maximum d'itérations
        if "maximum_iterations" in self.iterative_solver:
            self.ksp.setTolerances(max_it=self.iterative_solver["maximum_iterations"])
        
        # Options du préconditionneur (solveur direct)
        self.pc.setType(self.direct_solver["type"])
        
        # Définir le type de solveur direct
        opts = PETSc.Options()
        opts["pc_factor_mat_solver_type"] = self.direct_solver["solver"]
        
        # Options spécifiques pour MUMPS avec compression BLR
        if self.direct_solver["solver"] == "mumps" and self.direct_solver.get("blr", False):
            opts["mat_mumps_icntl_35"] = 1  # Active BLR
            opts["mat_mumps_cntl_7"] = 1e-10  # Tolérance BLR
        
        # Paramètres généraux
        self.ksp.setErrorIfNotConverged(True)
        self.ksp.setNormType(PETSc.KSP.NormType.NATURAL)
        self.pc.setReusePreconditioner(True)
        
        # Appliquer les options
        self.ksp.setFromOptions()
    
    def print_log(self, message):
        """Affiche un message de log si le logging est activé"""
        if self.log and (self.comm.Get_rank() == 0):
            print(message)
    
    def print_timings(self, message):
        """Affiche le temps écoulé depuis le dernier tic"""
        if self.timings and self.tic is not None:
            print(f"{message} {time() - self.tic}")
    
    def start_timer(self):
        """Démarre le timer"""
        self.tic = time()
    
    def _update_infos(self):
        """Affiche des informations sur la résolution"""
        self.print_log(f"Converged in {self._it} iterations.")
        if self.reuse_preconditioner:
            self.print_log("Preconditioner will be reused on next solve.")
        else:
            self.print_log("Next solve will be a direct one with matrix factorization.")

    def preconditioner_choice(self):
        """
        Détermine si le préconditionneur doit être réutilisé en fonction
        du nombre d'itérations.
        """
        self._it = self.ksp.getIterationNumber()
        self._converged = self.ksp.getConvergedReason()
        
        if self._converged > 0:
            # Réutiliser si le nombre d'itérations est inférieur au seuil
            self.reuse_preconditioner = ((self._it < self.parameters["iteration_switch"]) 
                                         and self.parameters.get("user_switch", True))
        else:
            # Ne pas réutiliser si la résolution a échoué
            self.reuse_preconditioner = False
            
    def update_pc(self):
        """Configure la réutilisation du préconditionneur"""
        self.preconditioner_choice()
        self.pc.setReusePreconditioner(self.reuse_preconditioner)
        
        if self.log:
            self._update_infos()

        # Enregistrer la réutilisation ou la mise à jour du préconditionneur pour les statistiques
        if self.collect_stats:
            if self.reuse_preconditioner:
                self.stats.record_precond_reuse()
            else:
                self.stats.record_precond_update()

    def get_ksp(self):
        """
        Méthode à surcharger pour obtenir l'objet KSP.
        Returns
        -------
        PETSc.KSP Objet KSP pour la résolution
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les classes dérivées")
    
    def get_solver(self):
        """
        Méthode à surcharger pour obtenir le solveur.
        Returns
        -------
        object
            Le solveur (LinearProblem ou NewtonSolver)
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les classes dérivées")

    def solve(self, *args, **kwargs):
        """
        Résout le système avec la stratégie hybride.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments à passer au solveur
        """
        self.start_timer()
        
        # Afficher le type de solveur au début pour debugging
        current_solver_type = self.ksp.getType()
        self.print_log(f"Starting solve with solver type: {current_solver_type}")
        self.print_log(f"Direct solve: {self.is_direct_solve}, Preconditioner reuse: {self.reuse_preconditioner}")
        
        if self.collect_stats:
            self.residual_history = []
        
        try:
            # Tentative de résolution normale
            solver = self.get_solver()
            solver.solve(*args, **kwargs)
            solve_time = time() - self.tic
            self.print_timings("Solve time:")
            
            # Enregistrer les statistiques
            if self.collect_stats:
                self._it = self.ksp.getIterationNumber()
                self._converged = self.ksp.getConvergedReason()
                self.stats.record_solve(self.is_direct_solve, self._it, solve_time, self._converged)
                self.print_log(f"Solve completed with {self._it} iterations, using {current_solver_type} solver")
                    
        except Exception as e:
            # En cas d'échec, forcer une résolution directe
            self.print_log(f"Error caught: {str(e)}")
            self.print_log("Switching to direct solver...")
            
            # Configurer pour résolution directe, on mentionne ici à PETSc de n'utiliser 
            # que le préconditionneur pour résoudre le système, sans itérations Krylov supplémentaires. 
            # Combiné avec un préconditionneur de type factorisation directe (comme MUMPS, LU, Cholesky, etc.), cela donne un solveur direct.
            self.ksp.setType(PETSc.KSP.Type.PREONLY)
            self.ksp.setFromOptions()
            
            # Nouvelle tentative
            self.start_timer()
            try:
                self.is_direct_solve = True
                self.get_solver().solve(*args, **kwargs)
                solve_time = time() - self.tic
                self.print_timings("Direct solve time:")
                
                # Enregistrer les statistiques
                if self.collect_stats:
                    self._it = 1  # Une seule itération pour le solveur direct
                    self._converged = self.ksp.getConvergedReason()
                    self.stats.record_solve(self.is_direct_solve, self._it, solve_time, self._converged)
            except Exception as e2:
                # Échec même avec le solveur direct
                self.print_log(f"Direct solver also failed: {str(e2)}")
                if self.collect_stats:
                    self.stats.record_failure()
                raise
                
            # Rétablir le solveur itératif pour la prochaine résolution
            self.ksp.setType(self.iterative_solver["solver"])
            self.reuse_preconditioner = True
            self.pc.setReusePreconditioner(self.reuse_preconditioner)
            self.ksp.setFromOptions()
         
        # Si c'était une résolution directe, passer au solveur itératif pour la prochaine fois
        if self.is_direct_solve:
            self.print_log("First direct solve completed, switching to iterative solver for next solves")
            self.ksp.setType(self.iterative_solver["solver"])
            self.ksp.setFromOptions()
            self.is_direct_solve = False  # Pour les appels suivants
        
        # Mettre à jour le préconditionneur pour la prochaine résolution
        self.update_pc()
        
        # Retourner les informations sur les itérations
        it_direct = 0 if self.reuse_preconditioner else 1
        return [self._it, it_direct]
        
    def get_statistics(self):
        """
        Retourne l'objet de statistiques
        
        Returns
        -------
        SolverStatistics L'objet contenant les statistiques du solveur
        """
        if not hasattr(self, 'collect_stats') or not self.collect_stats:
            self.print_log("Statistics collection is disabled.")
            return None
        return self.stats
    
    def print_statistics(self, detailed=False):
        """
        Affiche les statistiques de résolution
        
        Parameters
        ----------
        detailed : bool, optional Si True, affiche des statistiques détaillées
        """
        if not hasattr(self, 'collect_stats') or not self.collect_stats:
            self.print_log("Statistics collection is disabled.")
            return
        
        if self.comm.Get_rank() == 0:
            self.stats.print_summary(detailed)


class HybridLinearSolver(HybridSolver):
    """
    Version hybride d'un solveur linéaire pour les problèmes de la forme Ax = b.
    """
    def __init__(self, problem, comm, parameters=None, direct_solver=None, 
                 iterative_solver=None, log=True, timings=False, collect_stats=True):
        """
        Initialise le solveur linéaire hybride.
        
        Parameters
        ----------
        problem : dolfinx.fem.petsc.LinearProblem Le problème linéaire à résoudre
        comm : mpi4py.MPI.Comm, optional Le communicateur MPI à utiliser (si None, essaie de l'extraire de problem)
        parameters : dict, optional Paramètres pour le comportement hybride
        direct_solver : dict, optional Configuration du solveur direct
        iterative_solver : dict, optional Configuration du solveur itératif
        log : bool, optional Activer le logging
        timings : bool, optional Activer le suivi des temps d'exécution
        collect_stats : bool, optional Activer la collecte de statistiques
        """
        self.problem = problem
        super().__init__(comm, parameters, direct_solver, iterative_solver, log, timings, collect_stats)

    def get_ksp(self):
        """
        Retourne l'objet KSP sous-jacent au LinearProblem
        Returns
        -------
        PETSc.KSP Objet KSP du LinearProblem
        """
        return self.problem.solver
    
    def get_solver(self):
        """
        Retourne le solveur (LinearProblem)
        
        Returns
        -------
        dolfinx.fem.petsc.LinearProblem Le problème linéaire
        """
        return self.problem


class HybridNewtonSolver(HybridSolver):
    """
    Version hybride d'un solveur de Newton pour les problèmes non-linéaires.
    """
    def __init__(self, nl_problem, comm, parameters=None, direct_solver=None, 
                 iterative_solver=None, log=True, timings=False, collect_stats=True):
        """
        Initialise le solveur non-linéaire hybride.
        
        Parameters
        ----------
        nl_problem : dolfinx.nls.petsc.NonlinearProblem  Le problème non-linéaire à résoudre
        comm : mpi4py.MPI.Comm, optional Le communicateur MPI à utiliser (si None, essaie de l'extraire de nl_problem)
        parameters : dict, optional Paramètres pour le comportement hybride
        direct_solver : dict, optional Configuration du solveur direct
        iterative_solver : dict, optional Configuration du solveur itératif
        log : bool, optional Activer le logging
        timings : bool, optional Activer le suivi des temps d'exécution
        collect_stats : bool, optional
            Activer la collecte de statistiques
        """
        self.nl_problem = nl_problem
        # Créer le solveur de Newton avec le communicateur
        self.solver = NewtonSolver(comm, nl_problem)
        
        # Appel au constructeur de la classe parente
        super().__init__(comm, parameters, direct_solver, iterative_solver, log, timings, collect_stats)
        
    def get_ksp(self):
        """
        Retourne l'objet KSP sous-jacent au solveur de Newton
        Returns
        -------
        PETSc.KSP Objet KSP du solveur de Newton
        """
        return self.solver.krylov_solver
    
    def get_solver(self):
        """
        Retourne le solveur (NewtonSolver)
        
        Returns
        -------
        dolfinx.nls.petsc.NewtonSolver Le solveur de Newton
        """
        return self.solver


def create_linear_solver(a, L, u, bcs=None, comm=None, solver_type="hybrid", parameters=None, 
                         direct_solver=None, iterative_solver=None, log=True, timings=False, 
                         collect_stats=True):
    """
    Fonction utilitaire pour créer un solveur linéaire.
    
    Parameters
    ----------
    a : ufl.Form Forme bilinéaire définissant la matrice A
    L : ufl.Form Forme linéaire définissant le vecteur b
    u : dolfinx.fem.Function Fonction solution
    bcs : list of dolfinx.fem.DirichletBC or dolfinx.fem.DirichletBC, optional
        Condition(s) aux limites. Peut être une seule condition ou une liste de conditions.
    comm : mpi4py.MPI.Comm, optional Le communicateur MPI à utiliser (si None, extrait de u.function_space.mesh)
    solver_type : str, optional Type de solveur ("default" ou "hybrid")
    parameters : dict, optional Paramètres pour le comportement hybride
    direct_solver : dict, optional Configuration du solveur direct
    iterative_solver : dict, optional Configuration du solveur itératif
    log : bool, optional Activer le logging
    timings : bool, optional Activer le suivi des temps d'exécution
    collect_stats : bool, optional Activer la collecte de statistiques
    Returns
    -------
    solver : dolfinx.fem.petsc.LinearProblem or HybridLinearSolver Le solveur créé
    """
    problem = LinearProblem(a, L, u=u, bcs=bcs)
    
    # Si comm n'est pas fourni, tenter de l'extraire de u
    if comm is None and u is not None:
        comm = u.function_space.mesh.comm
    
    if solver_type == "default":
        return problem
    elif solver_type == "hybrid":
        return HybridLinearSolver(problem, parameters, direct_solver, 
                                 iterative_solver, log, timings, collect_stats, comm)
    else:
        raise ValueError(f"Type de solveur inconnu: {solver_type}. Utilisez 'default' ou 'hybrid'.")