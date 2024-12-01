from models.nsga2 import NSGA2
from scipy.spatial import distance
from utils.utils import Utility
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np

class Metrics:
    def __init__(self):
        self.hypervolume = []
        self.inverted_generational_distance = []
        self.util = Utility()


    def true_pareto_front(self, solution_mofs_rfga, solution_nsga):
        """
            Permet de determiner le veritable front de pareto
        """
        suffle_solution = [sol for sol in solution_mofs_rfga if sol.rank == 1]
        suffle_solution.extend(solution_nsga) # ajouter le contenu de solution nsga2
        solution = NSGA2.fast_non_dominated_sort_v2(suffle_solution)

        true_pareto_front = [y for x in solution for  y in x if y.rank ==1 ]
        
        return true_pareto_front

    def calculate_igd(self, true_pareto_front, solution):
        """
            Permet de calculer la distance generational inversée
        """
        distances = []
        for sol in true_pareto_front:
            distances.append(min(distance.euclidean(sol.fitness, x.fitness) for x in solution))
        
        return np.sqrt(np.sum(distances))/len(true_pareto_front)

    def calculate_couverture(self, A,B):
        cpt = 0
        is_dominated = False
        for sol_a in A:
            for sol_b in B:
                if NSGA2.dominance2(sol_a.fitness,sol_b.fitness):
                    is_dominated = True
                    break
            if is_dominated:
                cpt +=1
                is_dominated = False
        return cpt/len(B)

    def hypervolume(self,front_pareto, reference_point):
        """
            Permet de calculer l'hypervolume
        """
        pass
    
    def calculate_metrics(self, df_name,true_pareto_front, nsga2_path, mofs_rfga_path):

        ind_nsga2 =  self.util.read_solution_i(nsga2_path, "nsga2")
        ind_mofs_rfga = self.util.read_solution_i(mofs_rfga_path, "mofs_rfga")
        ind_mofs_rfga = [x for x in ind_mofs_rfga if x.rank == 1]

        igd3_rf  = self.calculate_igd(true_pareto_front, ind_mofs_rfga)
        igd3  = self.calculate_igd(true_pareto_front, ind_nsga2)

        print("igd mofs3",igd3_rf)

        print("igd2 nsga2_3",igd3)
        #self.util.visualize_error_size(ind_mofs_rfga, "MOFS-RFGA", df_name, "red")
        #self.util.visualize_error_size(ind_nsga2, "NSGA-2", df_name, "Violet")

