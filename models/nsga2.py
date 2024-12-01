import numpy as np
import pandas as pd
import random
from utils.utils import Utility
from .Objectif_function import Objectif_Function
from .individual import Individual


class NSGA2:
    def __init__(self , X, y):
        self.X = X
        self.y = y
        self.population = []
        self.objectif_fct = Objectif_Function(X, y)
        self.utility = Utility()

    def evaluate(self,population):
        """
           Cette methode permet d'evaluer la population
        """
        for x in population:
            if np.sum(x) >0 and len(x) == self.X.shape[1]:
                self.population.append(Individual(x, self.objectif_fct.get_objectif_values(x)))
                #print("-----------------------------------------------------------")

    def dominance(self,x1, x2):
        """
        Permet de verifier la dominance au sens de Pareto
        On dit que le vecteur x1 domine le vecteur x2 si :
            x1 est au moins aussi bon que x2 dans tous les objectifs
            x1 est strictement meilleur que x2 dans au moins un objectif.
        """
        if((x1[0] < x2[0] and x1[1] < x2[1])  or (x1[0] <= x2[0] and x1[1] < x2[1]) or (x1[0] < x2[0] and x1[1] <= x2[1])):
            return True
        else:
            return False

    @staticmethod
    def dominance2(x1, x2):
        if((x1[0] < x2[0] and x1[1] < x2[1])  or (x1[0] <= x2[0] and x1[1] < x2[1]) or (x1[0] < x2[0] and x1[1] <= x2[1])):
            return True
        else:
            return False

    
    @staticmethod
    def fast_non_dominated_sort_v2( set_of_individuals):
        """ 
            Cette méthode permet d'effectuer le tri non dominé,
            ca particularité est qu elle est statique
        """
        fronts = []
        pop_copy = set_of_individuals.copy()
        for indiv in set_of_individuals:
            indiv.dominated_by = []
            indiv.rank =0

        for i, p in enumerate(pop_copy):
            for q in pop_copy[i + 1:]:
                if NSGA2.dominance2(p.fitness, q.fitness):
                    p.dominated_by.append(q)
                elif NSGA2.dominance2(q.fitness, p.fitness):
                    p.np += 1

            if p.np == 0:
                p.rank = 1
                fronts.append([p])

        k = 1
        while fronts[k - 1]:
            next_front = []
            for p in fronts[k - 1]:
                for q in p.dominated_by:
                    q.np -= 1
                    if q.np == 0:
                        q.rank = k + 1
                        next_front.append(q)

            if len(next_front) == 0:
                break 
            fronts.append(next_front)
            k += 1

        return fronts

    def fast_non_dominated_sort(self):
        """ 
            Cette méthode permet d'effectuer le tri non dominé
        """
        fronts = []
        pop_copy = self.population.copy()
        for indiv in self.population:
            indiv.dominated_by = []

        for i, p in enumerate(pop_copy):
            for q in pop_copy[i + 1:]:
                if self.dominance(p.fitness, q.fitness):
                    p.dominated_by.append(q)
                elif self.dominance(q.fitness, p.fitness):
                    p.np += 1

            if p.np == 0:
                p.rank = 1
                fronts.append([p])

        k = 1
        while fronts[k - 1]:
            next_front = []
            for p in fronts[k - 1]:
                for q in p.dominated_by:
                    q.np -= 1
                    if q.np == 0:
                        q.rank = k + 1
                        next_front.append(q)

            if len(next_front) == 0:
                break 
            fronts.append(next_front)
            k += 1

        return fronts

        

    def crowding_distance(self,front):
        nb_indiv_in_front = len(front)  #Nombre d'individus dans le front
        distances = np.zeros(nb_indiv_in_front)
        
        for o_i in range(2): #On parcourt les objectifs
            sorted_front = sorted(front, key=lambda ind: ind.fitness[o_i]) # On trie les individus du front selon l'objectif o_i
            min_obj_value = sorted_front[0].fitness[o_i] # On recupere la valeur de l'objectif o_i du premier individu
            max_obj_value = sorted_front[-1].fitness[o_i] # On recupere la valeur de l'objectif o_i du dernier individu
            
            distances[0] = distances[-1] = float("inf") # Tous les individus aux extremites ont une distance infinie
            # On calcule la distance de chaque individu aux individus qui l'entourent
            for i in range(1, nb_indiv_in_front - 1):
                if max_obj_value - min_obj_value == 0:
                    distances[i] += float("inf")
                else:
                    distances[i] += (sorted_front[i + 1].fitness[o_i] - sorted_front[i - 1].fitness[o_i]) / (max_obj_value - min_obj_value)
        
        for i, individu in enumerate(front):
            individu.crowding_distance = distances[i]

        #for ind in front:
            #print("individu",ind.fitness,"distance",ind.crowding_distance)
        return front


    def selection_environnemental(self,fronts, max_population_size):
        new_population = []
    
        for front in fronts:
            
            add_front = []
            for individual in front:
                add_front.append(individual)
        
            if len(new_population) + len(add_front) <= max_population_size:
                new_population.extend(add_front)
            else: # Si la taille de la nouvelle population dépasse la taille maximale 
                espace_restant = max_population_size - len(new_population)
                # Trier le front par ordre décroissant de distance de foule et de rang en priorité
                add_front.sort(key=lambda ind: (ind.crowding_distance, -ind.rank), reverse=True)
                new_population.extend(add_front[:espace_restant]) # permet de gerer l equilibre entre intensification et diversification
                break  # Sortir de la boucle, quand la taille maximale est atteinte
        
        #print("taille: ",len(new_population))
        #for ind in new_population:
            #print("individu",ind.fitness,"rank",ind.rank,"distance",ind.crowding_distance)
        return new_population

    def EnvironmentalSelection(self, R , max_population_size):
        P = []
        new_solution = []
        self.evaluate(R) #Evaluation de la population
        front = self.fast_non_dominated_sort() #On fait le fast non dominated sort
        front2 =[]
        for fr in front: 
            if(len(fr) > 0):
                front2.append(self.crowding_distance(fr)) #On calcule la distance de foule de chaque individu dans le front
        new_solution= self.selection_environnemental(front2 , max_population_size) #On selectionne les individus qui vont former la nouvelle population
        
        for sol in new_solution:
            P.append(sol.indiv)
        
        return P, new_solution