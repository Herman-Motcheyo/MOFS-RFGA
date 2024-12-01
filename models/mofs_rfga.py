import pandas as pd
import numpy as np

from utils.utils import Utility
from .nsga2 import NSGA2
from .Objectif_function import Objectif_Function
import random

class MOFS_RFGA:


    def __init__(self,X ,y, sc,D, N = 60, maxFES=200000 ):
        self.sc = sc #vecteur de poids fourni par l'algorithme filtre
        self.N = N #Taille de la population initiale
        self.maxFES = maxFES  # Nombre maximum d'évaluations de la fonction objectif
        self.X = X   
        self.y = y
        self.utility = Utility()
        self.nsga2 = NSGA2(X,y)
        self.objectif_function = Objectif_Function(X,y)
        self.D =D

    def initialisation(self):
        """
            Cette fonction permet d'initialiser la population initiale de taille N.

        return:
            list: Une liste contenant la population de départ initialisée.
        """
        P= []
        for _ in range(0,self.N):
            tmp_feature = np.zeros(len(self.sc), dtype=int)
            R = np.random.randint(low=1, high=self.D)
            #print(R)
            so_idx = self.binary_tournment_base_sc(R)
            tmp_feature[so_idx] = 1
            P.append(tmp_feature)
        return P

    
    def binary_tournment_base_sc(self, R):
        """
            Effectue un tournoi binaire pour sélectionner les indices de caractéristiques les plus performants.

                Cette fonction effectue R fois un tournoi binaire pour sélectionner les indices de caractéristiques les plus performants,
                en utilisant les scores de caractéristiques fournis dans `feature_scores`.

            Args:
                feature_scores : Une liste de scores des caractéristiques.
                R (int): Le nombre de tournois binaires à effectuer.

            Returns:
                list of int: Une liste contenant les indices sélectionnés des caractéristiques les plus performantes.

    
        """
        indices = np.arange(len(self.sc))
        selected_indices = []
        for _ in range(0,R):
            selected_index = np.random.choice(indices,size=2, replace=False)
            selected_indices.append(self.utility.find_max(self.sc,selected_index))
        return selected_indices



    def crossover_3_to_1(self,p1,p2,p3):
        """
            # Cette methode permet d effectuer le croissement entre 3 parents et retourne
             l'enfant  issu
        """
        offsprings = np.array([])
        for i in range(self.N):
            L1 = p1 & p2    #On cherche la position des genes qui sont communes
            L2 = p1 & p3
            L3 = p2 & p3
            O_i = np.logical_or(L1, np.logical_or(L2,L3)).astype(int) #enfant ayant les genes qui sont selectionnees plus de 2 fois
            S_3 = L1 & L2 & L3 # position des genes qui sont selectionees 3 fois chez le parent
            S_2 = O_i^S_3   # position des genes qui sont selectionees 3 fois chez le parent
            S_1 = np.logical_or(p1, np.logical_or(p2,p3))^S_3^S_2
            rand = np.random.rand()
            if(rand < 0.5):
                z1_idx, z2_idx = self.utility.select_index(S_2)
                z_idx = self.utility.compareL(z1_idx, z2_idx, self.sc)
                O_i[z1_idx] = 0
            else:
                o1_idx,o2_idx = self.utility.select_index(S_1)
                o_idx = self.utility.compareS(o1_idx, o2_idx, self.sc)
                O_i[o_idx] = 1

            offsprings = O_i
    
        return offsprings
    

    def mutation(self,offspring):
        """
            Permet de muter un individu de la population
        """
        for i in range(0,self.N):
            rand = random.random()
            if rand < 0.5 :
                s_idx = np.where(offspring ==1)[0]
                if len(s_idx != 0):
                    z_idx =self.binary_tournament_base_list_index_sc(s_idx,self.sc)
                    offspring[z_idx] =0
            else:
                us_idx = np.where(offspring ==0)[0]
                if len(us_idx != 0):
                    o_idx =self.binary_tournament_base_list_index_sc(us_idx,self.sc)
                    offspring[o_idx] =1
        return offspring
    

    def binary_tournament_base_list_index_sc(self,index_list, sc):
        """
            Permet de selectionner les indices des caracteristiques les plus performantes
        """
        selected_indices = []
        # print("reherche",index_list)
        if(len(index_list) == 1):
            return index_list[0]
        for _ in range(len(index_list)):
            select_index_num = np.random.choice(index_list,size=2, replace=False)
            selected_indices.append(np.max(select_index_num))
        return selected_indices

    def tournment_selection(self,P):
        P_p =[]
        for _ in range(0, 3): # 3 individus
            tmp = np.random.choice(len(P),size=2, replace=False)
            ind1 = self.objectif_function.get_objectif_values(P[tmp[0]])
            ind2 = self.objectif_function.get_objectif_values(P[tmp[1]])
            if NSGA2.dominance2(ind1, ind2) :
                P_p.append(P[tmp[0]])
            else:
                P_p.append(P[tmp[1]]) 
                
        return P_p


    def run_mofs_rfga(self):
        print("--------Start algorithm---------")
        P = self.initialisation() # initialisation of population
        nFE = 0
        P_new = []
        PF = []
        print("----------Iteration---",0,"-----------")
        while(nFE < self.maxFES):
            for i in  range(self.N):
                Pp = self.tournment_selection(P)
                Pc = self.crossover_3_to_1(Pp[0],Pp[1],Pp[2])
                Pc = self.mutation(Pc)
                nFE = nFE + 1
                P_new.append(Pc)            
            R = P+ P_new
            R = self.utility.remove_duplicates(R)
            P, new_solution = self.nsga2.EnvironmentalSelection(R, self.N)
            P = self.utility.remove_duplicates(P)
            #PF = [ sol for sol in new_solution if sol.rank == 1]
            PF = [ sol for sol in new_solution]
            print("----------Iteration---",nFE,"-----------")
        print("---------- Algorithme End -----------")
        #for sol in PF:
        #    print(sol)
        #print("---------- Algorithme solution -----------")
        return PF
