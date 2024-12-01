import numpy as np
import matplotlib.pyplot as plt
from models.individual import Individual
from matplotlib.ticker import MaxNLocator

class Utility:

    def find_max(self,feature_scores,selected_index):
        """
            Trouve l'indice de la valeur maximale entre deux indices spécifiés.

        Args:
            feature_scores (list): Une liste de scores des caractéristiques (valeurs numériques).
            selected_index (list): Une liste contenant deux indices à comparer.

        Return:
            int: L'indice de la valeur maximale entre les deux indices spécifiés.
        """

        if(feature_scores[selected_index[0]] >= feature_scores[selected_index[1]]):
            return selected_index[0]
        else:
            return selected_index[1]


    
    def compareL(self,index1,index2, sc):
        """
            Compare les scores associés à deux indices et renvoie l'indice de la valeur maximale.

            Cette fonction compare les scores associés aux indices `index1` et `index2` dans la liste `sc`.
            Elle renvoie l'indice de la valeur maximale entre les deux indices donnés.

        Args:
            index1 (int): Le premier indice à comparer.
            index2 (int): Le deuxième indice à comparer.
            sc : Une liste de scores associés aux indices.

        Return:
            int: L'indice de la valeur maximale entre les deux indices donnés.
    
        """
        if(sc[index1]>=sc[index2]):
            return index1
        else:
            return index2

    
    def compareS(self, index1, index2, sc):
        """
            Compare les scores associés à deux indices et renvoie l'indice de la valeur minimale.
        """
        if(sc[index1] <= sc[index2]):
            return index1
        else:
            return index2

    
    def select_index(self,vector):
        """
            Sélectionne deux indices aléatoires dans une liste.
            Retourne les deux indices sélectionnés.
        """
        z1_idx =  np.random.choice(len(vector))
        z2_idx =  np.random.choice(len(vector)) 
        return z1_idx, z2_idx

    def remove_duplicates(self,P):
        """
            Cette methode permet de supprimer les doublons dans une liste
        """
        unique_dict = {}  # Dictionnaire pour stocker les sous-listes uniques

        for sublist in P:
            unique_dict[tuple(sublist)] = sublist  # Utilise une clé de tuple pour stocker la sous-liste

        unique_lst_of_lsts = list(unique_dict.values())  # Convertit les valeurs du dictionnaire en liste

        return unique_lst_of_lsts
    
    def is_contains_individu(self,front , individu):
        """
            Cette methode permet de verifier si un individu est dans un front de pareto
        """
        for indiv in front : 
            if indiv is not None :
                if indiv == individu :
                    return True
        return False
    

    def visualize_error_size(self,solution ,methode, dataset_name, color_name):
        x = []
        y = []
        mark = ["o", '*', 's', 'p', '*', '+', 'x', 'D', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'h', 'H', 'd', '|', '_']

        for sol in solution:
            x.append(sol.fitness[1])
            y.append(sol.fitness[0])

        rk = max(sol.rank for sol in solution)

        error = [[] for _ in range(rk+1)]
        size_sol = [[] for _ in range(rk+1)]
        for sol in solution:
            error[sol.rank].append(sol.fitness[0])
            size_sol[sol.rank].append(int(sol.fitness[1]))

        plt.figure()
        K = 0
        for i, (x_point, y_point) in enumerate(zip(error, size_sol)):
            if len(x_point) > 0 and len(y_point) > 0:
                marker = mark[i % len(mark)]  # Utiliser le marqueur correspondant à l'indice actuel
                plt.scatter( y_point,x_point, label=f'Front {K+1}', marker=marker, color=color_name)
                K = K+1

        plt.xlabel('F2: Nombre de variables')
        plt.ylabel('F1: Erreur de classification en %')
        plt.title(f'Methode {methode} : dataset {dataset_name}')

        plt.legend()  
        plt.show()



    def visualize_double_solution(self,solution ,  solution_nsga ):
        
        error = []
        size_sol = []
        error_2 = []
        size_sol_2 = []
        error = [ sol.fitness[0] for sol in solution]
        size_sol =[int(sol.fitness[1]) for sol in solution]

        error_2 = [ sol.fitness[0] for sol in solution_nsga]
        size_sol_2 =[int(sol.fitness[1]) for sol in solution_nsga]
        plt.scatter(size_sol, error,label='MOFS-RFGA', marker='o', linestyle='-', color='blue')
        plt.scatter(size_sol_2, error_2,label='NSGA-2', marker='+', linestyle='--', color='red')

        plt.xlabel('F2: Number of features')
        plt.ylabel('F1: Classification error (%)')
        #plt.title(f'Erreur de classification en fonction du nombre de variable') 
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()  
        plt.savefig('best_pareto_front_training_set_mofs_rfga.eps')
        plt.show()

    def visualize_double_solution_reg(self, solution, solution_nsga):
               
        error = []
        size_sol = []
        error_2 = []
        size_sol_2 = []
        error = [ sol.fitness[0] for sol in solution]
        size_sol =[int(sol.fitness[1]) for sol in solution]

        error_2 = [ sol.fitness[0] for sol in solution_nsga]
        size_sol_2 =[int(sol.fitness[1]) for sol in solution_nsga]
        plt.scatter(size_sol, error,label='T-MOFS-RRGA', marker='o', linestyle='-', color='blue')
        plt.scatter(size_sol_2, error_2,label='NSGA-2', marker='+', linestyle='--', color='red')

        plt.xlabel('F2: Number of features')
        plt.ylabel('F1: Mean square error')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
       # plt.title(f'Erreur d\'apprentissage en fonction du nombre de variable')

        plt.legend()  
        plt.savefig('best_pareto_front_training_set_tmofs_rrga.eps')  

        plt.show()


    def visualize_front(self,solution):
        x = []
        y = []
        mark = ["o", '*', 's', 'p', '*', '+', 'x', 'D', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'h', 'H', 'd', '|', '_']

        for sol in solution:
            x.append(sol.fitness[1])
            y.append(sol.fitness[0])

        rk = max(sol.rank for sol in solution)

        error = [[] for _ in range(rk+1)]
        size_sol = [[] for _ in range(rk+1)]
        for sol in solution:
            error[sol.rank].append(sol.fitness[0])
            size_sol[sol.rank].append(int(sol.fitness[1]))

        plt.figure()
        K = 0
        for i, (x_point, y_point) in enumerate(zip(error, size_sol)):
            if len(x_point) > 0 and len(y_point) > 0:
                marker = mark[i % len(mark)]  # Utiliser le marqueur correspondant à l'indice actuel
                plt.scatter( y_point,x_point, label=f'Front {K+1}', marker=marker)
                K = K+1

        plt.xlabel('F2: Nombre de variables')
        plt.ylabel('F1: Erreur de classification en %')
        plt.title(f'Front de Pareto')

        plt.legend()  
        plt.show()


    def process_experimentation():
        pass

    def read_solution_i(self, path , algorithm_name):
        import re

        with open(path, 'r') as file:
                lines = file.read()
        individuals = []
        if (algorithm_name =="mofs_rfga"):
            ranks = []
            pattern = r"Individu : \[(.*?)\]\s+Objectifs : \[(.*?)\]\s+Rang : (\d+)"
            matches = re.findall(pattern, lines, re.DOTALL)

            for match in matches:
                indiv = [int(x) for x in match[0].split()]
                obj = [float(x) for x in match[1].split(', ')]
                individual = Individual(indiv, obj)
                individual.rank = int(match[2])
                individuals.append(individual)
        else:
            pattern = r"Individu : \[(.*?)\]\s+Objectifs : \((.*?), (.*?)\)\s+"
            matches = re.findall(pattern, lines)

            for match in matches:
                indiv = [int(x) for x in match[0].split(', ')]
                individuals.append(Individual(indiv, [float(match[1]), float(match[2])]))


        return individuals

        #methode permettant de visualiser les erreurs en fonction de chaque modeles
    def plot_model_error(self, error_list_value,error_label):
        width = 0.3
        error_type = error_list_value
        error_title = error_label
        k=0
        x = np.arange(len(model_list)) 
        palette=["#115f9a", "#a6d75b", "#22a7f0", "#ef9b20", "#f4f100", "#0080ff"]
        fig,ax = plt.subplots(2,2,figsize = (20,10))
        for i in range(2):
            for j in range(2):           
                rects = ax[i,j].bar(x, error_type[k], width)
                ax[i,j].set_ylabel(f'{error_title[k]}')
                ax[i,j].set_xlabel('Models')
                ax[i,j].set_title(f'{error_title[k]} des différents algorithmes')
                ax[i,j].set_xticks(x)
                ax[i,j].set_xticklabels(names_models, rotation=45)
                for l in range(6):      
                    rects[l].set_color(palette[l])
                
                    ''''Autolabel'''  
                
                for rect in rects:
                    height = rect.get_height()
                    ax[i,j].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                k=k+1
        fig.tight_layout()
        #ax.set_title("Visualisation des différents érreurs en fonction des modèles")
        plt.show()


#methode permettant de visualiser les erreurs en fonction de chaque modeles
    def Plot_Model_Error(error_list_value,error_label):
        width = 0.3
        error_type = error_list_value
        error_title = error_label
        k=0
        x = np.arange(len(model_list)) 
        palette=["#115f9a", "#a6d75b", "#22a7f0", "#ef9b20", "#f4f100", "#0080ff"]
        fig,ax = plt.subplots(2,2,figsize = (20,10))
        for i in range(2):
            for j in range(2):           
                rects = ax[i,j].bar(x, error_type[k], width)
                ax[i,j].set_ylabel(f'{error_title[k]}')
                ax[i,j].set_xlabel('Models')
                ax[i,j].set_title(f'{error_title[k]} des différents algorithmes')
                ax[i,j].set_xticks(x)
                ax[i,j].set_xticklabels(names_models, rotation=45)
                for l in range(6):      
                    rects[l].set_color(palette[l])
                
                ''''Autolabel'''  
                
                for rect in rects:
                    height = rect.get_height()
                    ax[i,j].annotate('{:.2f}'.format(height), 
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                k=k+1
        fig.tight_layout()
        #ax.set_title("Visualisation des différents érreurs en fonction des modèles")
        plt.show()

    
        
        