import numpy as np

class Individual:

    def __init__(self, values , fitness):
        """
            values: list des valeurs  ,
            fitness: list des fitness values
        """
        self.indiv  = values
        self.fitness = fitness
        self.np = 0 # Initialise np = 0. Pour compter le nombre d’individus qui dominent p
        self.rank = 0
        self.indice = 0 # Pour le tri rapide dans le front de pareto
        self.dominated_by = None
        self.crowding_distance = 0

    def __eq__(self, object2):
        """ 
            Permet de verifier si deux individus sont égaux
        """
        if isinstance(object2, Individual):
            if ((object2.np == self.np) and  (object2.rank == self.rank) and  np.array_equal(object2.fitness, self.fitness) and np.array_equal(object2.indiv, self.indiv)):
                return True
        return False


    def __str__(self):
        return f"Individual(fitness={self.fitness}, c_d={self.crowding_distance}), rank={self.rank}"