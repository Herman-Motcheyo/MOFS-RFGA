from utils.utils import Utility
from models.metrics_perfor import Metrics

util = Utility()
metrics = Metrics()

print("*****************Ionspherre *************************")
print("**" * 40)

# --------------------------------------------------------------------------
path1_ionosphere1 = "./solutions/run_nsga/run1_2000/solution_ionosphere.txt"
path1_ionosphere2 = "./solutions/run_mofs_rfga/run2_2000/solution_1.txt"

# Solution optimale pour le problème wdbc
ind_nsga2 =  util.read_solution_i(path1_ionosphere1, "nsga2")
ind_mofs_rfga = util.read_solution_i(path1_ionosphere2, "mofs_rfga")


true_pareto_front_ionosphere = metrics.true_pareto_front(ind_mofs_rfga, ind_nsga2)
solution_paths = [
    
    ("./solutions/run_nsga/run2_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run2_2000/solution_1.txt"),
    ("./solutions/run_nsga/run3_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run3_2000/solution_1.txt"),
    ("./solutions/run_nsga/run4_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run4_2000/solution_1.txt"),
    ("./solutions/run_nsga/run4_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run5_2000/solution_ionosphere.txt")
]

for path_nsga, path_mofs in solution_paths:
    ind_nsga2 = util.read_solution_i(path_nsga, "nsga2")
    ind_mofs_rfga = util.read_solution_i(path_mofs, "mofs_rfga")
    ind_mofs_rfga = [x for x in ind_mofs_rfga if x.rank == 1]

    igd_mofs = metrics.calculate_igd(true_pareto_front_ionosphere, ind_mofs_rfga)
    igd_nsga = metrics.calculate_igd(true_pareto_front_ionosphere, ind_nsga2)

    print("File Pair:", path_nsga, path_mofs)
    print("IGD MOFS:", igd_mofs)
    print("IGD NSGA2:", igd_nsga)
    print("-" * 40)



print("*****************Satellite *************************")
print("**" * 40)

# --------------------------------------------------------------------------
path1_ionosphere1 = "./solutions/run_nsga/run1_2000/solution_ionosphere.txt"
path1_ionosphere2 = "./solutions/run_mofs_rfga/run2_2000/solution_1.txt"

# Solution optimale pour le problème wdbc
ind_nsga2 =  util.read_solution_i(path1_ionosphere1, "nsga2")
ind_mofs_rfga = util.read_solution_i(path1_ionosphere2, "mofs_rfga")


true_pareto_front_ionosphere = metrics.true_pareto_front(ind_mofs_rfga, ind_nsga2)
solution_paths = [
    
    ("./solutions/run_nsga/run2_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run2_2000/solution_1.txt"),
    ("./solutions/run_nsga/run3_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run3_2000/solution_1.txt"),
    ("./solutions/run_nsga/run4_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run4_2000/solution_1.txt"),
    ("./solutions/run_nsga/run4_2000/solution_ionosphere.txt", "./solutions/run_mofs_rfga/run5_2000/solution_ionosphere.txt")
]

for path_nsga, path_mofs in solution_paths:
    ind_nsga2 = util.read_solution_i(path_nsga, "nsga2")
    ind_mofs_rfga = util.read_solution_i(path_mofs, "mofs_rfga")
    ind_mofs_rfga = [x for x in ind_mofs_rfga if x.rank == 1]

    igd_mofs = metrics.calculate_igd(true_pareto_front_ionosphere, ind_mofs_rfga)
    igd_nsga = metrics.calculate_igd(true_pareto_front_ionosphere, ind_nsga2)

    print("File Pair:", path_nsga, path_mofs)
    print("IGD MOFS:", igd_mofs)
    print("IGD NSGA2:", igd_nsga)
    print("-" * 40)
