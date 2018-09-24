import subprocess
import pandas as pd
import numpy as np

DNA_SIZE = 13            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE_META = 0.003    # mutation probability
N_GENERATIONS = 200


POPULATION = np.arange(10,160,10)
CROSSOVER_RATE = np.arange(0.25,1.00,0.05)
MUTATION_RATE = [0]
for i in range(0,-7,-1):
    MUTATION_RATE.append(2**i)
#GENERATION_GAP = np.arange(0.3,1,0.1)
#SCALING_WINDOW = np.arange(8)
SELECTION_STRATEGY = ['RANK', 'ROULETTEWHEEL', 'TOURNAMENT', 'BINARY_TOURNAMENT']


def branch_coverage(indices, i):

    population_size= POPULATION[indices[0][i]]
    crossover_rate= CROSSOVER_RATE[indices[1][i]]
    mutation_rate= MUTATION_RATE[indices[2][i]]
    selection_strategy= SELECTION_STRATEGY[indices[3][i]]
    process = subprocess.Popen(
        ['java', '-jar', '/home/ubuntu/evosuite-1.0.6.jar',
         '-target', '/home/ubuntu/SF100/1_tullibee/tullibee.jar',
         '-Dcrossover_rate={}'.format(crossover_rate), '-Dpopulation={}'.format(population_size),
         '-Dmutation_rate={}'.format(mutation_rate),
         '-Dsearch_budget=20', '-Dshow_progress=False', '-Doutput_variables=BranchCoverage'])
    process.wait()
    # process = subprocess.Popen(
    #     ['java', '-jar', 'C:/Users/Shayan Z/Downloads/SF100-EvoSuite-20120316/1_tullibee/evosuite-1.0.6.jar',
    #      '-target', 'tullibee.jar',
    #      '-criterion', 'branch', '-Dcrossover_rate={}'.format(crossover), '-Dpopulation={}'.format(population),
    #      '-Dsearch_budget=20', '-Doutput_variables=BranchCoverage', '-Dshow_progress=False'],
    #     cwd='C:/Users/Shayan Z/Downloads/SF100-EvoSuite-20120316/1_tullibee')
    file = pd.read_csv(
        '/home/ubuntu/SF100/1_tullibee/evosuite-files/evosuite-report/statistics.csv')
    r, c = file.shape
    total_array = file.at[r - 1, 'Total_Goals']
    print(total_array)
    covered_array = file.at[r - 1, 'Covered_Goals']
    coverage = covered_array / total_array
    return coverage


# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


def translateDNA(pop):
    state = pop.dot(2 ** np.arange(DNA_SIZE)[::-1])
    population_size_index = state%16

    state = state//16
    crossover_rate_index = state%16

    state = state//16
    mutation_rate_index = state%8

    #state = state//8
    #generation_gap_index = state%8

    #state = state//8
    #scaling_window_index = state%8

    state = state//8
    selection_strategy_index = state
    
    indices = [population_size_index, crossover_rate_index, mutation_rate_index, selection_strategy_index]
    return indices


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE_META:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA


for _ in range(N_GENERATIONS):
    indices = translateDNA(pop)
    for i in range(POP_SIZE):
        F_values[i] = branch_coverage(indices,i)    # compute function value by extracting DNA


    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
    print('iteration number:', _)
