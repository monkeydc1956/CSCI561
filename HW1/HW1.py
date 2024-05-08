import math

import numpy as np
import time
import copy
import random
def createInitialPopulation(size:int, cities:list, huristic_rate,distance):
    initial_population = []
    huristic_population = huristic_rate*size
    while huristic_population>0:
        initial_population.append(createInitialPopulationWithHuristic(cities,distance))
        huristic_population -= 1
    while len(initial_population) < size:
        random_path = list(np.random.permutation(len(cities))) # randomly creat paths
        # random_path.append(random_path[0]) # append start city
        # if random_path in initial_population:
        #     pass
        # else:
        #     initial_population.append(random_path)
        initial_population.append(random_path)
    initial_population = sorted(initial_population)
    return initial_population


def distanceCalculate(cities:list, distance:list):
    for i in range(len(cities)):
        for j in range(i+1, len(cities)): # reduce half calculation
            tmp_distance = np.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2 + (cities[i][2]-cities[j][2])**2)
            distance[i][j] = tmp_distance
            distance[j][i] = tmp_distance
    return

def population_fitness(path:list, distance:list):
    length = distance[path[0]][path[-1]]
    for i in range(len(path)-1):
        length += distance[path[i]][path[i+1]]
    return length

def tournamentSelection(old_poplation:list, size:int, distance:list): # tournament selection, which is better than RWS because it can provide more diversity in population
    selection_pool = []
    for _ in range(size):
        i, j = np.random.randint(0,size,2)
        if population_fitness(old_poplation[i],distance) < population_fitness(old_poplation[j],distance):
            selection_pool.append(old_poplation[i])
        else:
            selection_pool.append(old_poplation[j])
    return selection_pool

# def betterTournamentSelection():
#     selection_pool = []
#     return selection_pool

def RWSelection(oldpopulation:list, distance): #  Roulette wheel-based selection
    fitness = []
    for i in range(len(oldpopulation)):
        fitness.append(population_fitness(oldpopulation[i],distance))

    fitness_sum = []

    for i in range(len(fitness)):
        if i == 0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + fitness[i])

    for i in range(len(fitness_sum)):
        fitness_sum[i] /= sum(fitness)

    # select new population
    population_new = []
    for i in range(len(fitness)):
        rand = np.random.uniform(0, 1)
        for j in range(len(fitness)):
            if j == 0:
                if 0 < rand and rand <= fitness_sum[j]:
                    population_new.append(population[j])

            else:
                if fitness_sum[j - 1] < rand and rand <= fitness_sum[j]:
                    population_new.append(population[j])
    return population_new
############################
def rankPopulation(poplulation,distance):
    ranked_population_dic = {}
    for i in range(len(poplulation)):
        fitness = population_fitness(poplulation[i],distance)
        ranked_population_dic[i] = fitness

    return sorted(ranked_population_dic.items(), key=lambda x: x[1], reverse=True)


def selectRWSwithElite(population, population_rank, eliteSize):
    select_pop = []
    for i in range(eliteSize):
        select_pop.append(population[population_rank[i][0]])

    cumsum = 0
    cumsum_list = []
    temp_pop = copy.deepcopy(population_rank)
    for i in range(len(temp_pop)):
        cumsum += temp_pop[i][1]
        cumsum_list.append(cumsum)
    for i in range(len(temp_pop)):
        cumsum_list[i] /= cumsum

    for i in range(len(temp_pop) - eliteSize):
        rate = np.random.random()
        for j in range(len(temp_pop)):
            if cumsum_list[j] > rate:
                select_pop.append(population[population_rank[i][0]])
                break

    return select_pop


def betterCrossover(population, eliteSize):
    breed_population = []
    for i in range(eliteSize):
        breed_population.append(population[i])

    count = 0
    while count < (len(population) - eliteSize):
        index_1 = np.random.randint(0, len(population) - 1)
        index_2 = np.random.randint(0, len(population) - 1)
        if index_1 != index_2:
            parent_1, parent_2 = population[index_1], population[index_2]
            left  = np.random.randint(0, len(population[index_1]) - 1)
            right = np.random.randint(0, len(population[index_2]) - 1)
            start_index = min(left, right)
            end_index = max(left, right)
            child1 = []
            for j in range(start_index, end_index):
                child1.append(parent_1[j])
            child2 = []
            for j in parent_2:
                if j not in child1:
                    child2.append(j)
            breed_population.append(child1 + child2)
            count += 1
    return breed_population


def resembleMutate(population, mutationRate): #resemble mutation
    mutation_population = []
    for i in range(len(population)):
        for j in range(len(population[i])):
            possibility = np.random.random()
            if possibility < mutationRate:
                a = np.random.randint(0, len(population[i]) - 1)
                population[i][a], population[i][j] = population[i][j], population[i][a]
        mutation_population.append(population[i])

    return mutation_population


def next_population(population, eliteSize, mutationRate, distance):
    population_rank = rankPopulation(population,distance)
    select_population = selectRWSwithElite(population, population_rank, eliteSize)
    breed_population = betterCrossover(select_population, eliteSize)
    next_generation = resembleMutate(breed_population, mutationRate) # all population mutate

    return next_generation

def createInitialPopulationWithHuristic(cities:list,distance:list): # increase quality of initial population
    path = []
    cities_size = len(cities)
    original_cities = []
    for i in range(cities_size):
        original_cities.append(i)
    cur_city = np.random.randint(0, cities_size) # select random city as the start point, due to it is a circle, which city is the start does not matter
    path.append(cur_city)
    # print(cur_city)
    # print(original_cities)
    original_cities.remove(cur_city)
    while len(original_cities) > 0:
        target_city_distance = math.inf
        for i in original_cities:
            temp_distance = distance[cur_city][i]
            if target_city_distance>temp_distance:
                target_city_distance = temp_distance
                cur_city = i
        original_cities.remove(cur_city)
        path.append(cur_city)
    return path
############################




def mutation(child, mutation_p): # exchange two positions inside one path
    for i in range(len(child)):
        if np.random.uniform(0, 1) <= mutation_p:
            position1 = np.random.randint(0, len(child))
            position2 = np.random.randint(0, len(child))
            child[position1],child[position2] = child[position2],child[position1]
    return child

# could have bugs:  [4, 3, 7, 10, 10, 5, 6, 8, 1, 9, 2]
#                   [2, 3, 9, 10, 2, 5, 6, 8, 0, 1, 7]
# def reduceRepeatDNA(child_1:list, child_2:list):
#     city_1, city_2 = findRepeatDNA(child_1), findRepeatDNA(child_2)
#     print(child_1)
#     print(child_2)
#     while city_1 or city_2:
#         child_1[city_1], child_2[city_2] = child_2[city_2], child_1[city_1]
#         city_1, city_2 = findRepeatDNA(child_1), findRepeatDNA(child_2) #switch
#         print("############")
#         print(child_1)
#         print(child_2)
#     return child_1, child_2

def selfReduceRepeat(child:list):
    revise = {}
    lookup = []
    for i in range(len(child)):
        check = child.count(i)
        if check>1:
            revise[i] = child.index(i)
        elif check == 0:
            lookup.append(i)
        else:
            pass
    for key in revise.keys():
        child[revise[key]] = lookup.pop()
    return child

# def findRepeatDNA(path): # could have better implementation
#     for i in path[1:-1]:
#         if path.count(i) > 1:
#             return path.index(i)
#     return None

def crossover(mating_pool, size, crossover_p, mutation_p, distance):
    new_generation = []
    DNA_size = len(mating_pool[0])
    while len(new_generation) < size:
        i1, i2 = np.random.randint(0, size, 2)
        p1 = mating_pool[i1].copy()
        p2 = mating_pool[i2].copy()
        if p1 != p2:
            if np.random.uniform(0,1) < crossover_p: # strategy of crossover
                pass
                #new_generation.extend([p1, p2])
            else:
                i, j = get_2_randint(DNA_size)
                p1[i:j - 1], p2[i:j - 1] = p2[i:j - 1], p1[i:j - 1]
    #            p1, p2 = reduceRepeatDNA(p1, p2)
                p1 = selfReduceRepeat(p1)
                p2 = selfReduceRepeat(p2)
                p1 = mutation(p1,mutation_p)
                p2 = mutation(p2,mutation_p)
                # self-check?
                new_generation.append(p1)
                new_generation.append(p2)
                # if population_fitness(p1,distance) > population_fitness(p2,distance):
                #     new_generation.append(p1)
                #     #if p1 in new_generation:
                #     #    pass
                #     #else:
                #     #    new_generation.append(p1)
                # else:
                #     new_generation.append(p2)
                #     #if p2 not in new_generation:
                #     #    pass
                #     #else:
                #     #    new_generation.append(p2)
        else:
            pass

    return new_generation

def get_2_randint(DNA_size): #switiching size should >= 1
    start_index = np.random.randint(0,DNA_size-3)
    end_index = np.random.randint(start_index+2, DNA_size-1)
    return start_index, end_index

def readFiles(filename:str):
    cities = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            line_data = line.strip().split()
            integers = [[int(x) for x in line_data]]
            cities.extend(integers)
    file.close()
    return cities

def writeToFile(filename:str, result:list,distance:list,cities:list):
    file = open(filename, 'w')
    file.write(str(np.round(population_fitness(result,distance),3)))
    file.write('\n')
    for i in range(len(result)):
        city = cities[result[i]]
        for j in range(len(city)):
            file.write(str(city[j]))
            file.write(' ')
        file.write('\n')
    first_city = cities[result[0]]
    for j in range(len(first_city)):
        file.write(str(first_city[j]))
        file.write(' ')
    file.write('\n')
    file.close()
    return

def filterResult(population:list, distance:list):
    tmp_res = math.inf
    res = []
    for i in range(len(population)):
        tmp_fit = population_fitness(population[i],distance)
        if(tmp_fit < tmp_res):
            tmp_res = tmp_fit
            res = population[i]
        else:
            pass
    return res, tmp_res

if __name__ == "__main__":
    time_start = time.time()
    huristic_rate = 0.5
    eliteSize = 1
    cities = [[1,2,3],[2,2,2],[2,4,5],[3,4,5],[5,6,7]]
    #cities = [[1, 2, 3], [2, 3, 4], [4, 5, 6], [6, 7, 8], [7, 8, 9], [9, 10, 11], [11, 12, 13], [1, 1, 2], [1, 3, 2], [1, 5, 6], [5, 4, 6]]
    #cities = readFiles('./input1.txt')
    population_size = len(cities)
    mutation_P = 0.005 # dynamic change strategy
    crossover_P = 0.2 # dynamic change strategy
    distance_matrix = np.zeros((len(cities),len(cities))).tolist()
    distanceCalculate(cities, distance_matrix)
    population = createInitialPopulation(size=population_size,cities=cities,huristic_rate=huristic_rate,distance=distance_matrix) # initial population
    res = math.inf
    final_list = []
    early_stop_flag = 0
    epoch = 0
    while epoch < math.inf: # math.inf is insane?
        epoch += 1
        mutation_P /= epoch+1 # lower the mutation rate, the module
        crossover_P /= epoch+1
        #pool = tournamentSelection(population, population_size,distance_matrix)
        #pool = RWSelection(population,distance_matrix)
        #pop_rank = rank(population,distance_matrix)
        #pool = select(population,pop_rank,eliteSize)
        #population = crossover(pool, population_size,crossover_P,mutation_P,distance_matrix)
        #population = next_population(population,eliteSize,mutation_P,dista  nce_matrix) # assemble operation
        population_rank = rankPopulation(population, distance_matrix)
        select_population = selectRWSwithElite(population, population_rank, eliteSize)
        breed_population = betterCrossover(select_population, eliteSize)
        next_generation = resembleMutate(breed_population, mutation_P)
        #result = min([population_fitness(i,distance_matrix) for i in population])
        resultList,result = filterResult(population,distance_matrix)
        if result < res:
            final_list = resultList
            res = result
            early_stop_flag = 0
        else:
            early_stop_flag += 1

        if early_stop_flag > 200: # after 200 generations, no more improvement
            #print(epoch)
            break
    print(res)
    writeToFile('./output.txt',final_list,distance_matrix,cities)
    time_end = time.time()
    print(time_end-time_start)




