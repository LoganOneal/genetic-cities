'''
"The nearness preferences are on a six-point scale, from
 A=Absolutely Necessary and X = Not Desirable. The cells are one-indexed." 
 (https://profiles.stanford.edu/griffin-holt?tab=research-and-scholarship)
'''
NEARNESS_SCALE = {"A": 1, "E": 0, "I": 0, "O": 0, "U": 0, "X": 0}

W, H = 8,8

ZONE_W = 2
ZONE_H = 2

NUM_GENERATIONS = 5
NUM_PARENTS_MATING = 20
NUM_PARENTS_KEPT = 10
MUTATION_RATE = 0.05
POPULATION_SIZE = 100
PARENT_SELECTION_TYPE = "tournament"
CROSSOVER_TYPE = "uniform"

MAX_ROAD_NODES = W*H
