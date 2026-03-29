from enum import Enum

INITIAL_HAND_SIZE = 7
MAXIMUM_MANA_VALUE = 7
FINAL_TURN = 9
DECK_SIZE = 40

# List of turns to optimize and their relative weights in the combined objective.
# Lengths must match. Weights do not need to be normalized.
TURNS = [3,4,5,6,7,8,9]
TURN_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Folders for generated data files
SEQUENCES_DIR = 'sequences'
TREES_DIR = 'trees'

MAXIMUM_NUMBER_OF_SEQUENCES = 10000
MULLIGAN_THRESHOLD = 0.0


class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    SINGLE = 3
    NOTHING = 4
    
STRATEGY = Strategy.SINGLE

# for hill climbing
INITIAL_COMBINATION = [0] * (MAXIMUM_MANA_VALUE+1)
INITIAL_COMBINATION[0] = 17
INITIAL_COMBINATION[2] = DECK_SIZE - INITIAL_COMBINATION[0]
#INITIAL_COMBINATION = [17,0,12,7,2]
# for single
if STRATEGY == Strategy.SINGLE:
     #INITIAL_COMBINATION = [17,5,9,6,3,0,0,0] # - Optimal turn 5
    #INITIAL_COMBINATION = [18,0,9,6,5,2,0,0] #- Optimal turn 6
    #INITIAL_COMBINATION = [18,0,6,6,5,4,1,0] #- Optimal turn 7
    INITIAL_COMBINATION = [19,0,2,6,4,4,5,0] #- Optimal turn 8


# for multi hill climbing
MULTI_HILL_CLIMBING_ITERATIONS = 5
DECK_MINIMUMS = [0] * (MAXIMUM_MANA_VALUE+1)
# DECK_MINIMUMS[4] = 4
# DECK_MINIMUMS[5] = 2
