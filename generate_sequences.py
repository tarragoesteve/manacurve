from itertools import combinations_with_replacement, count, permutations
from scipy.stats import multivariate_hypergeom
from tqdm import tqdm
import csv
import copy
import random
from enum import Enum
from typing import List

class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    SINGLE = 3
    NOTHING = 4

MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 7
FINAL_TURN = 8
DECK_SIZE = 60
MAXIMUM_NUMBER_SEQUENCES = 2000
OVERWRITE_SEQUENCES = True
STRATEGY = Strategy.NOTHING
MULTI_HILL_CLIMBING_ITERATIONS = 10


# Impact Methods
def sequence_impact(sequence: List[List[int]]) -> float:
    """Given a sequence computes the expected impact in a game

    Args:
        sequence (List[List[int]]): for each turn what cards have been played

    Returns:
        float: expected impact
    """
    total = 0.0
    for turn in sequence:
        total += sum([impact(i) for i in turn])
    return total

def turn_impact(turn : List[int]) ->  float:
    """Given a turn computes the impact in a game

    Args:
        turn (List[int]): cards played in a turn

    Returns:
        float: impact
    """
    return sum([impact(i) for i in turn])

def impact(mana_value: int) -> float:
    """Given a card mana value computes the impact in a game
    """
    if mana_value == 0:
        return 0
    elif mana_value == 1:
        return mana_value + 1/0.95173
    elif mana_value == 2:
        return mana_value + 1/0.8241
    elif mana_value == 3:
        return mana_value + 1/0.63784
    elif mana_value == 4:
        return mana_value + 1/0.44051
    elif mana_value == 5:
        return mana_value + 1/0.27278
    elif mana_value == 6:
        return mana_value + 1/0.15229
    elif mana_value == 7:
        return mana_value + 1/0.07698
    return mana_value+1.0
        
# Generating sequences methods
def sequence_without_lands(sequence: List[List[int]]) -> List[List[int]]:
    """Given a sequence removes lands

    Args:
        sequence (List[List[int]]): 

    Returns:
        List[List[int]]: 
    """
    ret = []
    for turn in sequence:
        ret.append([i for i in turn if i != 0])
    return ret
        
def possible_sequences():
    """Generates all possible sequences of turns"""
    yield from possible_sequences_auxiliar(0, 0, [], INITIAL_HAND_SIZE)

def possible_sequences_auxiliar(turn: int, landrops: int,
                                current_sequence: List[List[int]], remaining_cards: int):
    """Generates all possible sequences of turns

    Args:
        turn (int): 
        landrops (int): 
        current_sequence (List[List[int]]): 
        remaining_cards (int): 

    Yields:
        List[List[int]]: sequence of turns
    """
    if turn == FINAL_TURN:
        yield current_sequence
    else:
        # case we do not play land
        for turn_sequence in possible_turn([], landrops, remaining_cards):
            yield from possible_sequences_auxiliar(turn + 1, landrops, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
        # case we play a land
        for turn_sequence in possible_turn([0], landrops + 1, remaining_cards-1):
            yield from possible_sequences_auxiliar(turn + 1, landrops + 1, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))


def possible_turn(turn : list[int], available_mana: int, remaining_cards: int):
    """Generates all possible turns

    Args:
        turn (list[int]): The cards played so far in the turn
        available_mana (int): Amount of mana remaining
        remaining_cards (int): Amount of cards remaining

    Yields:
        _type_: _description_
    """
    yield turn
    if remaining_cards > 0 and available_mana > 0:
        #only ascending order
        last_played = 1
        if len(turn) > 0:
            last_played = turn[-1]
            last_played = max(1, last_played) # we cannot repeat landrops
        for i in range(last_played, min(available_mana+1, MAXIMUM_MANA_VALUE+1)):
            yield from possible_turn(copy.deepcopy(turn + [i]), available_mana - i, remaining_cards - 1)

# Probability computation auxiliar method
def non_valid_possible_combinations(Ks: List[int], Cs: List[int], free_cards: int):
    """Returns all the possible combinations that are not valid

    Args:
        Ks (List[int]): Minim number of each k to be a valid restriction
        Cs (List[int]): Maximum number of k that are available
        free_cards (int): Number of cards that can be played

    Yields:
        List[int]: combination 
    """
    yield from non_valid_possible_combinations_aux(0, [], True, Ks, Cs, free_cards)
    
    
def non_valid_possible_combinations_aux(index: int, combination: List[int],
                                        is_valid: bool,
                                        Ks: List[int], Cs: List[int], free_cards: int):
    if index >= len(Ks):
        if not is_valid:
            yield combination
    else:
        up_to = min(Cs[index], free_cards)
        for i in range(up_to+1):
            yield from non_valid_possible_combinations_aux(index+1,
                                          combination + [i],
                                          is_valid and i >= Ks[index],
                                          Ks,
                                          Cs,
                                          free_cards - i)

class Node():
    def __init__(self, turn = [], sequence = [], impact = 0.0, children = []) -> None:
        self.turn = turn
        self.sequence = sequence
        self.impact = impact
        self.children = children

# Score computation method
def score_auxiliar(tree_sequence: Node, Cs: List[int], free_cards: int, node_probability, minimum_probability = 0.0):
    turn_conditioned_probability = 1.0
    ks = [sum(1 for j in tree_sequence.turn if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
    ks_index = [i for i in range(MAXIMUM_MANA_VALUE+1) if ks[i] > 0]
    ks_values = [ks[i] for i in ks_index]
    if any([k > c for k, c in zip(ks, Cs)]) or sum(ks) > free_cards:
        turn_conditioned_probability = 0
    else:
        for combination in non_valid_possible_combinations(ks_values, [Cs[i] for i in ks_index], free_cards):
            k_other = free_cards - sum(combination)
            K_other = sum(Cs) - sum(Cs[i] for i in ks_index)
            if K_other < k_other:
                # This combination is not possible
                pass
            else:
                combination_prob = multivariate_hypergeom.pmf(x=[i for i in combination]+[k_other], m=[Cs[i] for i in ks_index]+[K_other], n=free_cards)
                turn_conditioned_probability -= combination_prob
    node_probability *= turn_conditioned_probability
    if node_probability > minimum_probability:
        # continue exploring childs only if they have a significant probability
        #print("Turn", tree_sequence.turn, "Probability", node_probability, "Impact", tree_sequence.impact)
        for child in tree_sequence.children:
            yield from score_auxiliar(child, [c - k for k, c in zip(ks, Cs)], free_cards - sum(ks) + 1, node_probability, minimum_probability)
        if len(tree_sequence.children) == 0:
            yield node_probability, tree_sequence.impact, tree_sequence.sequence                  


def score(tree_sequence: Node, Cs: List[int], minimum_probability = 0.0) -> float:
    return sum([probability * impact for probability, impact, _ in score_auxiliar(tree_sequence, Cs, INITIAL_HAND_SIZE-1, 1.0, minimum_probability)])


if OVERWRITE_SEQUENCES:
    saved_combinations = {}
    possible = 0
    print("Exploring all possible sequences")
    for sequence in tqdm(possible_sequences()):
        possible += 1
        if sequence_impact(sequence) > 0:
            if str(sequence_without_lands(sequence)) not in saved_combinations:
                saved_combinations[str(sequence_without_lands(sequence))] = sequence
            else:
                if len(sequence) < len(saved_combinations[str(sequence_without_lands(sequence))]):
                    # shortest combination == useless lands
                    saved_combinations[str(sequence_without_lands(sequence))] = sequence
                elif len(sequence) == len(saved_combinations[str(sequence_without_lands(sequence))]):
                    for turn_a, turn_b in zip(sequence, saved_combinations[str(sequence_without_lands(sequence))]):
                        # we prefer to play lands in the latest turns
                        if len(turn_a) < len(turn_b):
                            saved_combinations[str(sequence_without_lands(sequence))] = sequence
                            break
                        elif len(turn_a) > len(turn_b):
                            break
                        
    print("Non-redudant", len(saved_combinations), "out of", possible)
    sequence_score = [(sequence_impact(sequence), sequence) for sequence in saved_combinations.values()]
    sequence_score.sort(key=lambda x: x[0], reverse=True)
    with open('sequences.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        for i in range(min(len(sequence_score), MAXIMUM_NUMBER_SEQUENCES)):
            writer.writerow([sequence_score[i][0]] + sequence_score[i][1])
    
print("Generating tree")
with open('sequences.csv', newline='') as csvfile:
    number_of_nodes = 1
    reader = csv.reader(csvfile, delimiter=";")  
    root_node = Node()
    for row in tqdm(reader):
        current_node = root_node
        sequence = []
        for turn_string in row[1:]:
            if len(sequence) < FINAL_TURN:
                remove_brackets = turn_string[1:-1]
                turn = [int(i) for i in remove_brackets.split(",")]
                sequence.append(turn)
                if turn not in [child.turn for child in current_node.children]:
                    number_of_nodes +=1
                    current_node.children.append(Node(copy.deepcopy(turn), copy.deepcopy(sequence), current_node.impact + turn_impact(turn), []))
                current_node = current_node.children[[child.turn for child in current_node.children].index(turn)]
    print("Number of nodes", number_of_nodes)

def hill_climbing(initial_combination, root_node):
    best_score = 0
    best_combination = initial_combination
    keep_exploring = True
    last_direction = (0, 1)
    while keep_exploring:
        keep_exploring = False
        directions = [(i, j) for j in range(len(best_combination)) for i in range(len(best_combination)) if i!=j and (i, j) != last_direction]
        random.shuffle(directions)
        directions = [last_direction] + directions
        for (i, j) in tqdm(directions):
            combination = copy.deepcopy(best_combination)
            combination[j] += 1
            combination[i] -= 1
            if all([k >= 0 for k in combination]):
                combination_score = score(root_node, combination)
                if combination_score > best_score:
                    last_direction = (i, j)
                    best_score = combination_score
                    keep_exploring = True
                    new_best_combination = combination
                    #break
        if keep_exploring:
            best_combination = new_best_combination
            yield best_combination, best_score
    

print("Selected strategy", STRATEGY)

if STRATEGY == Strategy.NOTHING:
    print("Nothing to do")
else:
    with open('curves.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        best_score = 0.0
        best_combination = []
        if STRATEGY == Strategy.FULL_EXPLORATION:
            for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE),
                                    total=len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)))):
                Ks = [sum(1 for j in combination if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
                combination_score = score(root_node, Ks)
                writer.writerow([combination_score] + Ks)
                if combination_score > best_score:
                    print("[FULL_EXPLORATION] Better score found:", combination_score, Ks)
                    best_score = combination_score
                    best_combination = Ks
        elif STRATEGY == Strategy.HILL_CLIMBING:
            # initial solution and loop variables
            initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
            initial_combination[1] = DECK_SIZE
            initial_combination = [19,26,13,2,0]
            for combination, combination_score in tqdm(hill_climbing(initial_combination, root_node)):
                if combination_score > best_score:
                    #print("[HILL_CLIMBING] Better score found:", combination_score, combination)
                    best_score = combination_score
                    best_combination = combination
                    writer.writerow([combination_score] + combination)
                    csvfile.flush()
        elif STRATEGY == Strategy.MULTI_HILL_CLIMBING:
            cached_combinations = {}
            possible_combinations = list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE))
            for _ in tqdm(range(MULTI_HILL_CLIMBING_ITERATIONS)):
                selected = random.choice(possible_combinations)
                Ks = [sum(1 for j in selected if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
                for combination, combination_score in tqdm(hill_climbing(Ks, root_node)):
                    if combination_score > best_score:
                        #print("[MULTI_HILL_CLIMBING] Better score found:", combination_score, combination)
                        best_score = combination_score
                        best_combination = combination
                        writer.writerow([combination_score]+ combination)
                        csvfile.flush()
                    if str(combination) not in cached_combinations:
                        cached_combinations[str(combination)] = combination_score
                    else:
                        print("Stopping iteration, we arrived at", combination)
                        break
        elif STRATEGY == Strategy.SINGLE:
            combination = [20,10,10,5,5]
            minimum_probabilities = [0.8, 0.3, 0.05, 0.01, 0]
            for p in tqdm(minimum_probabilities):
                combination_score = score(root_node, combination, p)
                print(combination_score)
            best_combination = combination
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        for probability, sequence_impact, sequence in score_auxiliar(root_node, best_combination, INITIAL_HAND_SIZE-1, 1.0, 0.0):
            used_cards = sum([1 for turn in sequence for _ in turn])
            writer.writerow([round(probability,4), sequence_impact, used_cards] + sequence)
            
        
        
    
        


            

        
