from itertools import combinations_with_replacement, count, permutations
from scipy.stats import multivariate_hypergeom
from tqdm import tqdm
import csv
import copy
import random
from enum import Enum



MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 6
FINAL_TURN = 6
DECK_SIZE = 60

def sequence_impact(sequence):
    total = 0.0
    for turn in sequence:
        total += sum([impact(i) for i in turn])
    return total

def turn_impact(turn):
    return sum([impact(i) for i in turn])
        
def sequence_without_lands(sequence):
    ret = []
    for turn in sequence:
        ret.append([i for i in turn if i != 0])
    return ret
        

def impact(mana_value):
    if mana_value == 0:
        return 0
    return mana_value+1.5


def possible_sequences():
    yield from possible_sequences_auxiliar(0, 0, [], INITIAL_HAND_SIZE+1)

def possible_sequences_auxiliar(turn, landrops, current_sequence, remaining_cards):
    if turn == FINAL_TURN:
        yield current_sequence
    else:
        # case we do not play land
        for turn_sequence in possible_turn([], landrops, remaining_cards):
            yield from possible_sequences_auxiliar(turn + 1, landrops, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
        # case we play a land
        for turn_sequence in possible_turn([0], landrops + 1, remaining_cards-1):
            yield from possible_sequences_auxiliar(turn + 1, landrops + 1, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
    

def non_valid_possible_combinations(Ks, Cs, free_cards):
    yield from recursive_auxiliar(0, [], True, 0, Ks, Cs, free_cards)
    
    
def recursive_auxiliar(index, combination, is_valid, sum_combination, Ks, Cs, free_cards):
    if index >= len(Ks):
        if not is_valid:
            yield combination
    else:
        up_to = min(Cs[index], free_cards - sum_combination)
        for i in range(up_to+1):
            yield from recursive_auxiliar(index+1,
                                          combination + [i],
                                          is_valid and i >= Ks[index],
                                          sum_combination + i,
                                          Ks,
                                          Cs,
                                          free_cards)

class Node():
    def __init__(self, turn = [], impact = 0.0, children = []) -> None:
        self.turn = turn
        self.impact = impact
        self.children = children

def score_auxiliar(tree_sequence: Node, Cs, free_cards, node_probability, minimum_probability = 0.0) -> float:
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
    yield node_probability * tree_sequence.impact
    if node_probability > minimum_probability:
        # continue exploring childs only if they have a significant probability
        #print("Turn", tree_sequence.turn, "Probability", node_probability, "Impact", tree_sequence.impact)
        for child in tree_sequence.children:
            yield from score_auxiliar(child, [c - k for k, c in zip(ks, Cs)], free_cards - sum(ks) + 1, node_probability)        


def score(tree_sequence: Node, Cs, minimum_probability = 0.0) -> float:
    return sum(score_auxiliar(tree_sequence, Cs, INITIAL_HAND_SIZE, 1.0, minimum_probability))
    
    

def possible_turn(turn, available_mana, remaining_cards):
    yield turn
    if remaining_cards > 0 and available_mana > 0:
        #only ascending order
        last_played = 1
        if len(turn) > 0:
            last_played = turn[-1]
            last_played = max(1, last_played) # we cannot repeat landrops
        for i in range(last_played, min(available_mana+1, MAXIMUM_MANA_VALUE+1)):
            yield from possible_turn(copy.deepcopy(turn + [i]), available_mana - i, remaining_cards - 1)
        


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
print("Generating tree")
with open('sequences.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")  
    root_node = Node()
    for sequence in tqdm(saved_combinations.values()):
        current_node = root_node
        writer.writerow([sequence_impact(sequence)] + sequence)
        for turn in sequence:
            if turn not in [child.turn for child in current_node.children]:
                current_node.children.append(Node(turn, turn_impact(turn), [])) # consider only turn impact
            current_node = current_node.children[[child.turn for child in current_node.children].index(turn)]

class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    

def hill_climbing(initial_combination, root_node):
    best_score = 0
    best_combination = initial_combination
    keep_exploring = True
    minimum_probabilities = [0.8, 0.3, 0.05, 0.01]
    resolution_index = 0
    last_direction = (0, 1)
    while keep_exploring:
        keep_exploring = False
        for (i, j) in tqdm([last_direction]+[(i,j) for j in range(len(best_combination)) for i in range(len(best_combination)) if i!=j]):
            combination = copy.deepcopy(best_combination)
            combination[j] += 1
            combination[i] -= 1
            if all([k >= 0 for k in combination]):
                combination_score = score(root_node, combination, minimum_probabilities[resolution_index])
                if combination_score > best_score:
                    last_direction = (i, j)
                    best_score = combination_score
                    keep_exploring = True
                    new_best_combination = combination
                    break
        if keep_exploring:
            best_combination = new_best_combination
            yield best_combination, best_score
        if not keep_exploring and resolution_index + 1 < len(minimum_probabilities):
            keep_exploring = True
            resolution_index += 1
            print("Increasing resolution index to", resolution_index)
    
current_strategy = Strategy.MULTI_HILL_CLIMBING

print("Selected strategy", current_strategy)

saved_curves = {}
with open('curves.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    best_score = 0.0
    if current_strategy == Strategy.FULL_EXPLORATION:
        for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE),
                                total=len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)))):
            Ks = [sum(1 for j in combination if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
            combination_score = score(root_node, Ks)
            writer.writerow([combination_score] + Ks)
            if combination_score > best_score:
                print("[FULL_EXPLORATION] Better score found:", combination_score, Ks)
                best_score = combination_score
    elif current_strategy == Strategy.HILL_CLIMBING:
        # initial solution and loop variables
        initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
        initial_combination[1] = DECK_SIZE
        for combination, combination_score in tqdm(hill_climbing(initial_combination, root_node)):
            if combination_score > best_score:
                #print("[HILL_CLIMBING] Better score found:", combination_score, combination)
                best_score = combination_score
                writer.writerow([combination_score] + combination)
                csvfile.flush()
    elif current_strategy == Strategy.MULTI_HILL_CLIMBING:
        possible_combinations = list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE))
        for _ in tqdm(range(10)):
            selected = random.choice(possible_combinations)
            Ks = [sum(1 for j in selected if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
            for combination, combination_score in tqdm(hill_climbing(Ks, root_node)):
                if combination_score > best_score:
                    #print("[MULTI_HILL_CLIMBING] Better score found:", combination_score, combination)
                    best_score = combination_score
                    writer.writerow([combination_score]+ combination)
                    csvfile.flush()
