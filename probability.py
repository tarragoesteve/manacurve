import csv
import itertools
from tqdm import tqdm
from itertools import combinations_with_replacement
import math
import copy
from scipy.stats import multivariate_hypergeom
from enum import Enum
import random

DECK_SIZE = 40
MAXIMUM_MANA_VALUE = 4

class Sequence:
    def __init__(self, impact = 0.0, turns = []) -> None:
        self.impact = impact
        self.turns = turns

sequences = []
# Parse input
with open('sequences_4turns.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for row in reader:
        impact = float(row[0])
        turns = []
        for column in row[1:]:
            remove_brackets = column[1:-1]
            if len(remove_brackets) > 0:
                turns.append([int(i) for i in column[1:-1].split(',')])
            else:
                turns.append([])
        sequences.append(Sequence(impact, turns))

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

def probability(sequence: Sequence, Ks, initial_hand_size: int =  6) -> float:
    total_probability = 1.0
    free_cards = initial_hand_size
    for turn_number,turn in enumerate(sequence.turns):
        free_cards += 1 # draw of the turn
        turn_conditioned_probability = 1.0
        ks = []
        ks_index = []
        for i in range(MAXIMUM_MANA_VALUE+1):
            ki = sum(1 for j in turn if j == i)            
            if ki > 0:
                ks_index.append(i)
                ks.append(ki)
        # ks now the minimum number of cards of each type in the turn
        # ks_index contains the information on the corresponding index
        # the restriction for the turn is that we need to draw at least ks[i] cards of type ks_index[i]
        if len(ks) > 0:
            if any([ks > Ks[ks_index[i]] for i, ks in enumerate(ks)]) or sum(ks) > free_cards:
                turn_conditioned_probability = 0
            else:
                # Generate the product of iterations
                for combination in non_valid_possible_combinations(ks, Ks, free_cards):
                    k_other = free_cards - sum(combination)
                    K_other = sum(Ks) - sum(Ks[i] for i in ks_index)
                    if K_other < k_other:
                        # This combination is passing the requirements or is impossible
                        # print("Epa!")
                        pass
                    else:
                        combination_prob = multivariate_hypergeom.pmf(x=[i for i in combination]+[k_other], m=[Ks[i] for i in ks_index]+[K_other], n=free_cards)
                        turn_conditioned_probability -= combination_prob
        total_probability *= turn_conditioned_probability
        #print("Turn", turn_number, "Conditioned", turn_conditioned_probability, "Total", total_probability, "Ks", Ks)
        # update unknown cards information after turn
        free_cards -= sum(ks)
        # update deck information after turn
        for i, ki in enumerate(ks):
            Ks[ks_index[i]] -= ki
    return total_probability


class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    
current_strategy = Strategy.MULTI_HILL_CLIMBING

def hill_climbing(initial_combination, sequences):
    best_score = 0
    best_combination = initial_combination
    keep_exploring = True
    while keep_exploring:
        keep_exploring = False
        for i, _ in enumerate(best_combination):
            for j, _ in enumerate(best_combination):
                if i != j:
                    combination = copy.deepcopy(best_combination)
                    combination[j] += 1
                    combination[i] -= 1
                    if all([k >= 0 for k in combination]):
                        score = sum([sequence.impact * probability(sequence, copy.deepcopy(combination)) for sequence in sequences])
                        if score > best_score:
                            best_score = score
                            keep_exploring = True
                            new_best_combination = combination
        if keep_exploring:
            best_combination = new_best_combination
            print("New best score found:", score, best_combination)
    return best_combination, best_score
    
total = len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)))
print("Will compute", total)
with open('curves.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    best_score = 0.0
    if current_strategy == Strategy.FULL_EXPLORATION:
        for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE), total=total):
            Ks = []
            for k in range(MAXIMUM_MANA_VALUE+1):
                Ks.append(sum(1 for i in combination if i == k))
            score = sum([sequence.impact * probability(sequence, copy.deepcopy(Ks)) for sequence in sequences])
            writer.writerow(Ks + [score])
            if score > best_score:
                print("New best score found:", score, Ks)
                best_score = score
    elif current_strategy == Strategy.HILL_CLIMBING:
        # initial solution and loop variables
        initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
        initial_combination[1] = DECK_SIZE
        hill_climbing(initial_combination, sequences)
    elif current_strategy == Strategy.MULTI_HILL_CLIMBING:
        best_score = 0
        best_combination = []
        possible_combinations = list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE))
        for _ in tqdm(range(10)):
            selected = random.choice(possible_combinations)
            Ks = []
            for k in range(MAXIMUM_MANA_VALUE+1):
                Ks.append(sum(1 for i in selected if i == k))
            initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
            initial_combination[1] = DECK_SIZE
            combination, score = hill_climbing(Ks, sequences)
            writer.writerow(best_combination + [best_score])
            if score > best_score:
                print("Better combination found:", best_combination, best_score)
                best_score = score
                best_combination = combination