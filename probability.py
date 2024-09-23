import csv
import itertools
from tqdm import tqdm
from itertools import combinations_with_replacement
import math
import copy
from scipy.stats import multivariate_hypergeom

DECK_SIZE = 56
MAXIMUM_MANA_VALUE = 4

class Sequence:
    def __init__(self, impact = 0.0, turns = []) -> None:
        self.impact = impact
        self.turns = turns

sequences = []
# Parse input
with open('sequences.csv', 'r', newline='') as csvfile:
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


# sequence = Sequence(100.0, [[1, 0], [2, 0], [3, 0], [4, 0]])
# Ks =  [50, 3, 3, 3, 1]
# probability(sequence, copy.deepcopy(Ks))

# print(list(non_valid_possible_combinations([1,2],[4,6], 5)))

print("Will compute", len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE))))
with open('curves.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    best_score = 0.0
    for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)):
        Ks = []
        for k in range(MAXIMUM_MANA_VALUE+1):            
            Ks.append(sum(1 for i in combination if i == k))
        score = sum([sequence.impact * probability(sequence, copy.deepcopy(Ks)) for sequence in sequences])
        #score = sequence.impact * probability(sequence, copy.deepcopy(Ks))
        writer.writerow(Ks + [score])
        if score > best_score:
            print("New best score found:", score, Ks)
            best_score = score