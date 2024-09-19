import csv
import itertools
from tqdm import tqdm
from itertools import combinations_with_replacement
import math
import copy

DECK_SIZE = 60
MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 6

class Sequence:
    def __init__(self, impact = 0, turns = []) -> None:
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


def probability(sequence: Sequence, Ks) -> float:
    total_probability = 1.0
    free_cards = INITIAL_HAND_SIZE
    for turn in sequence.turns:
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
                for combination in itertools.product(*[range(n+1) for n in ks]):
                    if all([combination[i] >= ks[i] for i in range(len(combination))]):
                        # This is the last combination and should be skipped as it is the first one to accomplish the restriction
                        pass
                    else:
                        k_other = free_cards - sum(combination)
                        K_other = sum(Ks) - sum(Ks[i] for i in ks_index)
                        numerator = 1
                        for i, ki in enumerate(combination):
                            Ki = Ks[ks_index[i]]
                            numerator *= math.comb(Ki, ki)
                        if k_other > 0 and K_other > 0:
                            numerator *= math.comb(K_other, k_other)
                        denominator = math.comb(sum(Ks), k_other + sum(combination))
                        turn_conditioned_probability -= numerator/denominator
        total_probability *= turn_conditioned_probability
        # update unknown cards information after turn
        free_cards -= sum(ks)
        # update deck information after turn
        for i, ki in enumerate(ks):
            Ks[ks_index[i]] -= ki
        print(turn, ks, ks_index, turn_conditioned_probability, total_probability)

    return total_probability

Ks = [20,25,0,10,5] # K0, K1, K2, K3, K4
# print(sequence.turns)
# print(probability(sequence, copy.deepcopy(Ks)))
# print(sequence.impact)

# for sequence in sequences:
#     print(sequence.impact, sequence.turns, probability(sequence, copy.deepcopy(Ks)))
# score = sum([sequence.impact * probability(sequence, copy.deepcopy(Ks)) for sequence in sequences[71:72]])
# print(score)

# print("Will compute", len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE))))

sequence = Sequence(1, [[1, 0],[1, 1, 0], [3,0]])
Ks = [30, 20, 0, 10, 0]

print("Ks",Ks)
print("Seq",sequence.turns)
print("Prob", probability(sequence, copy.deepcopy(Ks)))

# with open('curves.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=";")
#     best_score = 0.0
#     for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)):
#         Ks = []
#         for k in range(MAXIMUM_MANA_VALUE+1):            
#             Ks.append(sum(1 for i in combination if i == k))
#         #score = sum([sequence.impact * probability(sequence, copy.deepcopy(Ks)) for sequence in sequences])
#         score = sequence.impact * probability(sequence, copy.deepcopy(Ks))
#         writer.writerow(Ks + [score])
#         if score > best_score:
#             print("New best score found:", score, Ks)
#             best_score = score