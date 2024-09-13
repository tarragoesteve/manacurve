from itertools import combinations_with_replacement, count, permutations
from tqdm import tqdm
import csv

MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 6
FINAL_TURN = 4
print("Hello")
print(len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), INITIAL_HAND_SIZE + FINAL_TURN))))


def optimal_sequence(combination):
    lands = sum(1 for i in combination if i == 0)
    spells = [i for i in combination if i != 0]
    best_impact = 0
    best_permutation = []
    for permutation in permutations(spells):
        impact, length = sequence_impact(permutation, lands)
        if impact > best_impact:
            best_impact = impact
            best_permutation = permutation[:length]
    return best_permutation, best_impact



def sequence_impact(sequence, lands):
    i = 0
    total_impact = 0
    for turn in range(1,FINAL_TURN+1):
        available_mana = min(turn, lands)
        while available_mana > 0 and i < len(sequence):
            if available_mana >= sequence[i]:
                total_impact += impact(sequence[i])
                available_mana -= sequence[i]
                i +=1
            else:
                available_mana = 0 # exit while loop, we cannot play the card i in this turn
    return total_impact, i

def impact(mana_value):
    return mana_value+2

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)    
    for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), INITIAL_HAND_SIZE + FINAL_TURN)):
        ks = []
        for k in range(MAXIMUM_MANA_VALUE+1):            
            ks.append(sum(1 for i in combination if i == k))
        combination_optimal_sequence, combination_impact =  optimal_sequence(combination)
        writer.writerow(ks + [combination_impact] + list(combination_optimal_sequence))
    #print("Combination", combination, "Optimal sequence", combination_optimal_sequence, "impact", combination_impact)



