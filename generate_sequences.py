from itertools import combinations_with_replacement, count, permutations
from tqdm import tqdm
import csv
import copy

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
        impact, sequence_by_turns = sequence_impact(permutation, lands)
        if impact > best_impact:
            best_impact = impact
            best_permutation = sequence_by_turns
    return best_permutation, best_impact



def sequence_impact(sequence, lands):
    i = 0
    total_impact = 0
    sequence_by_turns = []
    remaining_mana = []
    for turn in range(FINAL_TURN):
        available_mana = min(turn+1, lands)
        played_this_turn = []
        while i < len(sequence) and available_mana >= sequence[i]:
            played_this_turn.append(sequence[i])
            total_impact += impact(sequence[i])
            available_mana -= sequence[i]
            i +=1
        remaining_mana.append(available_mana)
        sequence_by_turns.append(played_this_turn)
    
    # Play the lands at the latest possible moment to increase changes of having the sequence
    needed_lands = []
    for turn in range(FINAL_TURN):
        available_mana = min(turn+1, lands)
        needed_lands.append(available_mana - remaining_mana[turn])
    #propagate the needed landrops to the previous turns
    for i in reversed(range(1,len(needed_lands))):
        needed_lands[i-1] = max(needed_lands[i]-1, needed_lands[i-1])

    for i in range(FINAL_TURN):
        if i == 0:
            if needed_lands[0] == 1:
                sequence_by_turns[i].append(0)
        else:
            if needed_lands[i] > needed_lands[i-1]:
                sequence_by_turns[i].append(0)
    return total_impact, sequence_by_turns

def impact(mana_value):
    return mana_value+1.5

with open('sequences.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    saved_combinations = {}    
    for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), INITIAL_HAND_SIZE + FINAL_TURN)):
        ks = []
        for k in range(MAXIMUM_MANA_VALUE+1):            
            ks.append(sum(1 for i in combination if i == k))
        if ks[0] != 0:
            #if there are no lands impact will always be 0
            combination_optimal_sequence, combination_impact =  optimal_sequence(combination)
            if combination_impact > 0:
                if str(combination_optimal_sequence) not in saved_combinations:
                    writer.writerow([combination_impact] + list(combination_optimal_sequence))
                    saved_combinations[str(combination_optimal_sequence)] = True


