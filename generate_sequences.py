from itertools import combinations_with_replacement, count, permutations
from tqdm import tqdm
import csv
import copy

MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 6
FINAL_TURN = 5

def sequence_impact(sequence):
    total = 0.0
    for turn in sequence:
        total += sum([impact(i) for i in turn])
    return total
        
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
    yield from recursive_auxiliar(0, 0, [], INITIAL_HAND_SIZE+1)

def recursive_auxiliar(turn, landrops, current_sequence, remaining_cards):
    if turn == FINAL_TURN:
        yield current_sequence
    else:
        # case we do not play land
        for turn_sequence in possible_turn([], landrops, remaining_cards):
            yield from recursive_auxiliar(turn + 1, landrops, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
        # case we play a land
        for turn_sequence in possible_turn([0], landrops + 1, remaining_cards-1):
            yield from recursive_auxiliar(turn + 1, landrops + 1, copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
    
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
        


with open('sequences.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    saved_combinations = {}
    for sequence in tqdm(possible_sequences()):
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
    for sequence in saved_combinations.values():
        writer.writerow([sequence_impact(sequence)] + sequence)


