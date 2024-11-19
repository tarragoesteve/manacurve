from typing import List
from sequence_tree import SequenceTree
from itertools import combinations_with_replacement
from scipy.stats import multivariate_hypergeom
import copy
from tqdm import tqdm
from enum import Enum
import csv


MAXIMUM_MANA_VALUE = 5
INITIAL_HAND_SIZE = 7
MULLIGAN_THRESHOLD = 0.08
FINAL_TURNS = [6]
DECK_SIZE = 60
DECK_MINIMUMS = [0] * (MAXIMUM_MANA_VALUE+1)
MULTI_HILL_CLIMBING_ITERATIONS = 3


class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    SINGLE = 3
    NOTHING = 4


STRATEGY = Strategy.SINGLE

# for hill climbing
initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
initial_combination[0] = DECK_SIZE




class Optimizer():
    def __init__(self) -> None:
        pass







    # Score methods
    def score(draw_tree : DrawTree, Cs: List[int], mulligan_threshold = 0.0):
        for _ in Optimizer.score_auxiliar(draw_tree, Cs, 1.0, -1):
            pass
        return Optimizer.score_with_mulligan(draw_tree, mulligan_threshold)

    def score_auxiliar(draw_tree : DrawTree, Cs: List[int], probability : float, current_turn: int):
        turn_probability = probability
        Ks = []
        for i in range(len(Cs)):
            Ks.append(sum([1 for card in draw_tree.draw if card == i]))
        if any([k > c for k,c in zip(Ks, Cs)]):
            turn_probability = 0.0
        if sum(Ks) > 1:
            turn_probability *= multivariate_hypergeom.pmf(x=Ks, m=Cs, n=sum(Ks))
        elif len(draw_tree.draw) == 1:
            turn_probability *= Cs[draw_tree.draw[0]] / sum(Cs)
        else:
            #draw is empty, initial state
            pass

        draw_tree.probability = turn_probability
        new_Cs = [c-k for k,c in zip(Ks, Cs)]
        if turn_probability > 0 and draw_tree.best_sequence.impact > 0:
            if current_turn in FINAL_TURNS:
                draw_tree.expected_impact[current_turn] = draw_tree.best_sequence.impact
                yield turn_probability, draw_tree.best_sequence.impact, draw_tree.best_sequence.sequence, current_turn
            for child in draw_tree.children:
                yield from Optimizer.score_auxiliar(child, new_Cs, turn_probability, current_turn+1)
            for turn in [turn for turn in FINAL_TURNS if turn != current_turn]:
                accumulated_impact = 0.0
                for child in draw_tree.children:
                    try:
                        accumulated_impact += child.probability * child.expected_impact[turn]
                    except KeyError:
                        pass
                draw_tree.expected_impact[turn] = accumulated_impact / draw_tree.probability
                    
    def score_with_mulligan(draw_tree : DrawTree, mulligan_threshold = 0.0, turn_weight = {}):
        accumulated_probability = 0.0
        accumulated_expected_impact = 0.0
        hands = [(hand.expected_impact, hand.probability) for hand in draw_tree.children if hand.probability > 0]
        fixed_hands = 0
        for hand in hands:
            for turn in FINAL_TURNS:
                if turn not in hand[0]:
                    hand[0][turn] = 0.0
                    fixed_hands += 1
                #raise NameError("Error")
        if fixed_hands > 0:
            print("Fixed hands", fixed_hands, "out of", len(hands))
        
        hands.sort(key=lambda x: sum([x[0][turn] * turn_weight[turn] for turn in FINAL_TURNS]), reverse=True)
        while accumulated_probability < (1- mulligan_threshold) and len(hands) > 0:
            hand = hands.pop(0)
            accumulated_expected_impact += sum([hand[0][turn] * turn_weight[turn] for turn in FINAL_TURNS]) * hand[1]
            accumulated_probability += hand[1]
        if accumulated_probability > 1e-10:
            deck_expected_impact = accumulated_expected_impact / accumulated_probability
        else:
            deck_expected_impact = 0.0
        return deck_expected_impact



    
def hill_climbing(initial_combination, draw_node):
    best_score = 0
    best_combination = initial_combination
    keep_exploring = True
    while keep_exploring:
        keep_exploring = False
        directions = [(i, j) for j in range(len(best_combination)) for i in range(len(best_combination)) if i!=j]
        for (i, j) in tqdm(directions):
            combination = copy.deepcopy(best_combination)
            combination[j] += 1
            combination[i] -= 1
            if all([k >= DECK_MINIMUMS[index] for index,k in enumerate(combination)]):
                combination_score = score(draw_node, combination, MULLIGAN_THRESHOLD)
                if combination_score > best_score:
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
                combination_score = Optimizer.score(sequence_tree, Ks)
                writer.writerow([combination_score] + Ks)
                if combination_score > best_score:
                    print("[FULL_EXPLORATION] Better score found:", combination_score, Ks)
                    best_score = combination_score
                    best_combination = Ks
        elif STRATEGY == Strategy.HILL_CLIMBING:
            # initial solution and loop variables
            for combination, combination_score in tqdm(hill_climbing(initial_combination, draw_tree)):
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
                for combination, combination_score in tqdm(hill_climbing(Ks, draw_tree)):
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
            combination = [21,18,8,10,3]
            minimum_probabilities = [0.8, 0.3, 0.05, 0.01, 0]
            for p in tqdm(minimum_probabilities):
                combination_score = score(sequence_tree, combination, p)
                print(combination_score)
            best_combination = combination
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        total_probability = 0
        results = {}
        recover_sequence = {}
        for probability, _, sequence, _  in score_auxiliar(draw_tree, best_combination, 1.0, -1):
            if not str(sequence) in results:
                results[str(sequence)] = 0.0
                recover_sequence[str(sequence)] = sequence
            results[str(sequence)] += probability
            total_probability += probability
        for sequence_str, probability in results.items():
            writer.writerow([round(probability,8)] + recover_sequence[sequence_str])
        print("Total probability", total_probability)

