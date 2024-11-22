from typing import List
from itertools import combinations_with_replacement
import copy
from tqdm import tqdm
import csv
import random
from config import MAXIMUM_MANA_VALUE, DECK_SIZE, DECK_MINIMUMS, STRATEGY, Strategy, MULLIGAN_THRESHOLD, MULTI_HILL_CLIMBING_ITERATIONS
from draw_tree import DrawTree
from deck_probability import DeckProbability


# for hill climbing
initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
initial_combination[0] = DECK_SIZE


class Optimizer():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def hill_climbing(initial_combination : List[int] , draw_node : DrawTree):
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
                    combination_score = DeckProbability.score(draw_node, combination, MULLIGAN_THRESHOLD)
                    if combination_score > best_score:
                        best_score = combination_score
                        keep_exploring = True
                        new_best_combination = combination
                        #break
            if keep_exploring:
                best_combination = new_best_combination
                yield best_combination, best_score
        
    @staticmethod
    def run(draw_tree: DrawTree, output_file = 'curves.csv', results_file = 'results.csv'):
        print("Selected strategy", STRATEGY)
        if STRATEGY == Strategy.NOTHING:
            print("Nothing to do")
        else:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                best_score = 0.0
                best_combination = []
                if STRATEGY == Strategy.FULL_EXPLORATION:
                    for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE),
                                            total=len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)))):
                        Ks = [sum(1 for j in combination if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
                        combination_score = DeckProbability.score(draw_tree, Ks)
                        writer.writerow([combination_score] + Ks)
                        if combination_score > best_score:
                            print("[FULL_EXPLORATION] Better score found:", combination_score, Ks)
                            best_score = combination_score
                            best_combination = Ks
                elif STRATEGY == Strategy.HILL_CLIMBING:
                    # initial solution and loop variables
                    for combination, combination_score in tqdm(Optimizer.hill_climbing(initial_combination, draw_tree)):
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
                        for combination, combination_score in tqdm(Optimizer.hill_climbing(Ks, draw_tree)):
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
                    combination = [30, 10,10,10]
                    combination_score = DeckProbability.score(draw_tree, combination, MULLIGAN_THRESHOLD)
                    print("Combination score", combination_score)
                    best_combination = combination
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                total_probability = 0
                results = {}
                recover_sequence = {}
                for probability, _, sequence, _  in DeckProbability.recursive_tree_probability(draw_tree, best_combination, 1.0, -1):
                    if not str(sequence) in results:
                        results[str(sequence)] = 0.0
                        recover_sequence[str(sequence)] = sequence
                    results[str(sequence)] += probability
                    total_probability += probability
                for sequence_str, probability in results.items():
                    writer.writerow([round(probability,8)] + recover_sequence[sequence_str])
                print("Total probability", total_probability)

