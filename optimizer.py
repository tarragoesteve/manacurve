from typing import List, Tuple
from itertools import combinations_with_replacement
import copy
from tqdm import tqdm
import csv
import random
from config import MAXIMUM_MANA_VALUE, DECK_SIZE, DECK_MINIMUMS, STRATEGY, Strategy, MULLIGAN_THRESHOLD, MULTI_HILL_CLIMBING_ITERATIONS, INITIAL_COMBINATION
from draw_tree import RootTree
from deck_probability import DeckProbability





class Optimizer():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _weighted_score(weighted_trees: List[Tuple[float, RootTree]], combination: List[int]) -> float:
        return sum(w * DeckProbability.score(tree, combination, MULLIGAN_THRESHOLD)
                   for w, tree in weighted_trees)

    @staticmethod
    def _score_breakdown(weighted_trees: List[Tuple[float, RootTree]], combination: List[int]):
        """Returns list of (turn, weight, raw_score, weighted_score) for each tree."""
        rows = []
        for w, tree in weighted_trees:
            raw = DeckProbability.score(tree, combination, MULLIGAN_THRESHOLD)
            rows.append((tree.turn, w, raw, w * raw))
        return rows

    @staticmethod
    def _print_breakdown(weighted_trees: List[Tuple[float, RootTree]], combination: List[int]):
        total = 0.0
        print(f"  {'Turn':>6}  {'Weight':>8}  {'Score':>10}  {'Contribution':>13}")
        print(f"  {'----':>6}  {'------':>8}  {'-----':>10}  {'------------':>13}")
        for turn, w, raw, contrib in Optimizer._score_breakdown(weighted_trees, combination):
            total += contrib
            print(f"  {turn:>6}  {w:>8.3f}  {raw:>10.4f}  {contrib:>13.4f}")
        print(f"  {'':>6}  {'':>8}  {'Total:':>10}  {total:>13.4f}")

    @staticmethod
    def hill_climbing(initial_combination: List[int], weighted_trees: List[Tuple[float, RootTree]]):
        best_score = 0
        best_combination = initial_combination
        keep_exploring = True
        while keep_exploring:
            keep_exploring = False
            # TODO: Use config MAXIMUM_MANA_VALUE instead of len(best_combination)
            directions = [(i, j) for j in range(len(best_combination)) for i in range(len(best_combination)) if i!=j]
            for (i, j) in tqdm(directions):
                combination = copy.deepcopy(best_combination)
                combination[j] += 1
                combination[i] -= 1
                if all([k >= DECK_MINIMUMS[index] for index,k in enumerate(combination)]):
                    combination_score = Optimizer._weighted_score(weighted_trees, combination)
                    if combination_score > best_score:
                        best_score = combination_score
                        keep_exploring = True
                        new_best_combination = combination
                        #break
            if keep_exploring:
                best_combination = new_best_combination
                yield best_combination, best_score
        
    @staticmethod
    def run(weighted_trees: List[Tuple[float, RootTree]], output_file = 'curves.csv'):
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
                        combination_score = Optimizer._weighted_score(weighted_trees, Ks)
                        writer.writerow([combination_score] + Ks)
                        if combination_score > best_score:
                            print("[FULL_EXPLORATION] Better score found:", combination_score, Ks)
                            best_score = combination_score
                            best_combination = Ks
                elif STRATEGY == Strategy.HILL_CLIMBING:
                    # initial solution and loop variables
                    for combination, combination_score in tqdm(Optimizer.hill_climbing(INITIAL_COMBINATION, weighted_trees)):
                        if combination_score > best_score:
                            best_score = combination_score
                            best_combination = combination
                            writer.writerow([combination_score] + combination)
                            csvfile.flush()
                elif STRATEGY == Strategy.MULTI_HILL_CLIMBING:
                    cached_combinations = {}
                    for _ in tqdm(range(MULTI_HILL_CLIMBING_ITERATIONS)):
                        Ks = Optimizer.get_random_deck()
                        for combination, combination_score in tqdm(Optimizer.hill_climbing(Ks, weighted_trees)):
                            writer.writerow([combination_score]+ combination)
                            csvfile.flush()
                            if combination_score > best_score:
                                #print("[MULTI_HILL_CLIMBING] Better score found:", combination_score, combination)
                                best_score = combination_score
                                best_combination = combination
                            if str(combination) not in cached_combinations:
                                cached_combinations[str(combination)] = combination_score
                            else:
                                print("Stopping iteration, we arrived at", combination)
                                break
                elif STRATEGY == Strategy.SINGLE:
                    combination_score = Optimizer._weighted_score(weighted_trees, INITIAL_COMBINATION)
                    print("Combination score", combination_score)
                    best_combination = INITIAL_COMBINATION
            print("Best score:", best_score, "Best combination:", best_combination)
            if best_combination:
                print("\nPer-turn score breakdown for best combination:")
                Optimizer._print_breakdown(weighted_trees, best_combination)
    
    @staticmethod
    def get_random_deck():
        deck = [0] * (MAXIMUM_MANA_VALUE+1)
        while sum(deck) < DECK_SIZE:
            card = random.randint(0, MAXIMUM_MANA_VALUE)
            deck[card] += 1
        return deck
