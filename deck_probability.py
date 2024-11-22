from scipy.stats import multivariate_hypergeom
from typing import List
from draw_tree import DrawTree

from config import FINAL_TURN


class DeckProbability:
    @staticmethod
    def score(draw_tree : DrawTree, Cs: List[int], mulligan_threshold = 0.0):
        for _ in DeckProbability.recursive_tree_probability(draw_tree, Cs, 1.0, -1):
            pass
        return DeckProbability.score_with_mulligan(draw_tree, mulligan_threshold)
    
    @staticmethod
    def recursive_tree_probability(draw_tree : DrawTree, Cs: List[int], probability : float, current_turn: int):
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
        new_Cs = [c-k for k,c in zip(Ks, Cs)] # update Cs to reflect deck after draw

        if turn_probability > 0 and draw_tree.best_child_impact > 0:
            if current_turn == FINAL_TURN:
                draw_tree.expected_impact = draw_tree.best_child_impact
                yield turn_probability, draw_tree.best_child_impact, current_turn
            for child in draw_tree.children:
                yield from DeckProbability.recursive_tree_probability(child, new_Cs, turn_probability, current_turn+1)
            draw_tree.expected_impact = sum([child.probability * child.expected_impact for child in draw_tree.children]) / turn_probability


    @staticmethod                
    def score_with_mulligan(draw_tree : DrawTree, mulligan_threshold = 0.0):
        accumulated_probability = 0.0
        accumulated_expected_impact = 0.0
        hands = [(hand.expected_impact, hand.probability) for hand in draw_tree.children if hand.probability > 0]
        hands.sort(key=lambda x: sum(x), reverse=True) # sort by expected impact
        while accumulated_probability < (1- mulligan_threshold) and len(hands) > 0:
            hand = hands.pop(0)
            accumulated_expected_impact += hand[1]
            accumulated_probability += hand[1]
        # if len(hands) > 0:
        #     print("First mulligan hand expected impact is", hands[0][0])
        if accumulated_probability > 1e-10:
            deck_expected_impact = accumulated_expected_impact / accumulated_probability
        else:
            deck_expected_impact = 0.0
        return deck_expected_impact



