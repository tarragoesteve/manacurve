from scipy.stats import multivariate_hypergeom
from typing import List
from draw_tree import RootTree, DrawTree


class DeckProbability:
    @staticmethod
    def score(root_tree: RootTree, Cs: List[int], mulligan_threshold=0.0):
        for initial_hand in root_tree.children:
            hand_probability = multivariate_hypergeom.pmf(
                x=initial_hand.available_cards, m=Cs, n=sum(initial_hand.available_cards)
            )
            initial_hand.probability = hand_probability

            if hand_probability > 0:
                remaining_Cs = [c - k for c, k in zip(Cs, initial_hand.available_cards)]
                for child in initial_hand.children:
                    DeckProbability._recursive_probability(child, remaining_Cs, hand_probability)
                initial_hand.expected_impact = sum(
                    child.probability * child.expected_impact for child in initial_hand.children
                )
                #print("Hand", initial_hand.available_cards, "has probability", hand_probability, "and expected impact", initial_hand.expected_impact)
            else:
                initial_hand.expected_impact = 0.0

        return DeckProbability._score_with_mulligan(root_tree, mulligan_threshold)

    @staticmethod
    def _recursive_probability(draw_tree: DrawTree, Cs: List[int], parent_probability: float):
        total_cards = sum(Cs)
        if total_cards > 0 and Cs[draw_tree.draw] > 0:
            draw_probability = Cs[draw_tree.draw] / total_cards
        else:
            draw_probability = 0.0

        draw_tree.probability = draw_probability

        if draw_probability > 0:
            if len(draw_tree.children) == 0:
                draw_tree.expected_impact = draw_tree.impact
            else:
                remaining_Cs = list(Cs)
                remaining_Cs[draw_tree.draw] -= 1
                for child in draw_tree.children:
                    DeckProbability._recursive_probability(child, remaining_Cs, draw_probability)
                draw_tree.expected_impact = sum(
                    child.probability * child.expected_impact for child in draw_tree.children
                )
        else:
            draw_tree.expected_impact = 0.0

    @staticmethod
    def _score_with_mulligan(root_tree: RootTree, mulligan_threshold=0.0):
        accumulated_probability = 0.0
        accumulated_expected_impact = 0.0
        hands = [(hand.expected_impact, hand.probability) for hand in root_tree.children if hand.probability > 0]
        hands.sort(key=lambda x: sum(x), reverse=True)
        while accumulated_probability <= (1 - mulligan_threshold) and len(hands) > 0:
            hand = hands.pop(0)
            accumulated_expected_impact += hand[0] * hand[1]
            accumulated_probability += hand[1]
        if accumulated_probability > 1e-10:
            return accumulated_expected_impact / accumulated_probability
        return 0.0



