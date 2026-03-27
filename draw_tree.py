from typing import List
from sequence_list import SequenceList
import copy
from tqdm import tqdm
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN

class CommonTree():
    def __init__(self):
        self.available_cards = []
        self.expected_impact = 0
        self.probability = 0.0
        self.children: List[CommonTree] = []
        self.best_child_impact = 0.0

class RootTree(CommonTree):
    def __init__(self):
        super().__init__()
        initial_hands = list(possible_initial_hands(INITIAL_HAND_SIZE,MAXIMUM_MANA_VALUE))
        for hand in tqdm(initial_hands):
            self.children.append(InitialHand(hand))

    def populate_tree(self, sequence_list: SequenceList):
        print("Populating tree...")
        for impact, sequence in tqdm(sequence_list.sequences):
            self.populate_tree_from_sequence(sequence, impact)
        print("Hands with best child impact > 0:", sum(1 for hand in self.children if hand.best_child_impact > 0), "/", len(self.children))
        print("Hands with childrens > 0:", sum(1 for hand in self.children if len(hand.children) > 0), "/", len(self.children))

    def from_sparse_to_vector(self, drawn_cards: List[int]):
        available_cards = [0]*(MAXIMUM_MANA_VALUE+1)
        for card in drawn_cards:
            available_cards[card] += 1
        return available_cards

    def populate_tree_from_sequence(self, sequence : List[List[int]], impact: float):
        for hand in self.children:
            required_cards_for_first_turn = self.from_sparse_to_vector(sequence[0])
            if all(hand.available_cards[i] >= required_cards_for_first_turn[i] for i in range(MAXIMUM_MANA_VALUE+1)):
                available_cards = copy.deepcopy(hand.available_cards)
                for i in range(MAXIMUM_MANA_VALUE+1):
                    available_cards[i] -= required_cards_for_first_turn[i]
                # This hand can play the first turn of the sequence.
                if self.populate_tree_recursive(sequence[1:], impact, available_cards, hand):
                    hand.best_child_impact = max(hand.best_child_impact, impact)

    def populate_tree_recursive(self, sequence : List[List[int]], impact: float, available_cards: List[int], parent_tree: CommonTree):
        if len(sequence) == 0:
            return True

        required_cards = self.from_sparse_to_vector(sequence[0])

        has_any_successful_child = False
        for i in range(MAXIMUM_MANA_VALUE+1):
            child = self.get_child_with_draw(parent_tree, i)
            is_new = child is None
            if is_new:
                child = DrawTree(i)

            # Draw card i first, then check if we can play this turn
            child_available_cards = list(available_cards)
            child_available_cards[i] += 1

            if any(child_available_cards[j] < required_cards[j] for j in range(MAXIMUM_MANA_VALUE+1)):
                continue  # Can't play this turn with this draw

            # Subtract played cards
            for j in range(MAXIMUM_MANA_VALUE+1):
                child_available_cards[j] -= required_cards[j]

            if self.populate_tree_recursive(sequence[1:], impact, child_available_cards, child):
                if len(child.children) == 0:
                    child.impact = max(child.impact, impact)
                child.best_child_impact = max(child.best_child_impact, impact)
                if is_new:
                    parent_tree.children.append(child)
                has_any_successful_child = True
        return has_any_successful_child

    @staticmethod
    def get_child_with_draw(parent: CommonTree, draw: int):
        for child in parent.children:
            if isinstance(child, DrawTree) and child.draw == draw:
                return child
        return None


class InitialHand(CommonTree):
    def __init__(self, hand : List[int]):
        super().__init__()
        self.available_cards = hand
        pass

class DrawTree(CommonTree):
    def __init__(self, draw, impact = 0.0) -> None:
        super().__init__()
        self.draw: int = draw
        self.impact: float = impact

###INITIAL HANDS
def possible_initial_hands(hand_size: int, maximum_mana_value: int):
    yield from possible_initial_hands_aux(hand_size, maximum_mana_value, 0, [])

def possible_initial_hands_aux(hand_size: int, maximum_mana_value: int, mana_value: int, current_hand: List[int]):
    if mana_value == maximum_mana_value:
        yield current_hand + [hand_size]
    else:
        for i in range(hand_size+1):
            yield from possible_initial_hands_aux(hand_size-i, maximum_mana_value, mana_value+1, [i]+current_hand)
