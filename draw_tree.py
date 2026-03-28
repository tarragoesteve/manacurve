from typing import List
from sequence_list import SequenceList
import copy
import json
import os
from tqdm import tqdm
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN, TREES_DIR

class CommonTree():
    def __init__(self):
        self.available_cards = []
        self.expected_impact = 0
        self.probability = 0.0
        self.children: List[CommonTree] = []
        self.best_child_impact = 0.0

class RootTree(CommonTree):
    def __init__(self, turn: int = FINAL_TURN):
        super().__init__()
        self.turn = turn
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

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "children": [RootTree._initial_hand_to_dict(h) for h in self.children],
        }

    @staticmethod
    def _initial_hand_to_dict(hand: 'InitialHand') -> dict:
        return {
            "available_cards": hand.available_cards,
            "best_child_impact": hand.best_child_impact,
            "children": [RootTree._draw_tree_to_dict(c) for c in hand.children],
        }

    @staticmethod
    def _draw_tree_to_dict(node: 'DrawTree') -> dict:
        return {
            "draw": node.draw,
            "impact": node.impact,
            "best_child_impact": node.best_child_impact,
            "children": [RootTree._draw_tree_to_dict(c) for c in node.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RootTree':
        rt = cls.__new__(cls)
        CommonTree.__init__(rt)
        rt.turn = data["turn"]
        rt.children = [cls._initial_hand_from_dict(h) for h in data["children"]]
        return rt

    @staticmethod
    def _initial_hand_from_dict(data: dict) -> 'InitialHand':
        hand = InitialHand.__new__(InitialHand)
        CommonTree.__init__(hand)
        hand.available_cards = data["available_cards"]
        hand.best_child_impact = data["best_child_impact"]
        hand.children = [RootTree._draw_tree_from_dict(c) for c in data["children"]]
        return hand

    @staticmethod
    def _draw_tree_from_dict(data: dict) -> 'DrawTree':
        node = DrawTree.__new__(DrawTree)
        CommonTree.__init__(node)
        node.draw = data["draw"]
        node.impact = data["impact"]
        node.best_child_impact = data["best_child_impact"]
        node.children = [RootTree._draw_tree_from_dict(c) for c in data["children"]]
        return node

    def save_to_file(self, path: str = None):
        if path is None:
            path = f'{TREES_DIR}/tree_turn_{self.turn}.json'
        print(f"Saving tree to {path}...")
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_from_file(cls, turn: int, path: str = None) -> 'RootTree':
        if path is None:
            path = f'{TREES_DIR}/tree_turn_{turn}.json'
        print(f"  Loading tree from {path}...", end=' ', flush=True)
        with open(path, 'r') as f:
            data = json.load(f)
        rt = cls.from_dict(data)
        print(f"({len(rt.children)} initial hands)")
        return rt

    @classmethod
    def load_or_build(cls, turn: int, sequence_list: SequenceList) -> 'RootTree':
        path = f'{TREES_DIR}/tree_turn_{turn}.json'
        if os.path.isfile(path):
            return cls.load_from_file(turn, path)
        rt = cls(turn=turn)
        rt.populate_tree(sequence_list)
        rt.save_to_file(path)
        return rt


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
