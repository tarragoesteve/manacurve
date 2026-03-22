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
    def __init__(self, draw, impact = 0.0, children = None) -> None:
        super().__init__()
        self.draw: int = draw
        self.impact: float = impact

    # TODO: Deprecate this method
    def populate_tree(self):
        print("Generating draw tree")
        for initial_hand in tqdm(possible_initial_hands(INITIAL_HAND_SIZE,MAXIMUM_MANA_VALUE),
                                total=len(list(possible_initial_hands(INITIAL_HAND_SIZE,MAXIMUM_MANA_VALUE)))):
            draw = []
            for index, repetitions in enumerate(initial_hand):
                for _ in range(repetitions):
                    draw.append(index)
            self.children.append(DrawTree(draw))
            add_draws_to_node(self.children[-1], FINAL_TURN)

    # TODO: Deprecate this method
    def set_sequence_tree(self, sequence_tree: SequenceList):
        ### Associate each node to a sequence
        number_of_leaf = 0
        number_of_positive_leaf = 0
        total_impact = 0
        print("Associating draws and sequences...")
        for leaf_impact in tqdm(joint_draws_sequences(self, [], sequence_tree, -1)):
            number_of_leaf +=1
            total_impact += leaf_impact
            if leaf_impact > 0:
                number_of_positive_leaf += 1
        print(number_of_positive_leaf,"/",number_of_leaf," positive/total leaf. Avg impact:", total_impact/number_of_leaf)


# # Draws and Sequences    
# def joint_draws_sequences(draw_tree: DrawTree, drawing_sequence : List[List[int]], sequence_tree: SequenceTree, current_turn = -1):
#     for child in draw_tree.children:
#         yield from joint_draws_sequences(child, drawing_sequence + [draw_tree.draw], sequence_tree, current_turn + 1)
#         if child.best_child_impact > draw_tree.best_child_impact:
#             draw_tree.best_child_impact = child.best_child_impact
    
#     if len(draw_tree.children) == 0:
#         try:
#             fist_valid = next(get_valid_sequences(drawing_sequence + [draw_tree.draw], [], sequence_tree))
#             #get the first sequence that matches that draw
#             draw_tree.impact = fist_valid.impact
#             draw_tree.best_child_impact = fist_valid.impact
#         except StopIteration:
#             draw_tree.impact = 0
#         yield draw_tree.impact

# def get_valid_sequences(drawing_sequence : List[List[int]], remaining_cards: List[int], sequence_tree : SequenceTree):
#     if len(drawing_sequence) == 0:
#         # we are out of turns
#         available_cards = []
#     else:
#         available_cards = remaining_cards + drawing_sequence[0]
#     valid = True
#     for card in sequence_tree.turn:
#         try:
#             available_cards.remove(card)
#         except ValueError:
#             valid = False
            
#     if len(drawing_sequence) > 1:
#         next_drawing_sequence = drawing_sequence[1:]
#     else:
#         next_drawing_sequence = []
    
#     if valid:
#         if len(sequence_tree.children) == 0:
#             # we have a valid sequence
#             yield sequence_tree
#         for sequence_node in sequence_tree.children:
#             # drawing sequence is not empty
#             yield from get_valid_sequences(next_drawing_sequence, available_cards, sequence_node)

###INITIAL HANDS
def possible_initial_hands(hand_size: int, maximum_mana_value: int):
    yield from possible_initial_hands_aux(hand_size, maximum_mana_value, 0, [])

def possible_initial_hands_aux(hand_size: int, maximum_mana_value: int, mana_value: int, current_hand: List[int]):
    if mana_value == maximum_mana_value:
        yield current_hand + [hand_size]
    else:
        for i in range(hand_size+1):
            yield from possible_initial_hands_aux(hand_size-i, maximum_mana_value, mana_value+1, [i]+current_hand)
# TODO: Deprecate this method
def add_draws_to_node(node : DrawTree, turns: int):
    if turns == 1:
        # skipping the first turn
        pass
    else:
        for i in range(MAXIMUM_MANA_VALUE+1):
            node.children.append(DrawTree([i]))
            add_draws_to_node(node.children[-1], turns-1)