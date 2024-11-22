from typing import List
from sequence_tree import SequenceTree
import copy
from tqdm import tqdm
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN


class DrawTree():
    def __init__(self, draw = None, children = None, impact = 0.0, best_child_impact = 0.0, probability = 0.0, expected_impact = 0.0) -> None:
        self.draw: List[int] = draw
        if draw is None:
            self.draw = []
        self.children: List[DrawTree] = children
        if children is None:
            self.children = []
        self.impact = impact
        self.best_child_impact = best_child_impact
        self.probability = probability
        self.expected_impact = expected_impact

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

    
    def set_sequence_tree(self, sequence_tree: SequenceTree):
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


# Draws and Sequences    
def joint_draws_sequences(draw_tree: DrawTree, drawing_sequence : List[List[int]], sequence_tree: SequenceTree, current_turn = -1):
    for child in draw_tree.children:
        yield from joint_draws_sequences(child, drawing_sequence + [draw_tree.draw], sequence_tree, current_turn + 1)
        if child.impact > draw_tree.best_child_impact:
            draw_tree.best_child_impact = child.impact
    
    if len(draw_tree.children) == 0:
        try:
            fist_valid = next(get_valid_sequences(drawing_sequence + [draw_tree.draw], [], sequence_tree))
            #get the first sequence that matches that draw
            draw_tree.impact = fist_valid.impact
        except StopIteration:
            draw_tree.impact = 0
        yield draw_tree.impact

def get_valid_sequences(drawing_sequence : List[List[int]], remaining_cards: List[int], sequence_tree : SequenceTree):
    if len(drawing_sequence) == 0:
        # we are out of turns
        available_cards = []
    else:
        available_cards = remaining_cards + drawing_sequence[0]
    valid = True
    for card in sequence_tree.turn:
        try:
            available_cards.remove(card)
        except ValueError:
            valid = False
            
    if len(drawing_sequence) > 1:
        next_drawing_sequence = drawing_sequence[1:]
    else:
        next_drawing_sequence = []
    
    if valid:
        if len(sequence_tree.children) == 0:
            # we have a valid sequence
            yield sequence_tree
        for sequence_node in sequence_tree.children:
            # drawing sequence is not empty
            yield from get_valid_sequences(next_drawing_sequence, available_cards, sequence_node)

###INITIAL HANDS
def possible_initial_hands(hand_size: int, maximum_mana_value: int):
    yield from possible_initial_hands_aux(hand_size, maximum_mana_value, 0, [])

def possible_initial_hands_aux(hand_size: int, maximum_mana_value: int, mana_value: int, current_hand: List[int]):
    if mana_value == maximum_mana_value:
        yield current_hand + [hand_size]
    else:
        for i in range(hand_size+1):
            yield from possible_initial_hands_aux(hand_size-i, maximum_mana_value, mana_value+1, [i]+current_hand)

def add_draws_to_node(node : DrawTree, turns: int):
    if turns == 1:
        # skipping the first turn
        pass
    else:
        for i in range(MAXIMUM_MANA_VALUE+1):
            node.children.append(DrawTree([i]))
            add_draws_to_node(node.children[-1], turns-1)