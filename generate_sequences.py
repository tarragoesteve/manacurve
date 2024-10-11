from itertools import combinations_with_replacement, count, permutations
from scipy.stats import multivariate_hypergeom
from tqdm import tqdm
import csv
import copy
import random
from enum import Enum
from typing import List

class Strategy(Enum):
    FULL_EXPLORATION = 0
    HILL_CLIMBING = 1
    MULTI_HILL_CLIMBING = 2
    SINGLE = 3
    NOTHING = 4

MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 7
FINAL_TURNS = [4, 5]
TURN_WEIGHT = {4: 0.50,
               5: 0.50}
DECK_SIZE = 60
DECK_MINIMUMS = [0] * (MAXIMUM_MANA_VALUE+1)
#DECK_MINIMUMS[5] = 6
MAXIMUM_NUMBER_SEQUENCES = 10000
OVERWRITE_SEQUENCES = True
STRATEGY = Strategy.MULTI_HILL_CLIMBING
initial_combination = [0] * (MAXIMUM_MANA_VALUE+1)
initial_combination[0] = DECK_SIZE
# for hill climbing
#initial_combination = [22,5,10,9,4,6]
MULTI_HILL_CLIMBING_ITERATIONS = 2
MULLIGAN_THRESHOLD = .08


# Impact Methods
def sequence_impact(sequence: List[List[int]]) -> float:
    """Given a sequence computes the expected impact in a game

    Args:
        sequence (List[List[int]]): for each turn what cards have been played

    Returns:
        float: expected impact
    """
    total = 0.0
    for turn in sequence:
        total += sum([impact(i) for i in turn])
    return total

def turn_impact(turn : List[int]) ->  float:
    """Given a turn computes the impact in a game

    Args:
        turn (List[int]): cards played in a turn

    Returns:
        float: impact
    """
    return sum([impact(i) for i in turn])

def impact(mana_value: int) -> float:
    """Given a card mana value computes the impact in a game
    """
    if mana_value == 0:
        return 0
    #return mana_value + 1.5
    if mana_value == 1:
        return mana_value + 1/0.95173
    elif mana_value == 2:
        return mana_value + 1/0.8241
    elif mana_value == 3:
        return mana_value + 1/0.63784
    elif mana_value == 4:
        return mana_value + 1/0.44051
    elif mana_value == 5:
        return mana_value + 1/0.27278
    elif mana_value == 6:
        return mana_value + 1/0.15229
    elif mana_value == 7:
        return mana_value + 1/0.07698
    return mana_value+1.5
        
# Generating sequences methods
def sequence_without_lands(sequence: List[List[int]]) -> List[List[int]]:
    """Given a sequence removes lands

    Args:
        sequence (List[List[int]]): 

    Returns:
        List[List[int]]: 
    """
    ret = []
    for turn in sequence:
        ret.append([i for i in turn if i != 0])
    return ret
        
def possible_sequences(minimum_sequence_impact = 0.0, maximum_impact_in_turn = 10.0):
    """Generates all possible sequences of turns"""
    yield from possible_sequences_auxiliar(0, 0, 0.0, minimum_sequence_impact, maximum_impact_in_turn, [], INITIAL_HAND_SIZE)

def possible_sequences_auxiliar(turn: int, landrops: int, accumulated_impact: float, minimum_sequence_impact: float, maximum_impact_in_turn: float,
                                current_sequence: List[List[int]], remaining_cards: int):
    """Generates all possible sequences of turns

    Args:
        turn (int): 
        landrops (int): 
        current_sequence (List[List[int]]): 
        remaining_cards (int): 

    Yields:
        List[List[int]]: sequence of turns
    """
    if turn == max(FINAL_TURNS):
        if accumulated_impact >= minimum_sequence_impact:
            yield current_sequence
    elif (max(FINAL_TURNS) - turn) * maximum_impact_in_turn + accumulated_impact < minimum_sequence_impact:
        # we cannot reach the minimum impact
        pass
    else:
        # case we do not play land
        for turn_sequence in possible_turn([], landrops, remaining_cards):
            yield from possible_sequences_auxiliar(turn + 1, landrops,
                                                   accumulated_impact + turn_impact(turn_sequence), minimum_sequence_impact, maximum_impact_in_turn,
                                                   copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
        # case we play a land
        for turn_sequence in possible_turn([0], landrops + 1, remaining_cards-1):
            yield from possible_sequences_auxiliar(turn + 1, landrops + 1,
                                                   accumulated_impact + turn_impact(turn_sequence), minimum_sequence_impact, maximum_impact_in_turn,
                                                   copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))


def possible_turn(turn : List[int], available_mana: int, remaining_cards: int):
    """Generates all possible turns

    Args:
        turn (list[int]): The cards played so far in the turn
        available_mana (int): Amount of mana remaining
        remaining_cards (int): Amount of cards remaining

    Yields:
        _type_: _description_
    """
    yield turn
    if remaining_cards > 0 and available_mana > 0:
        #only ascending order
        last_played = 1
        if len(turn) > 0:
            last_played = turn[-1]
            last_played = max(1, last_played) # we cannot repeat landrops
        for i in range(last_played, min(available_mana+1, MAXIMUM_MANA_VALUE+1)):
            yield from possible_turn(copy.deepcopy(turn + [i]), available_mana - i, remaining_cards - 1)

# Probability computation auxiliar method
def non_valid_possible_combinations(Ks: List[int], Cs: List[int], free_cards: int):
    """Returns all the possible combinations that are not valid

    Args:
        Ks (List[int]): Minim number of each k to be a valid restriction
        Cs (List[int]): Maximum number of k that are available
        free_cards (int): Number of cards that can be played

    Yields:
        List[int]: combination 
    """
    yield from non_valid_possible_combinations_aux(0, [], True, Ks, Cs, free_cards)
    
    
def non_valid_possible_combinations_aux(index: int, combination: List[int],
                                        is_valid: bool,
                                        Ks: List[int], Cs: List[int], free_cards: int):
    if index >= len(Ks):
        if not is_valid:
            yield combination
    else:
        up_to = min(Cs[index], free_cards)
        for i in range(up_to+1):
            yield from non_valid_possible_combinations_aux(index+1,
                                          combination + [i],
                                          is_valid and i >= Ks[index],
                                          Ks,
                                          Cs,
                                          free_cards - i)
            
def possible_initial_hands(hand_size: int, maximum_mana_value: int):
    yield from possible_initial_hands_aux(hand_size, maximum_mana_value, 0, [])

def possible_initial_hands_aux(hand_size: int, maximum_mana_value: int, mana_value: int, current_hand: List[int]):
    if mana_value == maximum_mana_value:
        yield current_hand + [hand_size]
    else:
        for i in range(hand_size+1):
            yield from possible_initial_hands_aux(hand_size-i, maximum_mana_value, mana_value+1, [i]+current_hand)



class Node():
    def __init__(self, turn = [], sequence = [], impact = 0.0, children = []) -> None:
        self.turn = turn
        self.sequence = sequence
        self.impact = impact
        self.children : List[Node] = children

# Score computation method
def score_auxiliar(tree_sequence: Node, Cs: List[int], free_cards: int, node_probability, minimum_probability = 0.0):
    turn_conditioned_probability = 1.0
    ks = [sum(1 for j in tree_sequence.turn if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
    ks_index = [i for i in range(MAXIMUM_MANA_VALUE+1) if ks[i] > 0]
    ks_values = [ks[i] for i in ks_index]
    if any([k > c for k, c in zip(ks, Cs)]) or sum(ks) > free_cards:
        turn_conditioned_probability = 0
    else:
        for combination in non_valid_possible_combinations(ks_values, [Cs[i] for i in ks_index], free_cards):
            k_other = free_cards - sum(combination)
            K_other = sum(Cs) - sum(Cs[i] for i in ks_index)
            if K_other < k_other:
                # This combination is not possible
                pass
            else:
                combination_prob = multivariate_hypergeom.pmf(x=[i for i in combination]+[k_other], m=[Cs[i] for i in ks_index]+[K_other], n=free_cards)
                turn_conditioned_probability -= combination_prob
    node_probability *= turn_conditioned_probability
    if node_probability > minimum_probability:
        # continue exploring childs only if they have a significant probability
        #print("Turn", tree_sequence.turn, "Probability", node_probability, "Impact", tree_sequence.impact)
        for child in tree_sequence.children:
            yield from score_auxiliar(child, [c - k for k, c in zip(ks, Cs)], free_cards - sum(ks) + 1, node_probability, minimum_probability)
        if len(tree_sequence.children) == 0:
            yield node_probability, tree_sequence.impact, tree_sequence.sequence                  


def score(tree_sequence: Node, Cs: List[int], minimum_probability = 0.0) -> float:
    return sum([probability * impact for probability, impact, _ in score_auxiliar(tree_sequence, Cs, INITIAL_HAND_SIZE-1, 1.0, minimum_probability)])



if OVERWRITE_SEQUENCES:
    saved_combinations = {}
    possible = 0
    print("Exploring all possible sequences")
    for sequence in tqdm(possible_sequences(0, 12.5)):
        possible += 1
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
                        
    print("Non-redudant", len(saved_combinations), "out of", possible)
    sequence_score = [(sequence_impact(sequence), sequence) for sequence in saved_combinations.values()]
    sequence_score.sort(key=lambda x: x[0], reverse=True)
    with open('sequences.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        for i in range(min(len(sequence_score), MAXIMUM_NUMBER_SEQUENCES)):
            writer.writerow([sequence_score[i][0]] + sequence_score[i][1])
    
print("Generating tree")
with open('sequences.csv', newline='') as csvfile:
    number_of_nodes = 1
    reader = csv.reader(csvfile, delimiter=";")  
    sequence_tree = Node()
    for row in tqdm(reader):
        current_node = sequence_tree
        sequence = []
        sequence_imp = float(row[0])
        for turn_string in row[1:]:
            if len(sequence) < max(FINAL_TURNS):
                remove_brackets = turn_string[1:-1]
                try:
                    turn = [int(i) for i in remove_brackets.split(",")]
                except:
                    turn = []
                sequence.append(turn)
                if turn not in [child.turn for child in current_node.children]:
                    number_of_nodes +=1
                    current_node.children.append(Node(copy.deepcopy(turn), copy.deepcopy(sequence), sequence_imp, []))
                current_node = current_node.children[[child.turn for child in current_node.children].index(turn)]
    print("Number of nodes in sequence tree", number_of_nodes)

class DrawNode():
    def __init__(self, draw = [], accumulated_draw = [], children = [], best_sequence = None, probability = 0.0, expected_impact = {}) -> None:
        self.draw: List[int] = draw
        self.accumulated_draw: List[int] = accumulated_draw
        self.children: List[DrawNode] = children
        if best_sequence == None:
            self.best_sequence: Node = Node([],[],0,[])
        else:
            self.best_sequence = best_sequence
        self.probability = probability
        self.expected_impact = expected_impact

def add_draws_to_node(node : DrawNode, turns: int):
    if turns == 0:
        pass
    else:
        for i in range(MAXIMUM_MANA_VALUE+1):
            aux_node = DrawNode([i],node.accumulated_draw + [i], [])
            add_draws_to_node(aux_node, turns-1)
            node.children.append(copy.deepcopy(aux_node))

def new_score(draw_tree : DrawNode, Cs: List[int], mulligan_threshold = 0.0):
    for _ in new_score_auxiliar(draw_tree, Cs, 1.0, -1):
        pass
    return score_with_mulligan(draw_tree, mulligan_threshold, TURN_WEIGHT)

def new_score_auxiliar(draw_tree : DrawNode, Cs: List[int], probability : float, current_turn: int):
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
    if len(new_Cs) != MAXIMUM_MANA_VALUE+1:
        raise NameError("Incorrect Cs length")
    if turn_probability > 0 and draw_tree.best_sequence.impact > 0:
        if current_turn in FINAL_TURNS:
            if current_turn == 3:
                pass
            draw_tree.expected_impact[current_turn] = draw_tree.best_sequence.impact
            yield turn_probability, draw_tree.best_sequence.impact, draw_tree.best_sequence.sequence
        for child in draw_tree.children:
            yield from new_score_auxiliar(child, new_Cs, turn_probability, current_turn+1)
        for turn in [turn for turn in FINAL_TURNS if turn != current_turn]:
            accumulated_impact = 0.0
            for child in draw_tree.children:
                try:
                    accumulated_impact += child.probability * child.expected_impact[turn]
                except KeyError:
                    pass
            draw_tree.expected_impact[turn] = accumulated_impact / draw_tree.probability
        if len(draw_tree.children) > 0:
            if abs(sum([child.probability for child in draw_tree.children]) - draw_tree.probability) > 1e-6:
                raise NameError("Bad probabilities")

                
def score_with_mulligan(draw_tree : DrawNode, mulligan_threshold = 0.0, turn_weight = {}):
    accumulated_probability = 0.0
    accumulated_expected_impact = 0.0
    hands = [(hand.expected_impact, hand.probability) for hand in draw_tree.children if hand.probability > 0]
    for hand in hands:
        if len(hand[0]) != len(FINAL_TURNS):
            raise NameError("Error")
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






def joint_draws_sequences(draw_tree: DrawNode, drawing_sequence : List[List[int]],sequence_tree: Node, current_turn = -1):
    for child in draw_tree.children:
        yield from joint_draws_sequences(child, drawing_sequence + [draw_tree.draw], sequence_tree, current_turn + 1)
        if child.best_sequence.impact > draw_tree.best_sequence.impact:
            draw_tree.best_sequence = child.best_sequence
    
    if current_turn in FINAL_TURNS:
        try:
            fist_valid = next(get_valid_sequences(drawing_sequence + [draw_tree.draw], [], sequence_tree))
            #get the first sequence that matches that draw
            draw_tree.best_sequence = fist_valid
        except StopIteration:
            draw_tree.best_sequence = Node([],[],0,[])
        yield draw_tree.best_sequence.impact

def get_valid_sequences(drawing_sequence : List[List[int]], remaining_cards: List[int], sequence_tree : Node):
    # check that we fullfill the root with the drawn cards
    if len(drawing_sequence) == 0:
            raise IndexError
    available_cards = copy.deepcopy(remaining_cards + drawing_sequence[0])
    valid = True
    for card in sequence_tree.turn:
        try:
            available_cards.remove(card)
        except ValueError:
            valid = False
    
    if valid:
        if len(sequence_tree.children) == 0:
            #check that the sequence tree is empty
            yield sequence_tree
        for sequence_node in sequence_tree.children:
            yield from get_valid_sequences(drawing_sequence[1:], available_cards, sequence_node)

    
def hill_climbing(initial_combination, draw_node):
    best_score = 0
    best_combination = initial_combination
    keep_exploring = True
    last_direction = (0, 1)
    while keep_exploring:
        keep_exploring = False
        directions = [(i, j) for j in range(len(best_combination)) for i in range(len(best_combination)) if i!=j and (i, j) != last_direction]
        random.shuffle(directions)
        directions = [last_direction] + directions
        for (i, j) in tqdm(directions):
            combination = copy.deepcopy(best_combination)
            combination[j] += 1
            combination[i] -= 1
            if all([k >= DECK_MINIMUMS[index] for index,k in enumerate(combination)]):
                combination_score = new_score(draw_node, combination, MULLIGAN_THRESHOLD)
                if combination_score > best_score:
                    last_direction = (i, j)
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
    ### Generate draw tree
    draw_tree = DrawNode()
    print("Generating draw tree")
    for initial_hand in tqdm(possible_initial_hands(INITIAL_HAND_SIZE,MAXIMUM_MANA_VALUE),
                            total=len(list(possible_initial_hands(INITIAL_HAND_SIZE,MAXIMUM_MANA_VALUE)))):
        draw = []
        for index, repetitions in enumerate(initial_hand):
            for _ in range(repetitions):
                draw.append(index)
        aux_node = DrawNode(draw, draw, [])
        add_draws_to_node(aux_node, max(FINAL_TURNS)) # we do not draw in the fist turn
        draw_tree.children.append(copy.deepcopy(aux_node))

    ### Associate each node to a sequence
    number_of_leaf = 0
    number_of_positive_leaf = 0
    total_impact = 0
    print("Associating draws and sequences...")
    for leaf_impact in tqdm(joint_draws_sequences(draw_tree, [], sequence_tree, -1)):
        number_of_leaf +=1
        total_impact += leaf_impact
        if leaf_impact > 0:
            number_of_positive_leaf += 1
    print(number_of_positive_leaf,"/",number_of_leaf," positive/total leaf. Avg impact:", total_impact/number_of_leaf)
    with open('curves.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        best_score = 0.0
        best_combination = []
        if STRATEGY == Strategy.FULL_EXPLORATION:
            for combination in tqdm(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE),
                                    total=len(list(combinations_with_replacement(range(MAXIMUM_MANA_VALUE+1), DECK_SIZE)))):
                Ks = [sum(1 for j in combination if j == i) for i in range(MAXIMUM_MANA_VALUE+1)]
                combination_score = score(sequence_tree, Ks)
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
        for probability, sequence_impact, sequence in new_score_auxiliar(draw_tree, best_combination, 1.0, -1):
            if not str(sequence) in results:
                results[str(sequence)] = 0.0
                recover_sequence[str(sequence)] = sequence
            results[str(sequence)] += probability
            total_probability += probability
        for sequence_str, probability in results.items():
            writer.writerow([round(probability,8)] + recover_sequence[sequence_str])
        print("Total probability", total_probability)

            
        
        
    
        


            

        