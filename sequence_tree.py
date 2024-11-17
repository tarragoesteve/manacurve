from typing import List
from tqdm import tqdm
import copy
import csv

class SequenceTree():
    def __init__(self, turn = [], sequence = [], impact = 0.0, children = []) -> None:
        self.turn = turn
        self.sequence = sequence
        self.impact = impact
        self.children : List[SequenceTree] = children

    def load_from_file(self, filename = 'sequences.csv'):
        print("Generating tree")
        with open(filename, newline='') as csvfile:
            number_of_nodes = 1
            reader = csv.reader(csvfile, delimiter=";")  
            sequence_tree = SequenceTree()
            for row in tqdm(reader):
                current_node = sequence_tree
                sequence = []
                sequence_imp = float(row[0])
                for turn_string in row[1:]:
                    remove_brackets = turn_string[1:-1]
                    try:
                        turn = [int(i) for i in remove_brackets.split(",")]
                    except:
                        turn = []
                    sequence.append(turn)
                    if turn not in [child.turn for child in current_node.children]:
                        number_of_nodes +=1
                        current_node.children.append(SequenceTree(copy.deepcopy(turn), copy.deepcopy(sequence), sequence_imp, []))
                    current_node.impact = max(sequence_imp, current_node.impact)
                    current_node = current_node.children[[child.turn for child in current_node.children].index(turn)]
            print("Number of nodes in sequence tree", number_of_nodes)