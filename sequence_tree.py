from typing import List
from tqdm import tqdm
import copy
import csv

class SequenceTree():
    def __init__(self, turn = None, sequence = None, impact = 0.0, children = None) -> None:
        self.turn = turn
        if turn is None:
            self.turn = []
        self.sequence = sequence
        if sequence is None:
            self.sequence = []
        self.impact = impact
        self.children : List[SequenceTree] = children
        if children is None:
            self.children = []

    def load_from_file(self, filename = 'sequences.csv'):
        print("Loading sequence tree from " + filename)
        with open(filename, newline='') as csvfile:
            number_of_nodes = 1
            reader = csv.reader(csvfile, delimiter=";")  
            for row in tqdm(reader):
                current_node = self
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
                        current_node.children.append(SequenceTree(turn, sequence, sequence_imp))
                    current_node.impact = max(sequence_imp, current_node.impact)
                    current_node = current_node.children[[child.turn for child in current_node.children].index(turn)]
            print("Number of nodes in sequence tree", number_of_nodes)