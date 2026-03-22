from typing import List
from tqdm import tqdm
from config import MAXIMUM_NUMBER_OF_SEQUENCES
import csv

class SequenceList():
    def __init__(self) -> None:
        self.sequences : List[tuple[float, List[List[int]]]] = []
        pass

    def load_from_file(self, filename = 'sequences.csv'):
        print("Loading sequence tree from " + filename)
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=";")  
            for row in tqdm(reader):
                if len(self.sequences) >= MAXIMUM_NUMBER_OF_SEQUENCES:
                    break
                sequence = []
                impact = float(row[0])
                for turn_string in row[1:]:
                    remove_brackets = turn_string[1:-1]
                    try:
                        turn = [int(i) for i in remove_brackets.split(",")]
                    except:
                        turn = []
                    sequence.append(turn)
                self.sequences.append((impact, sequence))
        print("Number of nodes in sequence list", len(self.sequences))