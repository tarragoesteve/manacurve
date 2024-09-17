import csv

DECK_SIZE = 60
MAXIMUM_MANA_VALUE = 4
INITIAL_HAND_SIZE = 6
#FINAL_TURN = 4

class Sequence:
    def __init__(self, impact = 0, turns = []) -> None:
        self.impact = impact
        self.turns = turns

sequences = []
# Parse input
with open('sequences.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for row in reader:
        impact = float(row[0])
        turns = []
        for column in row[1:]:
            remove_brackets = column[1:-1]
            if len(remove_brackets) > 0:
                turns.append([int(i) for i in column[1:-1].split(',')])
            else:
                turns.append([])
        sequences.append(Sequence(impact, turns))

Ks = [20,10,10,10,10] # K0, K1, K2, K3, K4

def probability(sequence: Sequence, Ks : list[int]) -> float:
    ret = 1.0
    for turn in sequence.turns:
        ks = []
        for k in range(MAXIMUM_MANA_VALUE+1):            
            ks.append(sum(1 for i in turn if i == k))
        # un for per cada ks... product itertools?
        for k in ks:
            for i in range(k):
                pass
    return ret

print(sequences[0].turns[0][0])