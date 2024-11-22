from typing import List
from impact import Impact
import csv
import copy
from tqdm import tqdm
import os.path
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN, MAXIMUM_NUMBER_OF_SEQUENCES



class ExploreSequences:
    def __init__(self,
                 interactive = False) -> None:
        pass


    def possible_sequences(self):
        """Generates all possible sequences of turns"""
        yield from self.possible_sequences_auxiliary(0, 0, 0.0,
                                                    [], INITIAL_HAND_SIZE)

    def possible_sequences_auxiliary(self, turn: int, landrops: int, accumulated_impact: float,
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
        if turn == FINAL_TURN:
                yield current_sequence
        else:
            # case we do not play land
            for turn_sequence in self.possible_turn([], landrops, remaining_cards):
                yield from self.possible_sequences_auxiliary(turn + 1, landrops,
                                                    accumulated_impact + Impact.turn_impact(turn_sequence),
                                                    copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))
            # case we play a land
            for turn_sequence in self.possible_turn([0], landrops + 1, remaining_cards-1):
                yield from self.possible_sequences_auxiliary(turn + 1, landrops + 1,
                                                    accumulated_impact + Impact.turn_impact(turn_sequence),
                                                    copy.deepcopy(current_sequence + [turn_sequence]), remaining_cards + 1 - len(turn_sequence))

    def possible_turn(self, turn : List[int], available_mana: int, remaining_cards: int):
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
                yield from self.possible_turn(copy.deepcopy(turn + [i]), available_mana - i, remaining_cards - 1)



    # Generating sequences methods
    def sequence_without_lands(self, sequence: List[List[int]]) -> List[List[int]]:
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
    
    def save_to_file(self, filename = 'sequences.csv'):
        if os.path.isfile(filename):
            input_text = input(filename + " already exist, are you sure you want to overwrite? (y/n)")
            if input_text != 'y':
                print("Aborting sequence generation")
                return 1
        saved_combinations = {}
        possible = 0
        print("Exploring all possible sequences")
        for sequence in tqdm(self.possible_sequences()):
            possible += 1
            if Impact.sequence_impact(sequence) > 0:
                if str(self.sequence_without_lands(sequence)) not in saved_combinations:
                    saved_combinations[str(self.sequence_without_lands(sequence))] = sequence
                else:
                    if len(sequence) < len(saved_combinations[str(self.sequence_without_lands(sequence))]):
                        # shortest combination == useless lands
                        saved_combinations[str(self.sequence_without_lands(sequence))] = sequence
                    elif len(sequence) == len(saved_combinations[str(self.sequence_without_lands(sequence))]):
                        for turn_a, turn_b in zip(sequence, saved_combinations[str(self.sequence_without_lands(sequence))]):
                            # we prefer to play lands in the latest turns
                            if len(turn_a) < len(turn_b):
                                saved_combinations[str(self.sequence_without_lands(sequence))] = sequence
                                break
                            elif len(turn_a) > len(turn_b):
                                break
                            
        print("Non-redudant", len(saved_combinations), "out of", possible)
        sequence_score = [(Impact.sequence_impact(sequence), sequence) for sequence in saved_combinations.values()]
        sequence_score.sort(key=lambda x: x[0], reverse=True)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            for i in range(min(len(sequence_score), MAXIMUM_NUMBER_OF_SEQUENCES)):
                writer.writerow([sequence_score[i][0]] + sequence_score[i][1])
     