from typing import List, Dict, Set
from impact import Impact
import csv
import copy
from tqdm import tqdm
import os.path
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN, MAXIMUM_NUMBER_OF_SEQUENCES, SEQUENCES_DIR



class ExploreSequences:
    def __init__(self,
                 turn: int = FINAL_TURN,
                 interactive = False) -> None:
        self.turn = turn


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
        if turn == self.turn:
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
    
    def _explore_multi_aux(self, turn: int, landrops: int,
                           current_sequence: List[List[int]], remaining_cards: int,
                           turns_set: Set[int], max_turn: int):
        """Single-pass traversal that yields (turn_depth, sequence) at each depth in turns_set.
        Continues beyond yield points until max_turn is reached."""
        if turn in turns_set:
            yield turn, copy.deepcopy(current_sequence)
        if turn == max_turn:
            return
        # case we do not play land
        for turn_sequence in self.possible_turn([], landrops, remaining_cards):
            yield from self._explore_multi_aux(
                turn + 1, landrops,
                current_sequence + [turn_sequence],
                remaining_cards + 1 - len(turn_sequence),
                turns_set, max_turn)
        # case we play a land
        for turn_sequence in self.possible_turn([0], landrops + 1, remaining_cards - 1):
            yield from self._explore_multi_aux(
                turn + 1, landrops + 1,
                current_sequence + [turn_sequence],
                remaining_cards + 1 - len(turn_sequence),
                turns_set, max_turn)

    def save_for_turns(self, turns: List[int]):
        """Single-pass exploration that saves sequences_turn_{N}.csv for each turn in turns.
        Skips turns whose file already exists."""
        turns_needed = [t for t in turns if not os.path.isfile(f'{SEQUENCES_DIR}/sequences_turn_{t}.csv')]
        if not turns_needed:
            print("All sequence files already exist, skipping exploration.")
            return
        max_turn = max(turns_needed)
        turns_set = set(turns_needed)
        # One dedup dict per turn depth
        saved: Dict[int, dict] = {t: {} for t in turns_needed}
        print(f"Exploring sequences for turns {sorted(turns_needed)} (single pass to turn {max_turn})...")
        for depth, sequence in tqdm(self._explore_multi_aux(
                0, 0, [], INITIAL_HAND_SIZE, turns_set, max_turn)):
            if Impact.sequence_impact(sequence) <= 0:
                continue
            key = str(self.sequence_without_lands(sequence))
            existing = saved[depth].get(key)
            if existing is None:
                saved[depth][key] = sequence
            else:
                if len(sequence) < len(existing):
                    saved[depth][key] = sequence
                elif len(sequence) == len(existing):
                    for turn_a, turn_b in zip(sequence, existing):
                        if len(turn_a) < len(turn_b):
                            saved[depth][key] = sequence
                            break
                        elif len(turn_a) > len(turn_b):
                            break
        for t in turns_needed:
            combos = saved[t]
            print(f"Turn {t}: {len(combos)} non-redundant sequences")
            sequence_score = [(Impact.sequence_impact(seq), seq) for seq in combos.values()]
            sequence_score.sort(key=lambda x: x[0], reverse=True)
            filename = f'{SEQUENCES_DIR}/sequences_turn_{t}.csv'
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                for i in range(min(len(sequence_score), MAXIMUM_NUMBER_OF_SEQUENCES)):
                    writer.writerow([sequence_score[i][0]] + sequence_score[i][1])
            print(f"Saved {filename}")

    def save_to_file(self, filename = None):
        if filename is None:
            filename = f'{SEQUENCES_DIR}/sequences_turn_{self.turn}.csv'
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
     