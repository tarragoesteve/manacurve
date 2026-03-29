from typing import List, Dict, Set
from impact import Impact
import csv
import copy
import heapq
from tqdm import tqdm
import os.path
from config import INITIAL_HAND_SIZE, MAXIMUM_MANA_VALUE, FINAL_TURN, MAXIMUM_NUMBER_OF_SEQUENCES, SEQUENCES_DIR



class ExploreSequences:
    def __init__(self,
                 turn: int = FINAL_TURN,
                 interactive = False) -> None:
        self.turn = turn


    def possible_turn(self, turn: List[int], available_mana: int, remaining_cards: int):
        """Generates all possible turns (descending MV order for better B&B pruning).

        Args:
            turn (list[int]): The cards played so far in the turn
            available_mana (int): Amount of mana remaining
            remaining_cards (int): Amount of cards remaining

        Yields:
            _type_: _description_
        """
        yield turn
        if remaining_cards > 0 and available_mana > 0:
            # Descending order: high-MV cards first so high-impact sequences are
            # found early, raising the B&B threshold faster.
            upper = min(available_mana, MAXIMUM_MANA_VALUE)
            if len(turn) > 0 and turn[-1] != 0:
                upper = min(upper, turn[-1])  # non-increasing for non-land cards
            for i in range(upper, 0, -1):
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

    @staticmethod
    def _prefer_canonical(new_seq: List[List[int]], existing_seq: List[List[int]]) -> List[List[int]]:
        """Returns the preferred canonical form: fewest total cards, then lands in later turns."""
        if len(new_seq) < len(existing_seq):
            return new_seq
        elif len(new_seq) > len(existing_seq):
            return existing_seq
        for turn_a, turn_b in zip(new_seq, existing_seq):
            if len(turn_a) < len(turn_b):
                return new_seq
            elif len(turn_a) > len(turn_b):
                return existing_seq
        return existing_seq

    @staticmethod
    def _precompute_ub_table(max_turn: int) -> list:
        """Precomputes ub_table[turn][landrops][remaining_cards] = upper bound on
        total impact achievable from turns [turn, max_turn), given current state.

        Per-turn upper bound combines two independent constraints:
          - mana constraint: future_mana * card_impact(1)        (best impact per mana)
          - card constraint: future_remaining * card_impact(MV)  (best impact per card)
        Taking min of both gives the tightest per-turn bound; summing over future turns
        yields the overall upper bound.
        """
        impact_per_mana = Impact.card_impact(1)                   # 2.05 — best impact/mana
        impact_per_card = Impact.card_impact(MAXIMUM_MANA_VALUE)  # 7.85 — best impact/card
        max_remaining = INITIAL_HAND_SIZE + max_turn
        # Shape: (max_turn+1) x (max_turn+1) x (max_remaining+1)
        ub = [[[0.0] * (max_remaining + 1)
               for _ in range(max_turn + 1)]
              for _ in range(max_turn + 1)]
        for turn in range(max_turn + 1):
            for landrops in range(max_turn + 1):
                for remaining in range(max_remaining + 1):
                    total = 0.0
                    for t_prime in range(turn, max_turn):
                        # Mana upper bound: assume a land on every remaining turn
                        future_mana = min(landrops + (t_prime - turn + 1), MAXIMUM_MANA_VALUE)
                        # Card upper bound: assume drawing 1/turn, playing nothing
                        future_remaining = min(remaining + (t_prime - turn), max_remaining)
                        total += min(future_mana * impact_per_mana,
                                     future_remaining * impact_per_card)
                    ub[turn][landrops][remaining] = total
        return ub
    
    def _explore_multi_aux(self, turn: int, landrops: int,
                           current_sequence: List[List[int]], remaining_cards: int,
                           turns_set: Set[int], max_turn: int,
                           accumulated_impact: float,
                           ub_table: list,
                           thresholds: Dict[int, float]):
        """Branch-and-bound DFS: prunes branches whose upper bound on total impact
        cannot beat the current Nth-best score at any reachable depth.

        thresholds is a shared mutable dict updated by save_for_turns as the
        bounded heap fills; mutations are visible to the generator on each resume.
        """
        # Clamp indices to ub_table dimensions
        ub_landrops = min(landrops, max_turn)
        ub_remaining = min(remaining_cards, len(ub_table[0][0]) - 1)
        ub = accumulated_impact + ub_table[turn][ub_landrops][ub_remaining]
        # Prune if the best reachable score can't improve any relevant depth's heap
        min_threshold = min((thresholds[d] for d in turns_set if d >= turn), default=0.0)
        if ub <= min_threshold:
            return

        if turn in turns_set:
            yield turn, copy.deepcopy(current_sequence)
        if turn == max_turn:
            return
        # case we do not play land
        for turn_sequence in self.possible_turn([], landrops, remaining_cards):
            ti = Impact.turn_impact(turn_sequence)
            yield from self._explore_multi_aux(
                turn + 1, landrops,
                current_sequence + [turn_sequence],
                remaining_cards + 1 - len(turn_sequence),
                turns_set, max_turn,
                accumulated_impact + ti,
                ub_table,
                thresholds)
        # case we play a land
        for turn_sequence in self.possible_turn([0], landrops + 1, remaining_cards - 1):
            ti = Impact.turn_impact(turn_sequence)
            yield from self._explore_multi_aux(
                turn + 1, landrops + 1,
                current_sequence + [turn_sequence],
                remaining_cards + 1 - len(turn_sequence),
                turns_set, max_turn,
                accumulated_impact + ti,
                ub_table,
                thresholds)

    def save_for_turns(self, turns: List[int]):
        """Single-pass branch-and-bound exploration that saves sequences_turn_{N}.csv
        for each turn in turns. Skips turns whose file already exists."""
        turns_needed = [t for t in turns if not os.path.isfile(f'{SEQUENCES_DIR}/sequences_turn_{t}.csv')]
        if not turns_needed:
            print("All sequence files already exist, skipping exploration.")
            return
        max_turn = max(turns_needed)
        turns_set = set(turns_needed)

        # Precompute upper bound table once
        ub_table = self._precompute_ub_table(max_turn)

        # Per-depth bounded dedup: key -> (canonical_seq, score)
        dedup: Dict[int, Dict[str, tuple]] = {t: {} for t in turns_needed}
        # Per-depth min-heap: [(score, key), ...] bounded to MAXIMUM_NUMBER_OF_SEQUENCES
        heap: Dict[int, list] = {t: [] for t in turns_needed}
        # Current Nth-best score per depth; shared with generator for live pruning
        thresholds: Dict[int, float] = {t: 0.0 for t in turns_needed}

        print(f"Exploring sequences for turns {sorted(turns_needed)} (branch-and-bound to turn {max_turn})...")
        for depth, sequence in tqdm(self._explore_multi_aux(
                0, 0, [], INITIAL_HAND_SIZE, turns_set, max_turn,
                0.0, ub_table, thresholds)):
            score = Impact.sequence_impact(sequence)
            if score <= 0:
                continue
            key = str(self.sequence_without_lands(sequence))
            d_dedup = dedup[depth]
            d_heap = heap[depth]

            if key in d_dedup:
                # Score is unchanged for the same non-land key; only update canonical form
                existing_seq, existing_score = d_dedup[key]
                canonical = self._prefer_canonical(sequence, existing_seq)
                if canonical is not existing_seq:
                    d_dedup[key] = (canonical, existing_score)
            else:
                if len(d_heap) < MAXIMUM_NUMBER_OF_SEQUENCES:
                    heapq.heappush(d_heap, (score, key))
                    d_dedup[key] = (sequence, score)
                elif score > d_heap[0][0]:
                    evicted_score, evicted_key = heapq.heapreplace(d_heap, (score, key))
                    del d_dedup[evicted_key]
                    d_dedup[key] = (sequence, score)
                    thresholds[depth] = d_heap[0][0]
                # else: score <= Nth-best threshold, discard

        for t in turns_needed:
            combos = dedup[t]
            print(f"Turn {t}: {len(combos)} non-redundant sequences (capped at {MAXIMUM_NUMBER_OF_SEQUENCES})")
            sequence_score = sorted(((data[1], data[0]) for data in combos.values()), reverse=True)
            filename = f'{SEQUENCES_DIR}/sequences_turn_{t}.csv'
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                for score, seq in sequence_score:
                    writer.writerow([score] + seq)
            print(f"Saved {filename}")