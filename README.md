# manacurve

A tool for computing the optimal mana curve for a Magic: The Gathering deck. Given a deck size and a target number of turns, it finds the card-count distribution across mana values (0–N) that maximises the expected "impact" of the cards played over those turns.

---

## Summary

The optimizer works in three stages:

1. **Sequence generation** — enumerate every distinct play pattern (which cards, by mana value, are cast each turn) up to `FINAL_TURN`, pruning zero-impact lines. Results are saved to `sequences.csv`.
2. **Tree construction** — load the sequences and build two trees:
   - A **SequenceTree**: a prefix tree of all valid play sequences.
   - A **DrawTree**: a prefix tree of all possible card-draw patterns (initial hand + one draw per turn). Each leaf is matched to the best achievable sequence given those draws.
3. **Optimisation** — iterate over deck compositions (counts of each mana-value slot) and score each one via the hypergeometric probability of drawing the required cards, with optional mulligan simulation. The strategy (hill climbing, multi-start hill climbing, full exploration, or single evaluation) is configurable.

---

## Requirements

- Python 3.12
```
pip install -r requirements.txt
```

---

## How to run

```bash
python main.py
```

The script will:
1. Generate and save all play sequences to `sequences.csv` (skipped if the file already exists — you will be prompted to confirm an overwrite).
2. Load the sequence tree and build the draw tree.
3. Run the configured optimisation strategy, writing candidate curves to `curves.csv` and a full probability breakdown to `results.csv`.

---

## Configuration (`config.py`)

| Parameter | Description |
|---|---|
| `INITIAL_HAND_SIZE` | Cards in the opening hand (default 7) |
| `MAXIMUM_MANA_VALUE` | Highest mana value considered (default 4) |
| `FINAL_TURN` | Number of turns simulated (default 4) |
| `DECK_SIZE` | Total cards in the deck (default 60) |
| `STRATEGY` | Optimisation strategy (see below) |
| `INITIAL_COMBINATION` | Starting deck composition for hill climbing |
| `MULLIGAN_THRESHOLD` | Fraction of probability mass mulliganed away (default 0.15) |
| `MULTI_HILL_CLIMBING_ITERATIONS` | Number of random restarts for `MULTI_HILL_CLIMBING` |
| `DECK_MINIMUMS` | Minimum count constraint per mana-value slot |

### Strategies

| Strategy | Description |
|---|---|
| `HILL_CLIMBING` | Greedy local search from `INITIAL_COMBINATION` |
| `MULTI_HILL_CLIMBING` | Multiple random-restart hill climbs |
| `FULL_EXPLORATION` | Exhaustive search over all deck compositions |
| `SINGLE` | Score a single deck composition and exit |
| `NOTHING` | Skip optimisation (useful for debugging tree construction) |

---

## Code architecture

```
manacurve/
├── config.py              # Global constants and optimisation settings
├── impact.py              # Scoring: per-card and per-sequence impact values
├── explore_sequences.py   # Enumerate all valid play sequences → sequences.csv
├── sequence_tree.py       # Load sequences.csv into a prefix tree (SequenceTree)
├── draw_tree.py           # Build draw-pattern tree; match draws to sequences
├── deck_probability.py    # Hypergeometric deck scoring + mulligan simulation
├── optimizer.py           # Optimisation strategies (hill climbing, full search)
├── main.py                # Entry point: orchestrates all stages
├── probability_bu.py      # Legacy/backup probability module (not used by main)
└── generate_sequences_bu.py  # Legacy/backup sequence generator (not used by main)
```

### Module responsibilities

**`config.py`**  
Single source of truth for all tunable parameters. Edit this file to change deck size, simulation depth, or optimisation strategy before running.

**`impact.py` → `Impact`**  
Defines the scoring function. `card_impact(mana_value)` returns a float combining the mana value and a factor derived from the probability of the card being played on curve. `sequence_impact` sums across all turns of a play line.

**`explore_sequences.py` → `ExploreSequences`**  
Recursively generates every possible sequence of turns (with and without a land drop each turn, and every valid combination of spells given available mana). Sequences with zero total impact are discarded. Unique sequences (ignoring land placement) are written to `sequences.csv` with their impact score.

**`sequence_tree.py` → `SequenceTree`**  
Reads `sequences.csv` and assembles a prefix tree where each node represents one turn's play action. The tree is used for fast matching during draw-tree population.

**`draw_tree.py` → `DrawTree`**  
Builds a tree of all possible draw patterns (hand compositions at each turn, grouped by card counts per mana value). Each leaf is assigned the impact of the best sequence achievable with that sequence of draws, by walking the `SequenceTree`.

**`deck_probability.py` → `DeckProbability`**  
Given a deck composition `[c0, c1, c2, …]` (counts per mana value), computes the probability of each draw pattern via the multivariate hypergeometric distribution. Aggregates expected impact across all draw patterns and applies mulligan logic (`score_with_mulligan`): hands are sorted by total expected value and the worst hands are "mulliganed" until the retained probability mass reaches `1 - MULLIGAN_THRESHOLD`.

**`optimizer.py` → `Optimizer`**  
Implements the optimisation loop. `hill_climbing` repeatedly moves one card from one mana-value slot to another as long as the score improves. `run` dispatches to the selected strategy and writes results to CSV files.

**`main.py`**  
Wires all stages together in order: generate sequences → build sequence tree → build draw tree → run optimiser.
