print("Hello")
from explore_sequences import ExploreSequences
from sequence_list import SequenceList
from draw_tree import RootTree
from optimizer import Optimizer
import os
import config

os.makedirs(config.SEQUENCES_DIR, exist_ok=True)
os.makedirs(config.TREES_DIR, exist_ok=True)

print(f"\n=== Turns to optimize: {config.TURNS} (weights: {config.TURN_WEIGHTS}) ===")

# Single-pass exploration only for turns whose tree (and sequence file) doesn't exist yet
turns_needing_trees = [t for t in config.TURNS if not os.path.isfile(f'{config.TREES_DIR}/tree_turn_{t}.json')]
if turns_needing_trees:
    print(f"\n--- Step 1: Exploring sequences for turns {turns_needing_trees} ---")
    ExploreSequences().save_for_turns(turns_needing_trees)
else:
    print("\n--- Step 1: All tree files found on disk, skipping exploration ---")

# Load or build each tree (skips sequence loading if tree JSON already exists)
print("\n--- Step 2: Loading / building draw trees ---")
weighted_trees = []
for turn, weight in zip(config.TURNS, config.TURN_WEIGHTS):
    tree_path = f'{config.TREES_DIR}/tree_turn_{turn}.json'
    if os.path.isfile(tree_path):
        print(f"  Turn {turn} (weight={weight}): loading from {tree_path}")
        rt = RootTree.load_from_file(turn, tree_path)
    else:
        print(f"  Turn {turn} (weight={weight}): building from sequences")
        sl = SequenceList()
        sl.load_from_file(f"{config.SEQUENCES_DIR}/sequences_turn_{turn}.csv")
        rt = RootTree(turn=turn)
        rt.populate_tree(sl)
        rt.save_to_file(tree_path)
    weighted_trees.append((weight, rt))

print(f"\n--- Step 3: Optimizing over {len(weighted_trees)} tree(s) ---")
Optimizer.run(weighted_trees)
print("\n=== Done ===")

