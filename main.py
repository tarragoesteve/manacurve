print("Hello")
from explore_sequences import ExploreSequences
from sequence_list import SequenceList
from draw_tree import RootTree
from optimizer import Optimizer

es = ExploreSequences()
es.save_to_file()

sl = SequenceList()
sl.load_from_file()

rt = RootTree()
rt.populate_tree(sl)

Optimizer.run(rt)

