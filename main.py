print("Hello")
from explore_sequences import ExploreSequences
from sequence_tree import SequenceTree
from draw_tree import DrawTree

es = ExploreSequences()
es.save_to_file()

st = SequenceTree()
st.load_from_file()

dt = DrawTree()
dt.populate_tree()
dt.set_sequence_tree(st)

