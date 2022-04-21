from ..tag_extractor import TagExtractor
import numpy as np



def test_lsi_corpus2vec():
    tc = TagExtractor(num_lsi_topics=5)
    result = tc._lsi_corpus2vec([(0,10),(3,1)])
    assert (result == np.array([10,0,0,1,0])).all()


def test_rank():
    tc = TagExtractor()
    candidate_tags = [{'a':1},{'b':2},{'b':3},{'c':4}]
    selected_tags = ['b']
    new_ranking = tc.rank(candidate_tags, selected_tags)
    assert new_ranking == [2,1,0,3]