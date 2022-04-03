from gensim import corpora, models
import numpy as np
from collections import Counter
import itertools
import pickle
from .text_cleaner import TextCleaner


class TagExtractor:
    def __init__(self, 
                 word_count_min=2, 
                 word_length_min=2, 
                 num_lsi_topics=300):
        self.word_count_min = word_count_min
        self.word_length_min = word_length_min
        self.num_lsi_topics = num_lsi_topics
        
    def save(self, fname=None):
        with open('TagExtractor.p' if fname is None else fname, 'wb') as f:
            self.__dict__['tc_word_counts'] = dict(self.__dict__['tc'].word_counts)
            pickle.dump({k:v for k,v in self.__dict__.items() if k!='tc'}, f)
    
    def load(self, fname=None):
        self.__dict__.update(
            pickle.loads(
                open('TagExtractor.p' if fname is None else fname, 'rb').read()
            )
        )
        tc = TextCleaner(
            word_count_min = self.__dict__['word_count_min'],
            word_length_min = self.__dict__['word_length_min']
        )
        tc.word_counts = self.__dict__['tc_word_counts']
        self.__dict__['tc'] = tc
    
    def fit(self, documents):
        # Clean text
        self.tc = TextCleaner(word_count_min=self.word_count_min, 
                              word_length_min=self.word_length_min)
        cleaned_documents = self.tc.fit_transform(documents)
            
        # Create document lookup
        self.problem_docs = []
        self.doc2ind = {}
        for n, (doc, cleaned_doc) in enumerate(zip(documents, cleaned_documents)):
            if len(cleaned_doc)==0:
                import pdb; pdb.set_trace()
                self.problem_docs.append(doc)
            self.doc2ind[doc] = n
                
        # Warn for empty documents
        if len(self.problem_docs) > 0:
            print("""Warning: Some documents yield no clean tokens. These documents won't have tags. Check self.problem_docs for more detail.""")
        
        # Train TF-IDF
        self.dictionary = corpora.Dictionary(cleaned_documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in cleaned_documents]
        self.tfidf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = self.tfidf[self.corpus]

        # Train LSI
        self.lsi_model = models.LsiModel(self.corpus_tfidf, 
                                         id2word=self.dictionary, 
                                         num_topics=self.num_lsi_topics)
        self.corpus_lsi = self.lsi_model[self.corpus_tfidf]
        
        # Save the topic matrix for tag extraction
        self.lsi_topic_matrix = self.lsi_model.get_topics()
        
    def _transform(self, document):
        doc_ind = self.doc2ind.get(document) 
        if doc_ind is None:
            cleaned_documents = self.tc.transform([document])
            corpus = self.dictionary.doc2bow(cleaned_documents[0])
            corpus_tfidf = self.tfidf[corpus]
            corpus_lsi = self.lsi_model[corpus_tfidf]
        else:
            corpus_tfidf = self.corpus_tfidf[doc_ind]
            corpus_lsi = self.corpus_lsi[doc_ind]
        return corpus_tfidf, corpus_lsi
    
    def extract_tags_and_rank(self, ranked_inputs, 
                              n_input_tags=10, n_candidate_tags=5, 
                              selected_tags=[]):
        
        if isinstance(ranked_inputs[0], dict):
            if len(selected_tags) > 0:
                reranking_scores = []
                for tag_dict in ranked_inputs:
                    document_score = sum(tag_dict.get(tag,0) for tag in selected_tags)
                    reranking_scores.append(document_score)
                ranking = np.argsort(-np.array(reranking_scores)).tolist()
            else:
                ranking = list(range(len(ranked_inputs)))

            return ranking
        
        if isinstance(ranked_inputs[0], str):
            input_document = ranked_inputs[0]
            candidate_documents = ranked_inputs[1:]

            tfidf_input, lsi_input = self._transform(input_document)
            if (len(tfidf_input) == 0) | (len(lsi_input) == 0):
                return [], [], []

            candidate_tags = []
            for candidate_document in candidate_documents:
                tfidf_candidate, lsi_candidate = self._transform(candidate_document)

                if (len(tfidf_candidate) == 0) | (len(lsi_candidate) == 0):
                    candidate_tags.append({})
                else:

                    # Get shared words using the sparse TF-IDF corpus
                    shared_word_ids = np.array(list(set(np.array(tfidf_input)[:,0]) \
                                                  & set(np.array(tfidf_candidate)[:,0]))).astype(int)

                    # Get word-topic vectors for those shared words
                    shared_word_topics = self.lsi_topic_matrix.T[shared_word_ids]

                    # Sum those word-topic vectors weighed by each document's topic loadings
                    shared_lsi_word_loadings = np.sum((shared_word_topics*np.array(lsi_input)[:,1]) \
                                                    + (shared_word_topics*np.array(lsi_candidate)[:,1]), axis=1)

                    # Rank shared words by highest weighed word-topic loadings
                    ranking = np.argsort(shared_lsi_word_loadings)[::-1]
                    ranked_shared_word_ids = shared_word_ids[ranking]

                    tags = [self.dictionary[r] for r in ranked_shared_word_ids][:n_candidate_tags]
                    scores = shared_lsi_word_loadings[ranking][:n_candidate_tags].tolist()

                    candidate_tags.append(dict(zip(tags, scores)))

            all_candidate_tags = list(itertools.chain(*[list(ct.keys()) for ct in candidate_tags]))
            input_tags = Counter(all_candidate_tags).most_common(n_input_tags)
            
            return input_tags, candidate_tags