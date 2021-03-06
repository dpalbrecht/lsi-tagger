from gensim import corpora, models
import numpy as np
from collections import Counter
import itertools
import pickle
import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
from .text_cleaner import TextCleaner


class TagExtractor:
    """
    The TagExtractor trains an LSI model and uses it to extract tags and rank candidates.
    
    Args:
        word_count_min (int): The minimum number of times a word has to appear to be kept. Optional, default is 2.
        word_length_min (int): The minimum length a word must be to be kept. Optional, default is 2.
        num_lsi_topics (int): The number of LSI topics to extract during model training. 
            Optional, default is 300. Optimal values are known to be 100-400 but I've had great results up to 1,000.
        bigram_kwargs (dict): Bigram arguments. 
            bigrams_pmi_min_value: The minimum value of Pointwise Mutual Information for a bigram to be kept. 
                Values above 1 are best. Optional, default is 1.
            bigrams_min_freq: The minimum number of times a bigram has to appear to be kept. 
                Optional, default is 20.
            Optional, default is {'bigrams_pmi_min_value':1,'bigrams_min_freq':20}.
    """
    
    def __init__(self, 
                 word_count_min=2, 
                 word_length_min=2, 
                 num_lsi_topics=300,
                 bigram_kwargs={}):
        self.word_count_min = word_count_min
        self.word_length_min = word_length_min
        self.num_lsi_topics = num_lsi_topics
        self.bigram_kwargs = bigram_kwargs
        
    def save(self, fname=None):
        """
        Saves current model artifacts in the pickle format.
        
        Args:
            fname (str): File name to save to. Optional, default 'TagExtractor.p'.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        with open('TagExtractor.p' if fname is None else fname, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, fname=None):
        """
        Loads model artifacts from the pickle format.
        
        Args:
            fname (str): File name to load. Optional, default 'TagExtractor.p'.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        self.__dict__.update(
            pickle.loads(
                open('TagExtractor.p' if fname is None else fname, 'rb').read()
            )
        )
    
    def fit(self, documents):
        """
        Fits model to a set of documents.
        
        Args:
            documents (list of strings): Documents to model.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        # Clean text
        self.tc = TextCleaner(word_count_min=self.word_count_min, 
                              word_length_min=self.word_length_min,
                              bigram_kwargs=self.bigram_kwargs)
        cleaned_documents = self.tc.fit_transform(documents)
            
        # Create document lookup
        self.problem_docs = []
        self.doc2ind = {}
        for n, (doc, cleaned_doc) in enumerate(zip(documents, cleaned_documents)):
            if len(cleaned_doc)==0:
                self.problem_docs.append(doc)
            self.doc2ind[doc] = n
                
        # Warn for empty documents
        if len(self.problem_docs) > 0:
            logger.warning("Warning: Some documents yield no clean tokens. These documents won't have tags and are stored in self.problem_docs.")
            logger.warning(f'Examples: {self.problem_docs[:5]}')
        
        # Train TF-IDF
        logger.info('Training TF-IDF...')
        self.dictionary = corpora.Dictionary(cleaned_documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in cleaned_documents]
        self.tfidf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = self.tfidf[self.corpus]

        # Train LSI
        logger.info('Training LSI...')
        self.lsi_model = models.LsiModel(self.corpus_tfidf, 
                                         id2word=self.dictionary, 
                                         num_topics=self.num_lsi_topics)
        self.corpus_lsi = self.lsi_model[self.corpus_tfidf]
        
        # Save the topic matrix for tag extraction
        self.lsi_topic_matrix = self.lsi_model.get_topics()
        self.lsi_topic_matrix_T_normed = (
            self.lsi_topic_matrix.T/np.linalg.norm(self.lsi_topic_matrix.T, axis=1)[:,None]
        ).T
        
        logger.info('Tag Extractor training is done!')
        
    def _get_vector_representations(self, document):
        """
        Helper method to convert a document into its TF-IDF and LSI vector counterparts.
        
        Args:
            document (str): Document to convert to vectors.
            
        Raises:
            None.
            
        Returns:
            corpus_tfidf (list of tuples): Tuples containing word IDs and TF-IDF values.
            corpus_lsi (list of tuples): Tuples containing word IDs and LSI values.
        """
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
    
    def _lsi_corpus2vec(self, lsi_corpus_vec):
        """
        Helper method to convert a Gensim LSI representation to a numpy representation.
        
        Args:
            lsi_corpus_vec (list of tuples): Gensim LSI vector.
            
        Raises:
            None
        
        Returns:
            vec (array): Numpy LSI vector.
        """
        vec = np.zeros(self.num_lsi_topics)
        for ind, value in lsi_corpus_vec:
            vec[ind] = value
        return vec
    
    def _extract_adjacent_tags(self, input_tags, n_adjacent_tags):
        """
        Helper method to extract adjacent tags.
        
        Args:
            input_tags (list of strings): Input, reference tags.
            
        Raises:
            None
            
        Returns:
            adjacent_tags (list of strings): Adjacent tags.
        """
        bow = self.dictionary.doc2bow(input_tags)
        bow_vec = self._lsi_corpus2vec(self.lsi_model[self.tfidf[bow]])
        bow_vec_normed = bow_vec/np.linalg.norm(bow_vec)
        sims = bow_vec_normed.dot(self.lsi_topic_matrix_T_normed)
        top_inds = np.argsort(sims)[::-1]

        adjacent_tags = []
        count = 0
        for t in top_inds:
            tag = self.dictionary[t]
            flag = True
            for input_tag in input_tags:
                if (input_tag in tag) or (tag in input_tag):
                    flag = False
            if flag:
                count += 1
                adjacent_tags.append(tag)
            if count == n_adjacent_tags:
                break
        return adjacent_tags
    
    def transform(self, input_document, candidate_documents,
                  n_input_tags=10, n_candidate_tags=5, n_adjacent_tags=10):
        """
        Transforms pairs of input and candidate documents into their respective tag representations.
        
        Args:
            input_document (str): Document of input item.
            candidate_documents (list of strings): Documents of candidate items.
            n_input_tags (int): Maximum number of most common tags to aggregate associated with the input item.
            n_candidate_tags (int): Maximum of tags to extract from each candidate item.
            n_adjacent_tags (int): Maximum number of adjacent tags to extract, given the input_tags.
            
        Raises:
            None.
            
        Returns:
            input_tags (ordered list of tuples): Tuples of tags and number of times extracted from candidate_documents.
            candidate_tags (ordered list of tuples): Tuples of tags and LSI values.
            adjacent_tags (ordered list of strings): Adjacent tags to input_tags.
        """
        tfidf_input, lsi_input = self._get_vector_representations(input_document)
        if (len(tfidf_input) == 0) | (len(lsi_input) == 0):
            return [], [], []

        candidate_tags = []
        for candidate_document in candidate_documents:
            tfidf_candidate, lsi_candidate = self._get_vector_representations(candidate_document)

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
        adjacent_tags = []
        if n_adjacent_tags > 0:
            adjacent_tags = self._extract_adjacent_tags([t[0] for t in input_tags], n_adjacent_tags)

        return input_tags, candidate_tags, adjacent_tags
        
    def rank(self, candidate_tags, selected_tags):
        """
        Ranks an ordered list of candidate tags given selected tags.
        
        Args:
            candidate_tags (list of dictionaries): candidate_tags from the transform() method.
                Ordered dictionaries containing candidate tags and associated LSI values.
            selected_tags (list of strings): Tags chosen for re-ranking candidates.
            
        Raises:
            None.
            
        Returns:
            ranking (list of ints): List of indices in which to re-rank candidates.
        """
        inds, scores = [], []
        original_ranking = []
        for n, tag_dict in enumerate(candidate_tags):
            document_score = sum(tag_dict.get(tag,0) for tag in selected_tags)
            if document_score > 0:
                scores.append(document_score)
                inds.append(n)
            else:
                original_ranking.append(n)
        reranking = np.argsort(np.array(scores))[::-1].tolist()
        new_partial_ranking = np.array(inds)[reranking].tolist()
        new_ranking = new_partial_ranking + original_ranking
        return new_ranking