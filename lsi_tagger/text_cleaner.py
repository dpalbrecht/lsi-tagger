import re
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
import itertools
from tqdm import tqdm
import nltk
nltk.download('averaged_perceptron_tagger')


def _tqdm(iterator, display, desc=None):
    """
    Helper wrapper over tqdm to control displaying progress.
    
    Args:
        iterator (list, generator): Iterator to iterate through.
        display (bool): Whether to display progress.
        desc (str): Explanation for work on the iterator. Optional, default is None.
        
    Raises:
        None.
        
    Returns:
        iterator (list, generator, tqdm object): Iterator with tqdm wrapper if display is True.
    """
    if display:
        return tqdm(iterator, position=0, 
                    leave=True, desc=desc)
    return iterator


class TextCleaner:
    def __init__(self, word_count_min=1, word_length_min=2, bigram_kwargs={}):
        self.punctuation = """!"$%\'()*+,./:;<=>?@[\\]^_`{|}~"""
        self.remove_punctuation_rule = re.compile(f"[{re.escape(self.punctuation)}]")
        self.word_counts = defaultdict(lambda :0)
        self.word_count_min = word_count_min
        self.word_length_min = word_length_min
        self.bigrams = (len(bigram_kwargs)>0)
        if self.bigrams:
            self.bigrams_pmi_min_value = bigram_kwargs.get('bigrams_pmi_min_value', 1)
            self.bigrams_min_freq = bigram_kwargs.get('bigrams_min_freq', 20)
        keep_stopwords = ['top','bottom','back','front','full','her','him','herself',
                          'himself','his','hers','kg','km','cm','thick','thin','under',
                          'you','your','yours']
        self.STOPWORDS = keep_stopwords = [s for s in STOPWORDS if s not in keep_stopwords]
        
    def _pos_filter(self, bigram):
        """
        Helper method to filter bigrams based on part of speech (POS).
        
        Args:
            bigram (tuple): bigram to filter. Example: ('word1','word2').
            
        Raises:
            None.
            
        Returns:
            (bool): True/False whether to keep bigram.
        """
        for word in bigram:
            if (word in self.STOPWORDS) or (len(word)<self.word_length_min):
                return False
        acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
        second_type = ('NN', 'NNS', 'NNP', 'NNPS')
        tags = nltk.pos_tag(bigram)
        if (tags[0][1] in acceptable_types) and (tags[1][1] in second_type):
            return True
        else:
            return False

    def _create_bigrams_dict(self, tokens):
        """
        Helper method to create map of bigrams to extract.
        
        Args:
            tokens (list of strings): Ordered list of tokens.
            
        Raises:
            None.
            
        Returns:
            bigrams_dict (dict): Dictionary of bigrams to extract. 
        """
        # Create bigrams
        bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

        # Filter bigrams by minimum frequency
        bigram_finder.apply_freq_filter(self.bigrams_min_freq)

        # Calculate PMI
        self.bigrams_dict = list(bigram_finder.score_ngrams(nltk.collocations.BigramAssocMeasures().pmi))

        # Filter bigrams based on POS and min PMI
        self.bigrams_dict = {b[0]:f'{b[0][0]} {b[0][1]}' for b in self.bigrams_dict 
                             if ((b[1]>=self.bigrams_pmi_min_value) and self._pos_filter(b[0]))}
        
    def _get_bigrams(self, tokens, fit):
        """
        Helper method to extract bigrams from tokens.
        
        Args:
            tokens (list of lists of strings): List of documents split into tokens.
            fit (bool): Whether to fit new bigrams or use existing mapping to extract bigrams.
            
        Raises:
            None.
            
        Returns:
            bigram_documents (list of lists of strings): Extracted bigrams from tokens.
        """
        if fit:
            self._create_bigrams_dict(list(itertools.chain(*tokens)))
        bigram_documents = []
        for bigrams in _tqdm((list(zip(words[:-1], words[1:])) for words in tokens), 
                             display=fit, desc='Making bigrams'):
            temp_bigrams = []
            for bigram in bigrams:
                keep_bigram = self.bigrams_dict.get(bigram)
                if keep_bigram is not None:
                    temp_bigrams.append(keep_bigram)
                    if fit:
                        self.word_counts[keep_bigram] += 1
            bigram_documents.append(temp_bigrams)
        return bigram_documents
    
    def _preprocess_text(self, documents, fit):
        """
        Helper method to preprocess text by removing punctuation, extra spaces, and lower casing.
        
        Args:
            documents (list of strings): Documents to preprocess.
            fit (bool): Whether to display progress.
            
        Raises:
            None.
            
        Returns:
            tokens (list of lists of strings): Transformed documents into tokens.
        """
        tokens = []
        for text in _tqdm(documents, display=fit, desc='Preprocessing text'):
            text = self.remove_punctuation_rule.sub(' ', text)
            text = re.sub(' +', ' ', text)
            tokens.append(text.strip().lower().split())
        return tokens

    def _filter_tokens(self, tokens, fit):
        """
        Helper method to filter tokens based on stopwords and word_length_min, and fit word_counts.
        One step downstream of self._preprocess_text.
        
        Args:
            tokens (list of lists of strings): List of documents split into tokens.
            fit (bool): Whether to fit word_counts.
            
        Raises:
            None.
        
        Returns:
            cleaned_tokens (list of lists of strings): Cleaned tokens.
        """
        cleaned_tokens = []
        for token_list in _tqdm(tokens, display=fit, desc='Filtering tokens'):
            temp_tokens = []
            for word in token_list:
                if (word not in self.STOPWORDS) & (len(word) >= self.word_length_min):
                    temp_tokens.append(word)
                    if fit:
                        self.word_counts[word] += 1
            cleaned_tokens.append(temp_tokens)
        return cleaned_tokens
    
    def _filter_word_count(self, cleaned_tokens, fit):
        """
        Helper method to filter tokens based on word_count_min.
        
        Args:
            cleaned_tokens (list of lists of strings): List of documents split into cleaned tokens.
            fit (bool): Whether to display progress.
            
        Raises:
            None.
        
        Returns:
            (list of lists of strings): Filtered tokens.
        """
        return [[word for word in ct if (self.word_counts.get(word, 0) >= self.word_count_min)] 
                for ct in _tqdm(cleaned_tokens, display=fit, desc='Filtering by word count')]

    def _clean_documents(self, documents, fit):
        """
        Helper method to clean documents.
        
        Args:
            documents (list of strings): Documents to clean.
            fit (bool): Whether to fit cleaner and display progress.
            
        Raises:
            None.
            
        Returns:
            cleaned_tokens (list of lists of strings): Cleaned tokens.
        """
        tokens = self._preprocess_text(documents, fit)
        cleaned_tokens = self._filter_tokens(tokens, fit)
        if self.bigrams:
            bigram_tokens = self._get_bigrams(tokens, fit)
            cleaned_tokens = [doc[0]+doc[1] for doc in zip(bigram_tokens, cleaned_tokens)]
        if self.word_count_min <= 1:
            return cleaned_tokens
        else:
            return self._filter_word_count(cleaned_tokens, fit)
        
    def fit_transform(self, documents):
        """
        Fits TextCleaner and transforms documents.
        
        Args:
            documents (list of strings): Documents to fit and transform.
            
        Raises:
            None.
            
        Returns:
            cleaned_tokens (list of lists of strings): Cleaned tokens.
        """
        cleaned_tokens = self._clean_documents(documents, True)
        self.word_counts = dict(self.word_counts)
        return cleaned_tokens
    
    def transform(self, documents):
        """
        Transforms documents given the fit TextCleaner.
        
        Args:
            documents (list of strings): Documents to transform.
            
        Raises:
            None.
            
        Returns:
            (list of lists of strings): Cleaned tokens.
        """
        return self._clean_documents(documents, False)