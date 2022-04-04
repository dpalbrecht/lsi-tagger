import re
import string
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
import itertools
from datetime import datetime
import nltk
nltk.download('averaged_perceptron_tagger')


class TextCleaner:
    def __init__(self, word_count_min=1, word_length_min=2, bigram_kwargs={}):
        self.remove_punctuation_rule = re.compile(f"[{re.escape(string.punctuation)}]")
        self.word_counts = defaultdict(lambda :0)
        self.word_count_min = word_count_min
        self.word_length_min = word_length_min
        self.bigrams = (len(bigram_kwargs)>0)
        self.bigrams_pmi_min_value = bigram_kwargs.get('bigrams_pmi_min_value', 1)
        self.bigrams_min_freq = bigram_kwargs.get('bigrams_min_freq', 20)
        keep_stopwords = ['top','bottom','back','front','full','her','him','herself',
                          'himself','his','hers','kg','km','cm','thick','thin','under',
                          'you','your','yours']
        self.STOPWORDS = keep_stopwords = [s for s in STOPWORDS if s not in keep_stopwords]
        
    def timeit(func):
        def wrapper(*args):
            if args[2]:
                start = datetime.utcnow()
                process_name = func.__name__.lstrip('_').replace('_',' ').title()
                print(f"Starting the '{process_name}' process...")
            result = func(*args)
            if args[2]:
                print(f'Took {(datetime.utcnow() - start).total_seconds()/60:.2f} minutes.')
            return result
        return wrapper
        
    def _pos_filter(self, ngram):
        if ('-pron-' in ngram) or ('t' in ngram):
            return False
        for word in ngram:
            if (word in self.STOPWORDS) or word.isspace() or (word in string.punctuation):
                return False
        acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
        second_type = ('NN', 'NNS', 'NNP', 'NNPS')
        tags = nltk.pos_tag(ngram)
        if (tags[0][1] in acceptable_types) and (tags[1][1] in second_type):
            return True
        else:
            return False

    def _create_bigrams_dict(self, tokens):
        # Create bigrams
        bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

        # Filter bigrams by minimum frequency
        bigram_finder.apply_freq_filter(self.bigrams_min_freq)

        # Calculate PMI
        self.bigrams_dict = list(bigram_finder.score_ngrams(nltk.collocations.BigramAssocMeasures().pmi))

        # Filter bigrams based on POS and min PMI
        self.bigrams_dict = {b[0]:f'{b[0][0]} {b[0][1]}' for b in self.bigrams_dict 
                             if ((b[1]>1) and self._pos_filter(b[0]))}
        
    @timeit
    def _get_bigrams(self, tokens, fit):
        if fit:
            self._create_bigrams_dict(list(itertools.chain(*tokens)))
        bigram_documents = []
        for bigrams in (list(zip(words[:-1], words[1:])) for words in tokens):
            temp_bigrams = []
            for bigram in bigrams:
                keep_bigram = self.bigrams_dict.get(bigram)
                if keep_bigram is not None:
                    temp_bigrams.append(keep_bigram)
                    if fit:
                        self.word_counts[keep_bigram] += 1
            bigram_documents.append(temp_bigrams)
        return bigram_documents
    
    @timeit
    def _preprocess_text(self, documents, fit):
        tokens = []
        for text in documents:
            text = self.remove_punctuation_rule.sub(' ', text)
            text = re.sub(' +', ' ', text)
            tokens.append(text.strip().lower().split())
        return tokens

    @timeit
    def _clean_tokens(self, documents, fit):
        cleaned_tokens = []
        for tokens in documents:
            temp_tokens = []
            for word in tokens:
                if (word not in self.STOPWORDS) & (len(word) >= self.word_length_min):
                    temp_tokens.append(word)
                    if fit:
                        self.word_counts[word] += 1
            cleaned_tokens.append(temp_tokens)
        return cleaned_tokens
    
    @timeit
    def _filter_word_count(self, cleaned_tokens, fit):
        return [[word for word in ct if (self.word_counts.get(word, 0) > self.word_count_min)] 
                for ct in cleaned_tokens]

    def _clean_documents(self, documents, fit):
        documents = self._preprocess_text(documents, fit)
        cleaned_tokens = self._clean_tokens(documents, fit)
        if self.bigrams:
            bigram_tokens = self._get_bigrams(documents, fit)
            cleaned_tokens = [doc[0]+doc[1] for doc in zip(bigram_tokens, cleaned_tokens)]
        if self.word_count_min <= 1:
            return cleaned_tokens
        else:
            return self._filter_word_count(cleaned_tokens, fit)
        
    def fit_transform(self, documents):
        print("Starting the 'Text Cleaner' process...")
        cleaned_tokens = self._clean_documents(documents, True)
        self.word_counts = dict(self.word_counts)
        print('Done!')
        return cleaned_tokens
    
    def transform(self, documents):
        return self._clean_documents(documents, False)