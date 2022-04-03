import re
import string
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
import itertools
from tqdm import tqdm
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')


def pos_filter(ngram):
    if ('-pron-' in ngram) or ('t' in ngram):
        return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if (tags[0][1] in acceptable_types) and (tags[1][1] in second_type):
        return True
    else:
        return False

# TODO: Make this work without Pandas
def get_bigrams(tokens, bigram_freq_min=20, pmi_min_value=1):
    # Create bigrams
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    # Filter bigrams by minimum frequency
    bigram_finder.apply_freq_filter(bigram_freq_min)

    # Calculate PMI
    bigrams_df = pd.DataFrame(list(bigram_finder.score_ngrams(nltk.collocations.BigramAssocMeasures().pmi)), 
                            columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
    bigrams_df = bigrams_df[bigrams_df['PMI'] > 1]

    # Filter bigrams based on POS
    bigrams_df = bigrams_df[bigrams_df['bigram'].apply(lambda x: pos_filter(x))]
    
    bigram_dict = bigrams_df['bigram'].apply(lambda x: {x:f'{x[0]} {x[1]}'}).values
    bigram_dict = {list(dict_.keys())[0]:list(dict_.values())[0] for dict_ in bigram_dict}

    return bigram_dict

def _tqdm(documents, display_progress=True):
    if display_progress:
        return tqdm(documents, total=len(documents), position=0)
    else:
        return documents

# TODO: Bigrams should be created before text cleaning? And then stop words removed
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

    def _clean_text(self, text, fit):
        text = self.remove_punctuation_rule.sub(' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip().lower()
        cleaned_text = []
        for word in text.split():
            if (word not in self.STOPWORDS) & (len(word) >= self.word_length_min):
                cleaned_text.append(word)
                if fit:
                    self.word_counts[word] += 1
        return cleaned_text

    def _clean_documents(self, documents, fit, display_progress):
        cleaned_documents = [self._clean_text(d, fit) for d in 
                             _tqdm(documents, display_progress=display_progress)]
        if self.bigrams:
            if fit:
                self.bigrams_dict = get_bigrams(list(itertools.chain(*cleaned_documents)), 
                                                bigram_freq_min=self.bigrams_min_freq, 
                                                pmi_min_value=self.bigrams_pmi_min_value)
            bigram_documents = (list(zip(cleaned_document[:-1], cleaned_document[1:])) 
                                for cleaned_document in cleaned_documents)
            for n, bigrams in enumerate(bigram_documents):
                temp = []
                for bigram in bigrams:
                    keep_bigram = self.bigrams_dict.get(bigram)
                    if keep_bigram is not None:
                        temp.append(keep_bigram)
                        if fit:
                            self.word_counts[keep_bigram] += 1
                cleaned_documents[n].extend(temp)
        if self.word_count_min <= 1:
            return cleaned_documents
        else:
            return [[word for word in cd if self.word_counts[word] > self.word_count_min] 
                    for cd in cleaned_documents]
        
    def fit_transform(self, documents):
        cleaned_documents = self._clean_documents(documents, fit=True, display_progress=True)
        self.word_counts = dict(self.word_counts)
        return cleaned_documents
    
    def transform(self, documents):
        return self._clean_documents(documents, fit=False, display_progress=False)