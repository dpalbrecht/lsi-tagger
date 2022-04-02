import re
import string
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
from tqdm import tqdm


def _tqdm(documents, display_progress=True):
    if display_progress:
        return tqdm(documents, total=len(documents), position=0)
    else:
        return documents

# TODO: Add optional n-grams that use the n-gram as one token as well as the unigrams that compose it
class TextCleaner:
    def __init__(self, word_count_min=0, word_length_min=2):
        self.remove_punctuation_rule = re.compile(f"[{re.escape(string.punctuation)}]")
        self.word_counts = defaultdict(lambda :0)
        self.word_count_min = word_count_min
        self.word_length_min = word_length_min

    def _clean_text(self, text):
        text = self.remove_punctuation_rule.sub(' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip().lower()
        cleaned_text = []
        for word in text.split():
            if (word not in STOPWORDS) & (len(word) >= self.word_length_min):
                cleaned_text.append(word)
                self.word_counts[word] += 1
        return cleaned_text

    def clean_documents(self, documents, display_progress=True):
        cleaned_documents = [self._clean_text(d) for d in 
                             _tqdm(documents, display_progress=display_progress)]
        if self.word_count_min == 0:
            return cleaned_documents
        else:
            return [[word for word in cd if self.word_counts[word] > self.word_count_min] 
                    for cd in cleaned_documents]