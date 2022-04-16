from ..text_cleaner import TextCleaner
import pytest



@pytest.mark.parametrize("bigrams,expected_result",
                         [
                             (('short','sleeve'), True),
                             (('a','sleeve'), False),
                             (('the','sleeve'), False)
                         ])
def test_pos_filter(bigrams, expected_result):
    tc = TextCleaner(word_count_min=1, word_length_min=2, 
                     bigram_kwargs={'bigrams_pmi_min_value':1, 'bigrams_min_freq':20})
    assert tc._pos_filter(bigrams) == expected_result
    

@pytest.mark.parametrize("tokens,expected_result",
                         [
                             (['short','sleeve'], {('short','sleeve'):'short sleeve'}),
                             (['a','sleeve'], {}),
                             (['the','sleeve'], {})
                         ])
def test_create_bigrams_dict(tokens, expected_result):
    tc = TextCleaner(word_count_min=1, word_length_min=2, 
                     bigram_kwargs={'bigrams_pmi_min_value':0, 'bigrams_min_freq':0})
    tc._create_bigrams_dict(tokens)
    assert tc.bigrams_dict == expected_result
    
    
@pytest.mark.parametrize("tokens,expected_result",
                         [
                             ([['short','sleeve']], [['short sleeve']]),
                             ([['a','sleeve']], [[]]),
                             ([['the','sleeve']], [[]])
                         ])
def test_get_bigrams(tokens, expected_result):
    tc = TextCleaner(word_count_min=1, word_length_min=2, 
                     bigram_kwargs={'bigrams_pmi_min_value':0, 'bigrams_min_freq':0})
    assert tc._get_bigrams(tokens, fit=True) == expected_result


def test_preprocess_text_filter_tokens():
    tc = TextCleaner(word_count_min=1, word_length_min=2, bigram_kwargs={})
    documents = ['This is a short, exciting DoCuMeNt!!!', 'This is another.']
    tokenized_documents = tc._preprocess_text(documents, fit=True)
    assert tokenized_documents == [['this','is','a','short','exciting','document'], 
                                   ['this','is','another']]
    cleaned_tokens= tc._filter_tokens(tokenized_documents, fit=True)
    assert cleaned_tokens == [['short','exciting','document'],[]]
    

def test_filter_word_count():
    tc = TextCleaner(word_count_min=2, word_length_min=2, bigram_kwargs={})
    tc.word_counts = {'short':1,'exciting':2,'document':3}
    cleaned_tokens = [['short','exciting','document']]
    filtered_tokens = tc._filter_word_count(cleaned_tokens, fit=False)
    assert filtered_tokens == [['exciting','document']]
    
    
def test_clean_documents_no_bigrams():
    tc = TextCleaner(word_count_min=1, word_length_min=2, bigram_kwargs={})
    documents = ['This is a short, exciting DoCuMeNt!!!', 'This is another.']
    cleaned_tokens = tc._clean_documents(documents, fit=True)
    assert cleaned_tokens == [['short','exciting','document'],[]]
    

def test_clean_documents_yes_bigrams():
    tc = TextCleaner(word_count_min=1, word_length_min=2, 
                     bigram_kwargs={'bigrams_pmi_min_value':0, 'bigrams_min_freq':0})
    documents = ['short sleeve', 'another short sleeve']
    cleaned_tokens = tc._clean_documents(documents, fit=True)
    assert cleaned_tokens == [['short sleeve','short','sleeve'],
                              ['short sleeve','short','sleeve']]