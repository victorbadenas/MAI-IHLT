import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def get_tokenized_sentences(first_sentence: str, second_sentence: str, return_unique_words=False):
    first_tokenized = nltk.word_tokenize(first_sentence)
    second_tokenized = nltk.word_tokenize(second_sentence)
    if return_unique_words:
        return get_unique_words(first_tokenized), get_unique_words(second_tokenized)
    else:
        return first_tokenized, second_tokenized


def get_tokenized_sentences_lowercase(first_sentence: str, second_sentence: str, return_unique_words=False):
    first_sentence_lc = [word.lower() for word in first_sentence]
    second_sentence_lc = [word.lower() for word in second_sentence]
    if return_unique_words:
        return get_unique_words(first_sentence_lc), get_unique_words(second_sentence_lc)
    else:
        return first_sentence_lc, second_sentence_lc


def get_tokenized_without_stopwords(first_sentence: str, second_sentence: str, return_unique_words=False):
    first_cleaned = clean_sentence_from_stopword(first_sentence)
    second_cleaned = clean_sentence_from_stopword(second_sentence)
    if return_unique_words:
        return get_unique_words(first_cleaned), get_unique_words(second_cleaned)
    else:
        return first_cleaned, second_cleaned


def clean_sentence_from_stopword(sentence):
    stopwords_list = set(stopwords.words('english'))
    cleaned_text = [word for word in sentence if (word not in stopwords_list and word not in punctuation)]
    return cleaned_text


def get_lemmas(first_sentence: str, second_sentence: str, return_unique_words=False):
    first_tagged, second_tagged = pos_tagger(first_sentence, second_sentence)
    first_lemmatized = [extract_lemma(pair) for pair in first_tagged]
    second_lemmatized = [extract_lemma(pair) for pair in second_tagged]
    if return_unique_words:
        return get_unique_words(first_lemmatized), get_unique_words(second_lemmatized)
    else:
        return first_lemmatized, second_lemmatized


def pos_tagger(first_sentence: str, second_sentence: str):
    return pos_tag(first_sentence), pos_tag(second_sentence)


def extract_lemma(word_and_tag):
    wnl = WordNetLemmatizer()
    tag = word_and_tag[1][:2]
    if tag == 'NN':
        return wnl.lemmatize(word_and_tag[0], wn.NOUN)
    elif tag == 'VB':
        return wnl.lemmatize(word_and_tag[0], wn.VERB)
    elif tag == 'JJ':
        return wnl.lemmatize(word_and_tag[0], wn.ADJ)
    elif tag == 'RB':
        return wnl.lemmatize(word_and_tag[0], wn.ADV)
    return word_and_tag[0]


def get_unique_words(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def get_ngrams(sentence1: str, sentence2: str, n: int):
    sentences1 = nltk.sent_tokenize(sentence1)
    sentences2 = nltk.sent_tokenize(sentence2)
    sentence1_ngrams = []
    for s1 in sentences1:
        tokenized_1 = nltk.word_tokenize(s1)
        no_stopwords_1 = clean_sentence_from_stopword(tokenized_1)
        sentence1_ngrams += get_ngram(no_stopwords_1, n)
    sentence2_ngrams = []
    for s2 in sentences2:
        tokenized_2 = nltk.word_tokenize(s2)
        no_stopwords_2 = clean_sentence_from_stopword(tokenized_2)
        sentence2_ngrams += get_ngram(no_stopwords_2, n)
    return sentence1_ngrams, sentence2_ngrams


def get_ngram(sentence: list, n: int):
    if len(sentence) < n:
        return [tuple(sentence)]
    return list(nltk.ngrams(sentence, n))
