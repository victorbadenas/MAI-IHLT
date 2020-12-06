#!/usr/bin/env python
# coding: utf-8



import sys
sys.path.append('src')
import nltk
import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from nltk.metrics.distance import jaccard_distance
from collections.abc import Iterable
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from data_utils import load_data
from dimension.lexical import *
from dimension.syntactical import *

def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

set_logger('test.log', debug=True)

train_data, test_data = load_data('data/')

logging.info(
    f"train_data samples: {len(train_data)}, test_data samples: {len(test_data)}"
)


def jaccard_similarity(s1, s2):
    assert isinstance(s1, Iterable), f"s1 must be an iterable, not {type(s1)}"
    assert isinstance(s2, Iterable), f"s2 must be an iterable, not {type(s2)}"
    return 1 - jaccard_distance(set(s1), set(s2))


def overlap_similarity(s1, s2):
    assert isinstance(s1, Iterable), f"s1 must be an iterable, not {type(s1)}"
    assert isinstance(s2, Iterable), f"s2 must be an iterable, not {type(s2)}"
    s1 = set(s1)
    s2 = set(s2)
    intersection = s1.intersection(s2)
    return len(intersection) / min(len(s1), len(s2))


def cosine_similarity(s1, s2):
    assert isinstance(s1, Iterable), f"s1 must be an iterable, not {type(s1)}"
    assert isinstance(s2, Iterable), f"s2 must be an iterable, not {type(s2)}"
    s1 = set(s1)
    s2 = set(s2)
    intersection = s1.intersection(s2)
    return len(intersection) / ((len(s1) * len(s2))**2)


def dice_similarity(s1, s2):
    assert isinstance(s1, Iterable), f"s1 must be an iterable, not {type(s1)}"
    assert isinstance(s2, Iterable), f"s2 must be an iterable, not {type(s2)}"
    s1 = set(s1)
    s2 = set(s2)
    intersection = s1.intersection(s2)
    return 2 * len(intersection) / (len(s1) + len(s2))




def get_features(df: pd.DataFrame):
    assert "S1" in df.columns, "S1 not in dataframe"
    assert "S2" in df.columns, "S2 not in dataframe"

    features = [None] * len(df)  # preallocated for memory efficiency

    for index, row in df.iterrows():
        sentence1, sentence2 = row['S1'], row['S2']

        # Get all words
        tokenized_1, tokenized_2 = get_tokenized_sentences(
            sentence1, sentence2, return_unique_words=False)
        tokenized_lc_1, tokenized_lc_2 = get_tokenized_sentences_lowercase(
            tokenized_1, tokenized_2, return_unique_words=False)

        # Get words without stopwords
        no_stopwords_1, no_stopwords_2 = filter_stopwords(
            tokenized_1, tokenized_2, return_unique_words=False)
        no_stopwords_lc_1, no_stopwords_lc_2 = filter_stopwords(
            tokenized_lc_1, tokenized_lc_2, return_unique_words=False)

        # Lemmas
        lemmatized_1, lemmatized_2 = get_lemmas(tokenized_1,
                                                tokenized_2,
                                                return_unique_words=False)
        lemmatized_lc_1, lemmatized_lc_2 = get_lemmas(
            tokenized_lc_1, tokenized_lc_2, return_unique_words=False)

        # Name entities
        sentence_ne_1, sentence_ne_2 = get_named_entities(
            tokenized_1, tokenized_2)
        
        #lemmas cleaned from stopwords
        stopwords_and_lemmas1, stopwords_and_lemmas2 = get_lemmas(
            no_stopwords_1, no_stopwords_2, return_unique_words=False)

        stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2 = get_lemmas(
            no_stopwords_lc_1, no_stopwords_lc_2, return_unique_words=False)
        
        # Name entities without stopwords in lowercase
        ne_no_stopwords_1, ne_no_stopwords_2 = filter_stopwords(
            sentence_ne_1, sentence_ne_2, return_unique_words=False, filter_and_return_in_lowercase=True)
        
        # Name entities without stopwords in lowercase and lemmas
        ne_no_stopwords_lemmas_1, ne_no_stopwords_lemmas_2 = get_lemmas( ne_no_stopwords_1, ne_no_stopwords_2, 
                                                                        return_unique_words=False)

        # Bigrams
        bigrams_1, bigrams_2 = get_ngrams(no_stopwords_1, no_stopwords_2, n=2)
        trigrams_1, trigrams_2 = get_ngrams(no_stopwords_1, no_stopwords_2, n=3)
        
        # Bigrams trigrams with sentence tokenizer 
        bigrams_sent_1, bigrams_sent_2 = get_ngrams_with_sent_tokenize(sentence1, sentence2, n=2)
        trigrams_sent_1, trigrams_sent_2 = get_ngrams_with_sent_tokenize(sentence1, sentence2, n=3)
        
        # Lesk
        lesk_1, lesk_2 = get_lesk_sentences(tokenized_1, tokenized_2)
        lesk_lc_1, lesk_lc_2 = get_lesk_sentences(tokenized_lc_1, tokenized_lc_2)


        # Stemmer
        stemmed_1, stemmed_2 = get_stemmed_sentences(sentence1, sentence2)
        
        
        # Synset
        average_path = get_synset_similarity(tokenized_1, tokenized_2, "path")
        average_lch = get_synset_similarity(tokenized_1, tokenized_2, "lch")
        average_wup = get_synset_similarity(tokenized_1, tokenized_2, "wup")
        average_lin = get_synset_similarity(tokenized_1, tokenized_2, "lin")
        
        average_lc_path = get_synset_similarity(tokenized_lc_1, tokenized_lc_2, "path")
        average_lc_lch = get_synset_similarity(tokenized_lc_1, tokenized_lc_2, "lch")
        average_lc_wup = get_synset_similarity(tokenized_lc_1, tokenized_lc_2, "wup")
        average_lc_lin = get_synset_similarity(tokenized_lc_1, tokenized_lc_2, "lin")
        
        
        # ALL Features
        features[index] = [
            jaccard_similarity(tokenized_1, tokenized_2),
            jaccard_similarity(tokenized_lc_1, tokenized_lc_2),
            jaccard_similarity(no_stopwords_1, no_stopwords_2),
            jaccard_similarity(no_stopwords_lc_1, no_stopwords_lc_2),
            jaccard_similarity(lemmatized_1, lemmatized_2),
            jaccard_similarity(lemmatized_lc_1, lemmatized_lc_2),
            jaccard_similarity(sentence_ne_1, sentence_ne_2),
            jaccard_similarity(stopwords_and_lemmas1, stopwords_and_lemmas2),
            jaccard_similarity(stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2),
            jaccard_similarity(bigrams_1, bigrams_2),
            jaccard_similarity(trigrams_1, trigrams_2),
            jaccard_similarity(bigrams_sent_1, bigrams_sent_2),
            jaccard_similarity(trigrams_sent_1, trigrams_sent_2),
            jaccard_similarity(lesk_1, lesk_2),
            jaccard_similarity(lesk_lc_1, lesk_lc_2),
            jaccard_similarity(stemmed_1, stemmed_2),
            
            dice_similarity(tokenized_1, tokenized_2),
            dice_similarity(tokenized_lc_1, tokenized_lc_2),
            dice_similarity(no_stopwords_1, no_stopwords_2),
            dice_similarity(no_stopwords_lc_1, no_stopwords_lc_2),
            dice_similarity(lemmatized_1, lemmatized_2),
            dice_similarity(lemmatized_lc_1, lemmatized_lc_2),
            dice_similarity(sentence_ne_1, sentence_ne_2),
            dice_similarity(stopwords_and_lemmas1, stopwords_and_lemmas2),
            dice_similarity(stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2),
            dice_similarity(bigrams_1, bigrams_2),
            dice_similarity(trigrams_1, trigrams_2),
            dice_similarity(bigrams_sent_1, bigrams_sent_2),
            dice_similarity(trigrams_sent_1, trigrams_sent_2),
            dice_similarity(lesk_1, lesk_2),
            dice_similarity(lesk_lc_1, lesk_lc_2),
            dice_similarity(stemmed_1, stemmed_2),
            
            average_path,
            average_lch,
            average_wup,
            average_lin,
            average_lc_path,
            average_lc_lch,
            average_lc_wup,
            average_lc_lin
        ]
    return np.array(features)


train_features = get_features(train_data)
train_gs = train_data['Gs'].to_numpy()
logging.info(f"train_features.shape: {train_features.shape}")
logging.info(f"train_gs.shape: {train_gs.shape}")


test_features = get_features(test_data)
test_gs = test_data['Gs'].to_numpy()
logging.info(f"test_features.shape: {test_features.shape}")
logging.info(f"test_gs.shape: {test_gs.shape}")


scaler = StandardScaler()
scaler.fit(train_features)
train_features_scaled = scaler.transform(train_features)
test_features_scaled = scaler.transform(test_features)


def get_feature_score(feats, gs):
    return np.array([pearsonr(feats[:, i], gs)[0] for i in range(feats.shape[-1])])


from sklearn.model_selection import PredefinedSplit
all_data = np.concatenate([train_features_scaled, test_features_scaled])
all_labels = np.concatenate([train_gs, test_gs])
test_fold = np.array([-1]*train_features_scaled.shape[0] + [0]*test_features_scaled.shape[0])
logging.info(f"joined dataset shapes: {all_data.shape} and {test_fold.shape}")
ps = PredefinedSplit(test_fold)


def fit_predict_model(all_data, all_labels):
    # find best model
    pearson_scorer = make_scorer(lambda y, y_hat: pearsonr(y, y_hat)[0])

    gammas = np.logspace(-6, -1, 6)
    Cs = np.array([0.5, 1, 2, 4, 8, 10, 15, 20, 50, 100, 200, 375, 500, 1000])
    epsilons = np.linspace(0.1, 1, 10)
    param = dict(gamma=gammas, C=Cs, epsilon=epsilons)

    svr = SVR(kernel='rbf', tol=1)
    gs = GridSearchCV(svr, param, cv=ps, scoring=pearson_scorer, n_jobs=-1)
    gs = gs.fit(all_data, all_labels)
    best_parameters = gs.best_params_

    train_idx, test_idx = list(ps.split())[0]
    x_train = all_data[train_idx]
    y_train = all_labels[train_idx]
    x_test = all_data[test_idx]
    y_test = all_labels[test_idx]

    best_model = SVR(kernel='rbf', tol=1, **best_parameters)
    train_pred = best_model.fit(x_train, y_train).predict(x_train)
    test_pred = best_model.predict(x_test)
    return pearsonr(train_pred, y_train)[0], pearsonr(test_pred, y_test)[0]

train_features_scores = get_feature_score(train_features_scaled, train_gs)
logging.info(f"features correlation scores: {train_features_scores}")
num_features = train_features_scaled.shape[-1]

# -----------------------------------------------------------------------
# -------------------------- Feature Selection --------------------------
# -----------------------------------------------------------------------

selected_features = [np.argsort(train_features_scores)[-1]]
selected_features_test_pearson = [None]
selected_features_train_pearson = [None]
all_features_indexes = set(range(num_features))

from tqdm import tqdm
for epoch_idx in range(num_features-1):
    epoch_features_idx = all_features_indexes - set(selected_features)
    epoch_train_correlations = np.zeros(num_features)
    epoch_test_correlations = np.zeros(num_features)
    for feature_idx in tqdm(epoch_features_idx):
        now_trying_features = list(selected_features) + [feature_idx]
        sub_train_feats = all_data[:, now_trying_features]
        epoch_train_correlations[feature_idx], epoch_test_correlations[feature_idx] = fit_predict_model(sub_train_feats, all_labels)
    if np.any(np.isnan(epoch_test_correlations)):
        break
    selected_features.append(np.argmax(epoch_test_correlations))
    selected_features_train_pearson.append(np.max(epoch_train_correlations))
    selected_features_test_pearson.append(np.max(epoch_test_correlations))
    logging.info(
        f'best feature {selected_features[-1]} train: {selected_features_train_pearson[-1]} test: {selected_features_test_pearson[-1]}'
    )
