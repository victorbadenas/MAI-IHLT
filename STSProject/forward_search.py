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

        # Bigrams
        bigrams_1, bigrams_2 = get_ngrams(no_stopwords_1, no_stopwords_2, n=2)
        trigrams_1, trigrams_2 = get_ngrams(no_stopwords_1,
                                            no_stopwords_2,
                                            n=3)

        stopwords_and_lemmas1, stopwords_and_lemmas2 = get_lemmas(
            no_stopwords_1, no_stopwords_2, return_unique_words=False)

        stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2 = get_lemmas(
            no_stopwords_lc_1, no_stopwords_lc_2, return_unique_words=False)

        # Features
        features[index] = [
            jaccard_similarity(tokenized_1, tokenized_2),
            jaccard_similarity(tokenized_lc_1, tokenized_lc_2),
            jaccard_similarity(no_stopwords_1, no_stopwords_2),
            jaccard_similarity(no_stopwords_lc_1, no_stopwords_lc_2),
            jaccard_similarity(lemmatized_1, lemmatized_2),
            jaccard_similarity(lemmatized_lc_1, lemmatized_lc_2),
            jaccard_similarity(sentence_ne_1, sentence_ne_2),
            jaccard_similarity(bigrams_1, bigrams_2),
            jaccard_similarity(trigrams_1, trigrams_2),
            jaccard_similarity(stopwords_and_lemmas1, stopwords_and_lemmas2),
            jaccard_similarity(stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2),
            dice_similarity(tokenized_1, tokenized_2),
            dice_similarity(tokenized_lc_1, tokenized_lc_2),
            dice_similarity(no_stopwords_1, no_stopwords_2),
            dice_similarity(no_stopwords_lc_1, no_stopwords_lc_2),
            dice_similarity(lemmatized_1, lemmatized_2),
            dice_similarity(lemmatized_lc_1, lemmatized_lc_2),
            dice_similarity(sentence_ne_1, sentence_ne_2),
            dice_similarity(bigrams_1, bigrams_2),
            dice_similarity(trigrams_1, trigrams_2),
            dice_similarity(stopwords_and_lemmas1, stopwords_and_lemmas2),
            dice_similarity(stopwords_and_lemmas_lc_1, stopwords_and_lemmas_lc_2),
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




def fit_predict_model(x_train, y_train, x_test=None, y_test=None):
    # find best model
    pearson_scorer = make_scorer(lambda y, y_hat: pearsonr(y, y_hat)[0])

    gammas = np.logspace(-6, -1, 6)
    Cs = np.array([0.5, 1, 2, 4, 8, 10, 15, 20, 50, 100, 200, 375, 500, 1000])
    epsilons = np.linspace(0.1, 1, 10)
    param = dict(gamma=gammas, C=Cs, epsilon=epsilons)

    svr = SVR(kernel='rbf', tol=1)
    gssvr = GridSearchCV(svr, param, cv=5, scoring=pearson_scorer, n_jobs=-1)
    gssvr = gssvr.fit(x_train, y_train)
    if x_test is None and y_test is None:
        return gssvr.best_score_

    predictions = gssvr.predict(x_test)

    test_correlation = pearsonr(predictions, y_test)[0]
    return gssvr.best_score_, test_correlation


train_features_scores = get_feature_score(train_features_scaled, train_gs)
num_features = train_features_scaled.shape[-1]

# -----------------------------------------------------------------------
# -------------------------- Feature Selection --------------------------
# -----------------------------------------------------------------------

# selected_features = [np.argsort(train_features_scores)[-1]]
# selected_features_pearson = [np.max(train_features_scores)]
# all_features_indexes = set(range(num_features))

# from tqdm import tqdm
# for epoch_idx in range(num_features):
#     epoch_features_idx = all_features_indexes - set(selected_features)
#     epoch_correlations = np.zeros(num_features)
#     for feature_idx in tqdm(epoch_features_idx):
#         now_trying_features = list(selected_features) + [feature_idx]
#         sub_train_feats = train_features_scaled[:, now_trying_features]
#         epoch_correlations[feature_idx] = fit_predict_model(
#             sub_train_feats, train_gs)
#     if np.any(np.isnan(epoch_correlations)):
#         break
#     selected_features.append(np.argmax(epoch_correlations))
#     selected_features_pearson.append(np.max(epoch_correlations))
#     logging.info(
#         f'best feature {selected_features[-1]} with correlation {selected_features_pearson[-1]}'
#     )


# ----------------------------------------------------------------------
# ----------------------------- Inference ------------------------------
# ----------------------------------------------------------------------

ranking = [21, 7, 14, 18, 13, 8, 19, 11, 0, 2, 4, 17, 3, 10, 15, 6, 16, 1, 9, 12, 5, 20]
correlation = np.zeros_like(ranking).astype(np.float)

for i in range(num_features):
    now_trying_features = ranking[:i+1]
    sub_train_feats = train_features_scaled[:, now_trying_features]
    sub_test_feats = test_features_scaled[:, now_trying_features]
    train_correlation, correlation[i] = fit_predict_model(
        sub_train_feats, train_gs, sub_test_feats, test_gs)
    logging.info(
        f"Features {now_trying_features}: Train {train_correlation}, Test {correlation[i]}"
    )

logging.info('\n'.join(map(str, zip(ranking, correlation))))
