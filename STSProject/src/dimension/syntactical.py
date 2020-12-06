import nltk
from nltk import pos_tag, ne_chunk, Tree
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from itertools import product

from dimension import lexical
from dimension.lexical import filter_stopwords
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
brown_ic = wordnet_ic.ic('ic-brown.dat')


def get_named_entities(first_sentence: str, second_sentence: str):
    return apply_ne(first_sentence), apply_ne(second_sentence)


def apply_ne(sentence: str):
    sentences_ne = list(ne_chunk(pos_tag(sentence), binary=True))
    result = []
    for el in sentences_ne:
        if isinstance(el, Tree):
            leaves = el.leaves()
            result.append(" ".join(word[0] for word in leaves))
        else:
            result.append(el[0])
    return result


def get_lesk_sentences(first_sentence: str, second_sentence: str):
    return apply_lesk(first_sentence), apply_lesk(second_sentence)


def apply_lesk(sentence):
    sentence_tagged = pos_tag(sentence)
    resulting_sentence = []
    for word, tag in sentence_tagged:
        if tag.startswith('N') and (type(lesk(sentence, word, wn.NOUN)) != type(None)):
            resulting_sentence.append(lesk(sentence, word, wn.NOUN).name())
        elif tag.startswith('V') and (type(lesk(sentence, word, wn.VERB)) != type(None)):
            resulting_sentence.append(lesk(sentence, word, wn.VERB).name())
        elif tag.startswith('J') and (type(lesk(sentence, word, wn.ADJ)) != type(None)):
            resulting_sentence.append(lesk(sentence, word, wn.ADJ).name())
        elif tag.startswith('R') and (type(lesk(sentence, word, wn.ADV)) != type(None)):
            resulting_sentence.append(lesk(sentence, word, wn.ADV).name())
        else:
            resulting_sentence.append(word)
    return resulting_sentence


def get_synset_similarity(first_sentence: str, second_sentence: str, method: str):
    sentence_tagged_1 = pos_tag(first_sentence)
    sentence_tagged_2 = pos_tag(second_sentence)

    sentence_tagged_wn_1 = get_sentences_tagged_with_wn_and_cleaned(sentence_tagged_1)
    sentence_tagged_wn_2 = get_sentences_tagged_with_wn_and_cleaned(sentence_tagged_2)

    synsets = {}
    synsets, key_list1 = get_synset_tag(sentence_tagged_wn_1, synsets)
    synsets, key_list2 = get_synset_tag(sentence_tagged_wn_2, synsets)

    synsets_combinations = list(product(key_list1, key_list2))
    resulting_similarity = []
    for first_word, second_word in synsets_combinations:
        if first_word == second_word:
            resulting_similarity.append(1)
            continue
        first_synset = synsets[first_word][0]
        second_synset = synsets[second_word][0]
        first_tag = synsets[first_word][1]
        second_tag = synsets[second_word][1]
        if method == "path":
            path_sim = first_synset.path_similarity(second_synset)
            if path_sim is None:
                resulting_similarity.append(0)
            else:
                resulting_similarity.append(path_sim)

        if method == "lch":
            if first_tag == second_tag:
                lch_sim = wn.lch_similarity(first_synset, second_synset)
                if lch_sim is None:
                    resulting_similarity.append(0)
                else:
                    lch_norm = lch_sim / wn.lch_similarity(first_synset, first_synset)
                    resulting_similarity.append(lch_norm)

        if method == "wup":
            wup_sim = first_synset.wup_similarity(second_synset)
            if wup_sim is None:
                resulting_similarity.append(0)
            else:
                resulting_similarity.append(wup_sim)

        if method == "lin":
            if first_tag == second_tag and first_tag in ['n', 'v']:
                lin_sim = first_synset.lin_similarity(second_synset, brown_ic)
                if lin_sim is None:
                    resulting_similarity.append(0)
                else:
                    resulting_similarity.append(lin_sim)

    if not resulting_similarity:
        return 0
    else:
        return sum(resulting_similarity) / len(resulting_similarity)


def get_synset_tag(sentence_tagged, synsets):
    key_list = []
    for pair in sentence_tagged:
        synset = wn.synsets(pair[0], pair[1])
        if synset:
            synsets[pair[0]] = (synset[0], synset[0].pos())
            key_list.append(pair[0])
    return synsets, key_list


def get_sentences_tagged_with_wn_and_cleaned(sentence_tagged: str):
    pair_cleaned = []
    for pair in sentence_tagged:
        wordnet_tag = get_wordnet_pos(pair[1])
        if wordnet_tag is not None:
            pair_cleaned.append((pair[0], wordnet_tag))
    return pair_cleaned


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None
