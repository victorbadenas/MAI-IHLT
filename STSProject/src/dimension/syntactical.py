import nltk
from nltk import pos_tag, ne_chunk, Tree

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
