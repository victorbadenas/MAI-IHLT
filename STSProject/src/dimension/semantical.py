import nltk
from nltk import Nonterminal
from nltk import induce_pcfg
from nltk.parse import pchart
from nltk.parse import ViterbiParser
from nltk.corpus import treebank
nltk.download('treebank')


def train_grammar():
    productions = []
    for tree in treebank.parsed_sents():
        tree.collapse_unary(collapsePOS = False) #  Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2) #  Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()
    S = Nonterminal('S')
    return induce_pcfg(S, productions)

# GRAMMAR = train_grammar()

# PARSERS = {
#     'ViterbiParser': ViterbiParser(GRAMMAR),
#     'InsideChartParser': pchart.InsideChartParser(GRAMMAR),
#     'RandomChartParser': pchart.RandomChartParser(GRAMMAR),
#     'UnsortedChartParser': pchart.UnsortedChartParser(GRAMMAR),
#     'LongestChartParser': pchart.LongestChartParser(GRAMMAR)
# }

def get_sentences_trees_nodes(sentence1: list, sentence2: list, parser: str):
    return get_sentence_tree_nodes(sentence1, parser), get_sentence_tree_nodes(sentence1, parser)

def get_sentence_tree_nodes(sentence: list, parser: str):
    assert parser in PARSERS, f"parser \'{parser}\' not supported"
    parser = PARSERS[parser]
    try:
        trees = list(parser.parse(sentence))
    except ValueError as e:
        print(e)
        trees = []
    if len(trees) <= 0:
        return []
    get_nodes(trees[0])
    return get_nodes(trees[0])

ROOT = 'ROOT'

def get_nodes(parent):
    trees = []
    for node in parent:
        if isinstance(node, nltk.Tree):
            trees.append(node)
            subtrees = get_nodes(node)
            trees += subtrees
    print(trees)
    return trees

def get_ngrams(sentence1: list, sentence2: list, n: int):
    return get_ngram(sentence1, n), get_ngram(sentence1, n)

def get_ngram(sentence: list, n: int):
    return list(nltk.ngrams(sentence, n))
