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

GRAMMAR = train_grammar()

PARSERS = {
    'ViterbiParser': ViterbiParser(GRAMMAR),
    'InsideChartParser': pchart.InsideChartParser(GRAMMAR),
    'RandomChartParser': pchart.RandomChartParser(GRAMMAR),
    'UnsortedChartParser': pchart.UnsortedChartParser(GRAMMAR),
    'LongestChartParser': pchart.LongestChartParser(GRAMMAR)
}
