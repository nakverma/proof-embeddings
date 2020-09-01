"""
===================
bow_representation.py
===================
Author:
Date:
Create bag-of-words (bow) bigram/unigram representation of logic expressions
"""

import sys
from create_expressions_mistakes import *
import pickle as pkl
import string
import random
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import copy




class Bag_of_words():

    def __init__(self, source=None, out=None):
        # variables = set('pqr')
        # variables.add("T")
        # variables.add("F")
        # operators = {'∧', '∨', '→', '~'}
        # parenthesis = {'(', ')'}
        # delimiters = {'STA', 'EOS'}
        # all_symbols = variables.union(operators).union(parenthesis).union(delimiters)

        all_symbols = ['p', 'q', 'r']
        all_symbols.append('T')
        all_symbols.append('F')
        all_symbols.extend(['∧','∨','→','↔','~'])
        all_symbols.extend(['(', ')'])
        all_symbols.extend(['STA', 'EOS'])


        self.symbols = {}
        for symbol in all_symbols:
            self.symbols[symbol] = len(self.symbols)
        self.bigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                self.bigrams[(symbol1, symbol2)] = len(self.bigrams)
        self.trigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                for symbol3 in all_symbols:
                    self.trigrams[(symbol1, symbol2, symbol3)] = len(self.trigrams)

            self.trigrams[('STA', symbol1, None)] = len(self.trigrams)
            self.trigrams[(None, symbol1, 'EOS')] = len(self.trigrams)


        if source == None:
            self.trainer = pkl.load(open('../data/trainer.pkl', 'rb'))
        else:
            self.source = source
            source_path = '../data/' + source + '_trainer.pkl'
            self.trainer = pkl.load(open(source_path, 'rb'))

        if out == None:
            self.dump_loc = '../data/'
        else:
            self.dump_loc = '../data/' + out

        self.dataset = self.trainer.trees
        print(len(self.dataset))
        self.check_valid_dataset()

    def check_valid_dataset(self):
        for tree_id, trees in self.dataset.items():
            for symbol in trees[0].parse_tree():
                if symbol not in self.symbols:
                    print(symbol)
                    assert symbol in self.symbols
            for treetup in trees[1]:
                for symbol in treetup[0].parse_tree():
                    if symbol not in self.symbols:
                        print(symbol)
                        assert symbol in self.symbols

    def bow_unigram(self, expr):
        unigrams_count = defaultdict(int)
        for i in range(0, len(expr)):
            unigrams_count[expr[i]] += 1

        unigram_index_count = [0] * len(self.symbols)
        for expr in unigrams_count.keys():
            unigram_index_count[self.symbols[expr]] = unigrams_count[expr]
        return unigram_index_count

    def bow_bigram(self, expr):
        bigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 1):
            bigrams_count[(expr[i], expr[i + 1])] += 1
        bigrams_count[('STA', expr[0])] += 1
        bigrams_count[(expr[len(expr) - 1], 'EOS')] += 1

        bigram_index_count = [0] * len(self.bigrams)
        for (expr1, expr2) in bigrams_count.keys():
            bigram_index_count[self.bigrams[(expr1, expr2)]] = bigrams_count[(expr1, expr2)]
        return bigram_index_count

    def bow_trigram(self, expr):
        trigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 2):
            trigrams_count[(expr[i], expr[i + 1], expr[i + 2])] += 1
        try:
            trigrams_count[('STA', expr[0], expr[1])] += 1
        except IndexError:
            trigrams_count[('STA', expr[0], None)] += 1
        try:
            trigrams_count[(expr[len(expr) - 2], expr[len(expr) - 1], 'EOS')] += 1
        except IndexError:
            trigrams_count[(None, expr[len(expr) - 1], 'EOS')] += 1

        trigram_index_count = [0] * len(self.trigrams)
        for (expr1, expr2, expr3) in trigrams_count.keys():
            trigram_index_count[self.trigrams[(expr1, expr2, expr3)]] = trigrams_count[(expr1, expr2, expr3)]
        return trigram_index_count

    def bow_representation_expr(self, expr, trigrams=False):
        if not trigrams:
            return self.bow_unigram(expr) + self.bow_bigram(expr)
        else:
            return self.bow_unigram(expr) + self.bow_bigram(expr) + self.bow_trigram(expr)

    def bow_representation_bigram_dataset(self):
        bow_dataset = {}
        for tree_id, trees in self.dataset.items():
            tree_bow = self.bow_representation_expr(trees[0].parse_tree())
            tree_sequence = trees[1]
            new_tree_sequence = []
            for treetup in tree_sequence:
                new_tree_sequence.append((treetup[0].parse_tree(), treetup[1], \
                    self.bow_representation_expr(treetup[0].parse_tree())))
            bow_dataset[tree_id] = ((trees[0].parse_tree(), tree_bow), new_tree_sequence)
        print("dumping")
        pkl.dump(bow_dataset, open(self.dump_loc + '_bigram_dataset.pkl', 'wb'))

    def bow_representation_trigram_dataset(self):
        bow_dataset = {}
        for tree_id, trees in self.dataset.items():
            tree_bow = self.bow_representation_expr(trees[0].parse_tree(), trigrams=True)
            tree_sequence = trees[1]
            new_tree_sequence = []
            for treetup in tree_sequence:
                new_tree_sequence.append((treetup[0].parse_tree(), treetup[1], \
                    self.bow_representation_expr(treetup[0].parse_tree(), trigrams=True)))
            bow_dataset[tree_id] = ((trees[0].parse_tree(), tree_bow), new_tree_sequence)
        print("dumping")
        pkl.dump(bow_dataset, open(self.dump_loc + '_trigram_dataset.pkl', 'wb'))


    def dump_raw_matrix(self):
        dataset = pkl.load(open(self.dump_loc + '_bigram_dataset.pkl', 'rb'))
        matrix = []
        for tup in dataset.values():
            matrix.append(np.array(tup[0][1]))
        matrix = np.array(matrix)
        print("dumping", self.source, "matrix")
        pkl.dump(matrix, open(self.dump_loc + \
                '_matrix.pkl', 'wb'))
        print(self.source, "matrix to .mat")
        source_expr = copy.deepcopy(self.source)
        source_expr = source_expr.replace('→', '->')
        # source_expr = source_expr.replace('', '->')
        scipy.io.savemat(self.dump_loc +\
                '_matrix.mat', {source_expr+'_vectors':matrix})

        cols = []
        # for tup in dataset.values():
        #     # if tup[1][-1][1] in [1,2,3,4,5,6,10,11,12,13,14,19,25,27,30]:
        #     #     # IDENTITY
        #     #     cols.append("b")
        #     # if tup[1][-1][1] in [7,8,9]:
        #     #     # TAUTOLOGY
        #     #     cols.append("g")
        #     # if tup[1][-1][1] in [16,17,21,22]:
        #     #     # ASSOCIATIVITY
        #     #     cols.append("r")
        #     # if tup[1][-1][1] in [15,20]:
        #         # COMMUTATIVITY
        #         # cols.append("c")
        #     # if tup[1][-1][1] in [23,26]:
        #     #     # LOGICAL EQUIVALENCE
        #     #     cols.append("m")
        #     # if tup[1][-1][1] in [18,24,28,29,0]:
        #     #     # DEMORGAN
        #     #     cols.append("k")
        #
        #     else:
        #         # rest
        #         cols.append("k")

        colors = ['b','g','r','c','m','k']
        for tup in dataset.values():
            for i in range(1,7):
                if len(tup[1]) == i:
                    cols.append(colors[i-1])





        scipy.io.savemat(self.dump_loc +\
                '_cols.mat', {source_expr+'_cols':cols})


    def bow_dataset_to_txt(self):
        data = pkl.load(open('../data/bigram_dataset.pkl', 'rb'))
        outfile = open('../data/text.txt', 'w', encoding='utf-8')
        for first_tup in data.values():
            for tup in first_tup[1]:
                outfile.write(str((tup[0].parse_tree(), tup[1], tup[2])))
            outfile.write("\n")
        outfile.close()

    def covariance(self):
        try:
            bows = pkl.load(open('../data/bigram_dataset.pkl', 'rb'))
        except FileNotFoundError:
            print("dataset not found, creating a new one")
            self.bow_representation_bigram_dataset()
            bows = pkl.load(open('../data/bigram_dataset.pkl', 'rb'))

        bow_vecs = np.array([np.array(elem[0][1]) for elem in bows.values()])

        print("calculating covariance")
        covariance_matrix = np.cov(bow_vecs)

        print("dumping")
        pkl.dump(covariance_matrix, open('../data/cov_matrix.pkl', 'wb'))


    def bow_collisions(self, percentage=100, trigrams=False):
        print("calculating collisions")
        if not trigrams:
            try:
                bows = pkl.load(open(self.dump_loc + '_bigram_dataset.pkl', 'rb'))
            except FileNotFoundError:
                print("dataset not found, creating a new one")
                self.bow_representation_bigram_dataset()
                bows = pkl.load(open(self.dump_loc + '_bigram_dataset.pkl', 'rb'))
        else:
            try:
                bows = pkl.load(open(self.dump_loc + '_trigram_dataset.pkl', 'rb'))
            except FileNotFoundError:
                print("dataset not found, creating a new one")
                self.bow_representation_trigram_dataset()
                bows = pkl.load(open(self.dump_loc + '_trigram_dataset.pkl', 'rb'))

        stats_file = open('../data/dataset_stats/stats.txt', 'w')

        bows_vals = list(bows.values())
        all_exprs = [tup[0] for tup in bows_vals]

        unique_exprs = {}
        for tup in all_exprs:
            if tup[0] not in unique_exprs:
                unique_exprs[tup[0]] = tup[1]
            else:
                if unique_exprs[tup[0]] != tup[1]:
                    print("MAJOR ERROR, SAME EXPR BIGRAMS NOT SAME")

        unique_exprs_vecs = [(V, E) for E, V in unique_exprs.items()]
        same_count = 0
        total_count = 0
        for i in range(len(unique_exprs_vecs)):
            for j in range(i+1, len(unique_exprs_vecs)):
                total_count += 1
                if unique_exprs_vecs[i][0] == unique_exprs_vecs[j][0]:
                    same_count += 1
                    #print("Collision:", unique_exprs_vecs[i][1], unique_exprs_vecs[j][1])
        print("Total collisions:", same_count)
        print("Total pairs:", total_count)
        print("Collision rate:", same_count/total_count)
        return

        op_depths = [len(tup[1]) for tup in bows_vals]
        op_depths_dict = {depth:[] for depth in range(min(op_depths), max(op_depths)+1)}

        for idx in range(len(bows_vals)):
            op_depths_dict[op_depths[idx]].append(bows_vals[idx][1])

        for op_depth, bigrams in op_depths_dict.items():

            duplicates = []
            freqs = []

            for idx1 in range(len(bigrams)-1):
                bigram = bigrams[idx1]
                if bigram not in duplicates:
                    freq_count = 1
                    found = False
                    for idx2 in range(idx1+1, len(bigrams)):
                        if bigram == bigrams[idx2]:
                            found = True
                            freq_count += 1
                    if found:
                        duplicates.append(bigram)
                        freqs.append(freq_count)

            sum_duplicates = sum([freq for freq in freqs])
            write_string = "operation depth " + str(op_depth) + "%duplicates: "\
                            + str(100*(sum_duplicates/len(bigrams)))
            stats_file.write(write_string)
            stats_file.write("\n")

        print("closing stats file")
        stats_file.close()






class BOW_exprs():

    def __init__(self, source=None, out=None):
        # variables = set('pqr')
        # variables.add("T")
        # variables.add("F")
        # operators = {'∧', '∨', '→', '~'}
        # parenthesis = {'(', ')'}
        # delimiters = {'STA', 'EOS'}
        # all_symbols = variables.union(operators).union(parenthesis).union(delimiters)
        #
        # # print(list(all_symbols))

        all_symbols = ['p','q','r']
        all_symbols.append('T')
        all_symbols.append('F')
        all_symbols.extend(['∧','∨','→','↔','~'])
        all_symbols.extend(['(',')'])
        all_symbols.extend(['STA','EOS'])



        self.symbols = {}
        for symbol in all_symbols:
            self.symbols[symbol] = len(self.symbols)
        self.bigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                self.bigrams[(symbol1, symbol2)] = len(self.bigrams)
        self.trigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                for symbol3 in all_symbols:
                    self.trigrams[(symbol1, symbol2, symbol3)] = len(self.trigrams)

            self.trigrams[('STA', symbol1, None)] = len(self.trigrams)
            self.trigrams[(None, symbol1, 'EOS')] = len(self.trigrams)


    def bow_unigram(self, expr):
        unigrams_count = defaultdict(int)
        for i in range(0, len(expr)):
            unigrams_count[expr[i]] += 1

        unigram_index_count = [0] * len(self.symbols)
        for expr in unigrams_count.keys():
            unigram_index_count[self.symbols[expr]] = unigrams_count[expr]
        return unigram_index_count


    def bow_bigram(self, expr):
        bigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 1):
            bigrams_count[(expr[i], expr[i + 1])] += 1
        bigrams_count[('STA', expr[0])] += 1
        bigrams_count[(expr[len(expr) - 1], 'EOS')] += 1

        bigram_index_count = [0] * len(self.bigrams)
        for (expr1, expr2) in bigrams_count.keys():
            bigram_index_count[self.bigrams[(expr1, expr2)]] = bigrams_count[(expr1, expr2)]
        return bigram_index_count

    def bow_trigram(self, expr):
        trigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 2):
            trigrams_count[(expr[i], expr[i + 1], expr[i + 2])] += 1
        try:
            trigrams_count[('STA', expr[0], expr[1])] += 1
        except IndexError:
            trigrams_count[('STA', expr[0], None)] += 1
        try:
            trigrams_count[(expr[len(expr) - 2], expr[len(expr) - 1], 'EOS')] += 1
        except IndexError:
            trigrams_count[(None, expr[len(expr) - 1], 'EOS')] += 1

        trigram_index_count = [0] * len(self.trigrams)
        for (expr1, expr2, expr3) in trigrams_count.keys():
            trigram_index_count[self.trigrams[(expr1, expr2, expr3)]] = trigrams_count[(expr1, expr2, expr3)]
        return trigram_index_count

    def bow_representation_expr(self, expr, trigrams=False):
        if not trigrams:
            return self.bow_unigram(expr) + self.bow_bigram(expr)
        else:
            return self.bow_unigram(expr) + self.bow_bigram(expr) + self.bow_trigram(expr)




if __name__ == '__main__':


    # starting_exprs = ['T', 'F', 'p', 'q', 'r', '~p', '~q', '~r', 'p→q']
    # starting_exprs = ['p→q']

    starting_exprs = ['T']

    for expr in starting_exprs:

        print("1")
        bow = Bag_of_words(expr, expr)
        print("2")
        print("here we go")
        bow.bow_representation_bigram_dataset()









































# comment
