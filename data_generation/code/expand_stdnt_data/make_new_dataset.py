"""
doin it right

checking student response deterministically

"""
import sys
sys.path.append('../')
from create_expressions_mistakes import *


import pandas as pd
import numpy as np
import ast
import pickle as pkl
from collections import defaultdict
import random
from random import shuffle



class BOW_exprs():

    def __init__(self, source=None, out=None):

        all_symbols = ['p', 'q', 'r']
        all_symbols.append('T')
        all_symbols.append('F')
        all_symbols.extend(['∧', '∨', '→', '~'])
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
bow = BOW_exprs()



def convert_to_logic_symbols(expr):
    logic_symbols = ['∧', '∨', '→', '~']
    new_expr = expr.replace('^', '∧')
    new_expr = new_expr.replace('v', '∨')
    new_expr = new_expr.replace('->', '→')
    return new_expr


df = pd.read_csv('../../data/student_responses/student_answers_complex.csv')


correct_responses = []
for i in range(len(df)):
    if int(df['Answer Correct/Incorrect'][i]):
        rspons = list(map(convert_to_logic_symbols, \
                    ast.literal_eval(df['Student Response'][i])))

        rspons_id = str(df['QuestionID'][i])+'_'+str(df['AnswerID'][i])

        correct_responses.append((rspons, rspons_id))



st = []
st.append(convert_to_logic_symbols("(pvq)v(pv~q)"))
st.append(convert_to_logic_symbols("((p->r)^(q->r)^(pvq))->r"))
st.append(convert_to_logic_symbols("(~(~x))<->x"))
st.append(convert_to_logic_symbols("((p->q)^(q->r))->(p->r)"))

correct_responses = [r for r in correct_responses if r[0][0] in st]

all_symbols = list(bow.symbols.keys())

cr = []
for (r,id) in correct_responses:
    good = True
    for s in r:
        for c in s:
            if c not in all_symbols:
                good=False
                break
                break
    if good:
        cr.append((r,id))
correct_responses = cr.copy()


data_lst = []

"""
Columns:
StudentID   |   Skipped_Steps   |   Ammended_Steps  |   Ammended_Step_Types
"""


test = [correct_responses[0]]

print("Ammending", len(correct_responses), "responses")
for indxx in range(len(correct_responses)):
    print("response #", indxx)
    response, id = correct_responses[indxx]

    ammended = []

    ammendments = {}

    for idx in range(len(response)-1):
        e1 = response[idx]
        e2 = response[idx+1]

        parse_err = False
        try:
            trainer = LogicTreeTrainer(e1,expand=None, op_pairs=False)
            sntx_check = LogicTreeTrainer(e2)
        except:
            parse_err = True
            break

        num_ops = 0
        found_e2 = False
        while not found_e2 and num_ops < 3:
            num_ops += 1
            trainer.increment_ops()
            exps_lst = []
            ops_lst = []

            seqs = trainer.get_sequences()
            if len(seqs) < 1500:
                for seq in seqs:
                    if e2 in seq[-1][0].deep_parse_tree():
                        found_e2 = True

                        exps = [tr.parse_tree() for (tr,op) in seq]
                        ops = [op for (tr,op) in seq]

                        if e2 != seq[-1][0].parse_tree():
                            exps.append(e2)
                            ops.append('Parentheses(deepparse)')

                        exps_lst.append(exps)
                        ops_lst.append(ops)
            else:
                for seq in seqs:
                    if e2 == seq[-1][0].parse_tree():
                        found_e2 = True
                        exps_lst.append([tr.parse_tree() for (tr,op) in seq])
                        ops_lst.append([op for (tr,op) in seq])

        ammendments[idx] = (exps_lst, ops_lst)


    if parse_err or not found_e2:
        continue

    tmp_steps = []
    tmp_ops = []

    new_steps = []
    new_ops = []

    for idx in range(len(response)-1):
        steps, ops = ammendments[idx]

        if not new_steps:
            new_steps = steps
            new_ops = ops
        else:
            tmp_steps = new_steps
            tmp_ops = new_ops

            new_steps = []
            new_ops = []

            for i in range(len(tmp_steps)):
                for j in range(len(steps)):
                    tstps = tmp_steps[i].copy()
                    tops = tmp_ops[i].copy()

                    nstps = steps[j][1:]
                    nops = ops[j][1:]

                    tstps.extend(nstps)
                    tops.extend(nops)
                    new_steps.append(tstps)
                    new_ops.append(tops)

    assert len(new_steps) == len(new_ops)

    for i in range(len(new_steps)):
        data_lst.append((id, response, new_steps[i], new_ops[i]))

    pkl.dump(data_lst, open('./data.pkl', 'wb'))








# comment
