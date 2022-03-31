import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np
from astar_heuristics import *

from time import time, localtime, asctime
from multiprocessing import Process, cpu_count, Queue, Pool
import json
import os
from mod_astar import rules_astar
from test_astar_search import get_score


def softmax(x):
    y = np.exp(x-np.max(x))
    return y / np.sum(np.exp(x))


def search_func_rules(start, goal, heuristic, rule_count, queue, *args, **kwargs):  # could use decorator to time+queue it, but why bother?
    # queue.cancel_join_thread()
    res, rc = rules_astar(start, goal, heuristic, rule_count, *args, **kwargs)
    #print(res, rc, sep="\n")
    queue.put((res, rc))


def get_rule_score(questions, heuristic, rules, max_timeout=2):
    queue = Queue()
    evals = []
    rcs = []
    solved = 0
    for i, q in enumerate(questions):
        rule_count = {r: 0 for r in rules}
        start = q['premise']
        goal = q['target']
        search_process = Process(target=search_func_rules, args=(start, goal, heuristic, rule_count, queue))
        search_process.start()
        search_process.join(timeout=max_timeout)
        res, rc = queue.get(block=False) if search_process.exitcode is not None else (None, rule_count)
        #print(rc)
        if res is not None:
            solved += 1
            evals.append(1)
        else:
            evals.append(0)
        rcs.append(rc)
        #search_process.terminate()
        print("+" if res is not None else "-", end="")
    print(f"\nSolved {solved}/{len(questions)}")
    return evals, rcs


def learn_rule_weights(data_file="questions.json", lr=1, epochs=10):  # sorta like REINFORCE
    with open(data_file) as f:
        questions = json.load(f)['questions']

    ops = ["Start"] + list(operation_names.keys())
    wrd = WeightedRuleDist(ops)

    for ep in range(epochs):
        print(wrd.weights)
        evals, rcs = get_rule_score(questions, wrd.rule_dist, ops)
        for i, e in enumerate(evals):
            imp = np.array(list(rcs[i].values()))
            imp = e * imp / (np.max(imp)+1)
            wrd.weights = list(np.array(wrd.weights)+lr*imp)
            #print(imp, wrd.weights)

    with open("rule_weights.txt", "w") as f:
        for i, op in enumerate(wrd.ops):
            f.write(f'{op}: {wrd.weights[i]}\n')
    return wrd


def trial(rand_grid, meta_heur):
    best_score, best_wt = 0, rand_grid[0]
    for g in rand_grid:
        meta_heur.set_weights(g)
        score, _, _ = get_score(meta_heur.meta_dist, max_timeout=3)
        print(f"Weights: {g}\n")
        if score > best_score:
            best_wt = g
            best_score = score
    return best_wt, best_score


def search_meta_weights(heuristics, rand_grid=None, grid_size=10, low=-10, high=10, out_file="meta_weights.txt", parallel=True):
    mh = MetaHeuristic(heuristics)
    if rand_grid is None:
        rand_grid = [np.random.uniform(low=low, high=high, size=len(heuristics)) for _ in range(grid_size)]

    if parallel:
        pool = ThreadPool()
        result = pool.apply_async(trial, args=(rand_grid, mh))
        best_wt, best_score = result.get(timeout=1200)
    else:
        best_wt, best_score = trial(rand_grid, mh)

    with open(out_file, "w") as f:
        for i, h in enumerate(heuristics):
            f.write(f'{h.__name__}: {best_wt[i]}\n')

    print(f"Best weight: {best_wt}\n Best score: {best_score}")
    return best_wt, best_score


if __name__ == "__main__":
    # wrd = learn_rule_weights(epochs=10)
    heuristics = [levenshtein_distance, unitary_distance] + RuleDists().all_dists #, frontier_size, rare_rule_first]
    grid = [np.concatenate((
        np.random.uniform(low=0, high=10, size=1),
        np.random.uniform(low=-10, high=10, size=len(heuristics)-1)
    )) for _ in range(10)]
    bw, bs = search_meta_weights(heuristics, rand_grid=grid, grid_size=1, out_file="meta_rule_lev_wts.txt", parallel=False)

'''
[-4.72210198  6.53848175 -6.19700752  4.42757432 -2.366619   -3.27885157
 -1.46090436 -9.45742536 -9.76751637 -4.56582074 -4.50066776 -3.87381573
  3.98213994 -5.23208609]
'''