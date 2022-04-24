from astar_search import astar_search
from astar_heuristics import *
from neural_embedding_heuristic import SimNet, NeuralEmbeddingHeuristic

from time import time, localtime, asctime
from multiprocessing import Process, Queue
import json
import os


def search_func(start, goal, distance_func, heuristic, max_depth, queue, *args, **kwargs):  # could use decorator to time+queue it, but why bother?
    ts = time()
    res = astar_search(start, goal, distance_func, heuristic, max_depth, *args, **kwargs)
    te = time()
    queue.put((te - ts, res))


def simple_search(start, goal, heuristic, *args, **kwargs):  # could use decorator to time+queue it, but why bother?
    res = astar_search(start, goal, heuristic, heuristic, None, *args, **kwargs)


def test_search(distance_func, heuristic, max_timeout=5, max_depth=None, question_file="../questions.json"):
    results_file = os.path.join(
        'heuristics_and_results', 'heuristic', f'astar_results_d_{distance_func.__name__}_h_{heuristic.__name__}.txt'
    )
    with open(question_file, "r") as qf:
        questions = json.load(qf)['questions']
    queue = Queue()
    with open(results_file, "w") as rf:
        est = asctime(localtime())
        rf.write(
            "Search Test: {}\nDistance: {}, Heuristic: {}, Max Timeout: {} seconds\n\n"
            .format(est, distance_func.__name__, heuristic.__name__, max_timeout)
        )
    num_solved = 0
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        search_process = Process(target=search_func, args=(start, goal, distance_func, heuristic, max_depth, queue))
        search_process.start()
        search_process.join(timeout=max_timeout)
        with open(results_file, "a") as rf:
            info_str = "{}. Premise: {}, Target: {}. ".format(i + 1, start, goal)
            if search_process.exitcode is None:
                rf.write(info_str + "Timeout occurred.\n")
                print(info_str + "Timeout occurred.")
            else:
                res = queue.get()
                dur, sol = res[0], list(res[1])
                rf.write(info_str + "Solved in {:.4f} seconds. Solution: {}\n".format(dur, sol))
                print(info_str + "Solved in {:.4f} seconds. Solution: {}".format(dur, sol))
                num_solved += 1
        search_process.terminate()
    with open(results_file, "a") as rf:
        rf.write("\nSolved {}/{} questions.".format(num_solved, len(questions)))
        print("\nSolved {}/{} questions.".format(num_solved, len(questions)))


def get_score(heuristic, max_timeout=1, question_file="questions.json"):
    with open(question_file, "r") as qf:
        questions = json.load(qf)['questions']
    queue = Queue()
    num_correct = 0
    evals = []
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        search_process = Process(target=search_func, args=(start, goal, heuristic, heuristic, None, queue))
        search_process.start()
        search_process.join(timeout=max_timeout)
        if search_process.exitcode is not None:
            num_correct += 1
            evals.append(1)
        else:
            evals.append(-1)
        search_process.terminate()
        print(".", end="")
    print()
    print(f"Solved {num_correct}/{len(questions)} in {max_timeout} seconds each.")
    return num_correct, len(questions), evals


def get_score_simple(heuristic, questions, max_timeout=1):
    num_correct = 0
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        search_process = Process(target=simple_search, args=(start, goal, heuristic))
        search_process.start()
        search_process.join(timeout=max_timeout)
        if search_process.exitcode is not None:
            num_correct += 1
        search_process.terminate()
    return num_correct


def test_gene_heuristic(weight_file, max_timeout=5, max_depth=None, question_file="../questions.json"):
    gh = GeneHeuristic(None, None)
    gh.load(os.path.join("heuristics_and_results", "genetic_weights", weight_file))

    depth_str = '' if max_depth is None else f'_depth_{max_depth}'
    results_file = os.path.join(
        'heuristics_and_results', 'genetic', weight_file[:-4] + depth_str + '_score.txt'
    )
    with open(question_file, "r") as qf:
        questions = json.load(qf)['questions']
    queue = Queue()
    with open(results_file, "w+") as rf:
        est = asctime(localtime())
        rf.write(f"Search Test: {est}\nWeight File: {weight_file}, Max Timeout: {max_timeout} seconds\n"
                 f"Params: {gh.params}\n\n")
    num_solved = 0
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        search_process = Process(target=search_func, args=(start, goal, gh.gene_meta_dist, gh.gene_meta_dist, max_depth, queue))
        search_process.start()
        search_process.join(timeout=max_timeout)
        with open(results_file, "a") as rf:
            info_str = "{}. Premise: {}, Target: {}. ".format(i + 1, start, goal)
            if search_process.exitcode is None:
                rf.write(info_str + "Timeout occurred.\n")
                print(info_str + "Timeout occurred.")
            else:
                res = queue.get()
                dur, sol = res[0], list(res[1])
                rf.write(info_str + "Solved in {:.4f} seconds. Solution: {}\n".format(dur, sol))
                print(info_str + "Solved in {:.4f} seconds. Solution: {}".format(dur, sol))
                num_solved += 1
        search_process.terminate()
    with open(results_file, "a") as rf:
        rf.write("\nSolved {}/{} questions.".format(num_solved, len(questions)))
        print("\nSolved {}/{} questions.".format(num_solved, len(questions)))


if __name__ == "__main__":
    # test_search(random_weight, random_weight)
    # test_search(random_weight, levenshtein_distance)
    # test_search(levenshtein_distance, levenshtein_distance)
    # test_search(big_change_favored_weight, levenshtein_distance)
    # test_search(small_change_favored_weight, levenshtein_distance)
    # test_search(combo_weight, levenshtein_distance)
    # test_search(smallest_frontier, smallest_frontier)
    # test_search(len_distance, len_distance)
    # wrd = WeightedRuleDist(['Start'] + list(operation_names.keys()))
    # wrd.init_weights("test_weights.txt")
    # test_search(wrd.rule_dist, wrd.rule_dist)

    # print(*get_score(levenshtein_distance))

    # mh = MetaHeuristic()
    # mh.init_state("meta_rule_lev_wts.txt")
    # test_search(mh.meta_dist, mh.meta_dist)

    # neh = NeuralEmbeddingHeuristic(os.path.join("models", "dist_model_parameters.pt"))
    # test_search(neh.embedding_dist, neh.embedding_dist)

    test_gene_heuristic("gene_weights_tough.txt", max_timeout=5)
