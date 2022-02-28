from astar_search import astar_search
from astar_heuristics import *

from time import time, localtime, asctime
from multiprocessing import Process, Queue
import json


def search_func(start, goal, distance_func, heuristic, queue):  # could use decorator to time+queue it, but why bother?
    ts = time()
    res = astar_search(start, goal, distance_func, heuristic)
    te = time()
    queue.put((te - ts, res))


def test_search(distance_func, heuristic, max_timeout=5, question_file="questions.json"):
    results_file = f'astar_results_d_{distance_func.__name__}_h_{heuristic.__name__}.txt'
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
        search_process = Process(target=search_func, args=(start, goal, distance_func, heuristic, queue))
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


if __name__ == "__main__":
    test_search(levenshtein_distance, levenshtein_distance)
