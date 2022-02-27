from astar_search import *
from expression_parser import get_frontier
from astar_heuristics import *

from time import time, localtime, asctime
from multiprocessing import Process, Queue


def search_func(start, goal, goal_heuristic, adjacent_heuristic, queue):
    ts = time()
    res = get_goal_path(start, goal, goal_heuristic, adjacent_heuristic)
    te = time()
    queue.put((te-ts, res[0]))


def test_search(goal_heuristic, adjacent_heuristic, max_timeout=5,
                results_file="astar_test_results.txt", question_file="questions.json"):
    with open(question_file, "r") as qf:
        questions = json.load(qf)['questions']
    queue = Queue()
    with open(results_file, "w") as rf:
        est = asctime(localtime())
        rf.write("Search Test: {}\n\n".format(est))
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        search_process = Process(target=search_func, args=(start, goal, goal_heuristic, adjacent_heuristic, queue))
        search_process.start()
        search_process.join(timeout=max_timeout)
        with open(results_file, "a") as rf:
            info_str = "{}. Premise: {}, Target: {}. ".format(i, start, goal)
            if search_process.exitcode is None:
                rf.write(info_str + "Timeout occurred.\n")
            else:
                res = queue.get()
                rf.write(info_str + "Time: {}, Solution: {}\n".format(res[0], list(res[1])))
        search_process.terminate()


if __name__ == "__main__":
    test_search(levenshtein_distance, unitary_distance)
