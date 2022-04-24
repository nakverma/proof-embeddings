from astar_heuristics import *
import json
from heapq import heappush, heappop
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import expression_parser as ep
import logic_rule_transforms as lrt

inf = float('inf')


def frontier_func(x):
    fr = ep.get_frontier(x[0], include_paren=False, allowed_ops=lrt.search_operations)
    return fr


def goal_func(x, target):
    return x[0] == target[0]


def astar_search(start, goal, distance_func, heuristic, max_depth=None, *args, **kwargs):

    class SearchNode:
        def __init__(self, data, fscore=inf, gscore=inf):
            self.data = data
            self.fscore = fscore
            self.gscore = gscore
            self.out_of_openset = True
            self.completed = False
            self.prev = None
            self.depth = None

        def __lt__(self, other):
            return self.fscore < other.fscore

    class NodeDict(dict):  # can't replace with defaultdict because it doesn't accept args in lambda :(

        def __missing__(self, key):
            value = SearchNode(key)
            self.__setitem__(key, value)
            return value

    start = (start, "Start")
    goal = ep.ExpressionParser().parse(goal)
    goal = ep.TreeToString().transform(goal) if type(goal) == Tree else goal.value
    goal = (goal, None)

    if goal_func(start, goal):
        return [start]

    search_dict = NodeDict()
    start_node = search_dict[start] = SearchNode(start, fscore=heuristic(start, goal, *args, **kwargs), gscore=.0)
    start_node.depth = 0
    open_set = []
    heappush(open_set, start_node)

    while open_set:
        current_node = heappop(open_set)
        current_node.out_of_openset = True
        current_node.completed = True
        if max_depth is not None and current_node.depth > max_depth:
            continue

        if goal_func(current_node.data, goal):
            rev_sol = []
            while current_node is not None:
                rev_sol.append(current_node.data)
                current_node = current_node.prev
            return reversed(rev_sol)

        for neighbor in map(lambda n: search_dict[n], frontier_func(current_node.data)):
            if neighbor.completed:
                continue
            tentative_gscore = current_node.gscore + distance_func(current_node.data, neighbor.data)
            if tentative_gscore < neighbor.gscore:
                neighbor.prev = current_node
                neighbor.depth = current_node.depth + 1
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + heuristic(neighbor.data, goal)
                if neighbor.out_of_openset:
                    neighbor.out_of_openset = False
                    heappush(open_set, neighbor)
                else:
                    open_set.remove(neighbor)
                    heappush(open_set, neighbor)

    return None


if __name__ == "__main__":
    with open('../questions.json') as f:
        questions = json.load(f)['questions']
    for q in questions[4:5]:
        q["premise"] = "(qvp)^(qv~q)"
        gp = astar_search(q['premise'], q['target'], levenshtein_distance, levenshtein_distance, max_depth=None)
        print(list(gp))
