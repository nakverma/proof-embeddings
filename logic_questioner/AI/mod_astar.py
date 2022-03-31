from expression_parser import *
from astar_heuristics import *
import json
from heapq import heappush, heappop

inf = float('inf')


def frontier_func(x):
    fr = get_frontier(x[0])
    return fr


def goal_func(x, target):
    return x[0] == target[0]


def rules_astar(start, goal, heuristic, rule_count, max_steps=1e3, *args, **kwargs):

    class SearchNode:
        def __init__(self, data, fscore=inf, gscore=inf):
            self.data = data
            self.fscore = fscore
            self.gscore = gscore
            self.out_of_openset = True
            self.completed = False
            self.prev = None

        def __lt__(self, other):
            return self.fscore < other.fscore

    class NodeDict(dict):  # can't replace with defaultdict because it doesn't accept args in lambda :(

        def __missing__(self, key):
            value = SearchNode(key)
            self.__setitem__(key, value)
            return value

    start = (start, "Start")
    goal = (goal, None)

    if goal_func(start, goal):
        return [start], rule_count

    search_dict = NodeDict()
    start_node = search_dict[start] = SearchNode(start, fscore=heuristic(start, goal, *args, **kwargs), gscore=.0)
    open_set = []
    heappush(open_set, start_node)
    step_count = 0

    while open_set:
        current_node = heappop(open_set)
        if goal_func(current_node.data, goal):
            rev_sol = []
            while current_node is not None:
                rev_sol.append(current_node.data)
                current_node = current_node.prev
            return reversed(rev_sol), rule_count

        current_node.out_of_openset = True
        current_node.completed = True

        for neighbor in map(lambda n: search_dict[n], frontier_func(current_node.data)):
            if neighbor.completed:
                continue
            tentative_gscore = current_node.gscore + heuristic(current_node.data, neighbor.data)
            if tentative_gscore < neighbor.gscore:
                neighbor.prev = current_node
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + heuristic(neighbor.data, goal)
                if neighbor.out_of_openset:
                    neighbor.out_of_openset = False
                    heappush(open_set, neighbor)
                else:
                    open_set.remove(neighbor)
                    heappush(open_set, neighbor)
                rule_count[neighbor.data[1]] += 1

            step_count += 1
            if step_count > max_steps:
                #print("Step limit")
                return None, rule_count

    return None, rule_count


def astar_search(start, goal, distance_func, heuristic, *args, **kwargs):

    class SearchNode:
        def __init__(self, data, fscore=inf, gscore=inf):
            self.data = data
            self.fscore = fscore
            self.gscore = gscore
            self.out_of_openset = True
            self.completed = False
            self.prev = None

        def __lt__(self, other):
            return self.fscore < other.fscore

    class NodeDict(dict):  # can't replace with defaultdict because it doesn't accept args in lambda :(

        def __missing__(self, key):
            value = SearchNode(key)
            self.__setitem__(key, value)
            return value

    start = (start, "Start")
    goal = (goal, None)

    if goal_func(start, goal):
        return [start]

    search_dict = NodeDict()
    start_node = search_dict[start] = SearchNode(start, fscore=heuristic(start, goal, *args, **kwargs), gscore=.0)
    open_set = []
    heappush(open_set, start_node)

    while open_set:
        current_node = heappop(open_set)
        if goal_func(current_node.data, goal):
            rev_sol = []
            while current_node is not None:
                rev_sol.append(current_node.data)
                current_node = current_node.prev
            return reversed(rev_sol)

        current_node.out_of_openset = True
        current_node.completed = True

        for neighbor in map(lambda n: search_dict[n], frontier_func(current_node.data)):
            if neighbor.completed:
                continue
            tentative_gscore = current_node.gscore + distance_func(current_node.data, neighbor.data)
            if tentative_gscore < neighbor.gscore:
                neighbor.prev = current_node
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
    for q in questions[:1]:
        gp, rc = rules_astar(q['premise'], q['target'], levenshtein_distance, ["Start"] + list(operation_names.keys()))
        print(list(gp))
        print(rc)
