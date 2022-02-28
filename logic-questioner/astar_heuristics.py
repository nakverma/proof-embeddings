from collections import defaultdict
from random import random

from Levenshtein import distance
import gensim.downloader

# all heuristics expect Tuple<expr: str, law:str> as inputs. Change typing to a StepNode : {expr: str, law: str} object


def random_weight(n1, n2):
    return random() * 10


def levenshtein_distance(n1, n2):
    return distance(n1[0], n2[0]) / 10


def unitary_distance(n1, n2):
    return 1


def big_change_favored_weight(n1, n2):  # implemented in the old code
    scores = defaultdict(int, {
        "Double Negation": 6,
        "Implication as Disjunction": 2,
        "Iff as Implication": 1,
        "Idempotence": 7,
        "Identity": 7,
        "Domination": 7,
        "Commutativity": 8,
        "Associativity": 8,
        "Negation": 6,
        "Absorption": 5,
        "Distributivity": 4,
        "De Morgan's Law": 3
    })
    return scores[n2[1]]


def small_change_favored_weight(n1, n2):  # implemented in the old code
    return 9 - big_change_favored_weight(n1, n2)


def combo_weight(n1, n2, heuristic=levenshtein_distance, cutoff=4):  # implemented in old code
    if heuristic(n1, n2) > cutoff:  # apparently, optimal cutoff weight not determined
        return big_change_favored_weight(n1, n2)
    return small_change_favored_weight(n1, n2)




if __name__ == "__main__":
    n1, n2 = ('p->q', None), ('~pvq', None)
    print(levenshtein_distance(n1, n2))
    print(unitary_distance(n1, n2))

"""

def h_abs_difference_left(start, next, ans): #was h6, fixed per last week's meeting
    return abs(abs(len(start)-len(ans)) - abs(len(next)-len(ans)))

#_______edit distance h functions

def h_edit_dist_to_end(start, next, ans):
    return distance(next, ans)

def h_edit_proportion_change(start, next, ans):
    return distance(next, ans) / distance(start, ans)

def h_edit_w1_difference_left(start, next, ans):
    return big_change_favored_weight(start, next) * distance(next, ans) / distance(start, ans)

def h_edit_abs_difference_left(start, next, ans): #was h6, fixed per last week's meeting
    return abs(distance(start, ans) - distance(next, ans))

#______
# #g's were not dependant on disntance, so we don't need to add new ones for edit distance
def g_big(n1, n2):
    return big_change_favored_weight(n1, n2)

def g_small(n1, n2):
    return small_change_favored_weight(n1, n2)

def g_combo(n1, n2,  ans, h):
    CUTOFF = 4 # Needs to be tested and changed
    if h(n1, n2, ans) > CUTOFF:
        return big_change_favored_weight(n1, n2)
    else:
        return small_change_favored_weight(n1, n2)

"""