import random
from expression_parser import *
from logic_rule_transforms import *


reverse_rule_pairs = {
    #identity: reverse_identity,
    distributivity: reverse_distributivity,
    commutativity: commutativity,
    impl_to_disj: disj_to_impl,
    dblimpl_to_impl: impl_to_dblimpl,
    demorgan: reverse_demorgan,
    absorption: reverse_absorption,
    double_negate: simplify_multiple_negation
}

class QuestionGenerator:

    def __init__(self, variables=("p", "q", "r", "s", "T", "F")):
        self.variables = variables
        self.reverse = reverse_rule_pairs
        self.reverse.update({v: k for k, v in reverse_rule_pairs.items()})

    def _get_next_step(self, prev_expr, prev_rule):
        frontier = get_frontier(prev_expr)
        possible_next_rules = list({exp[1] for exp in frontier} - {prev_rule})
        next_rule = random.choice(possible_next_rules)
        next_step = random.choice(list(filter(lambda exp: exp[1] == next_rule, frontier)))
        return next_step

    def generate(self, seed, max_depth=5, difficulty="mild"):
        solution = []
        prev_expr, prev_rule = seed, None
        for d in range(max_depth):
            try:
                next_expr, next_rule = self._get_next_step(prev_expr, prev_rule)
            except Exception as e:
                print(e)
                break
            solution.append((prev_expr, next_rule))
            prev_expr, prev_rule = next_expr, next_rule
        solution.append((prev_expr, "Start"))
        solution.reverse()
        return {
            "premise": solution[0][0],
            "target": seed,
            "solution": solution,
            "difficulty": difficulty
        }


if __name__ == "__main__":
    qg = QuestionGenerator()
    q1 = qg.generate("p")
    for k, v in q1.items():
        print("{}: {}".format(k, v))
