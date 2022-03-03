import random
from expression_parser import *
from logic_rule_transforms import *


reverse_rule_pairs = {
    # identity: reverse_identity,
    distributivity: reverse_distributivity,
    commutativity: commutativity,
    impl_to_disj: disj_to_impl,
    dblimpl_to_impl: impl_to_dblimpl,
    demorgan: reverse_demorgan,
    absorption: reverse_absorption,
    # double_negate: simplify_multiple_negation,
    associativity_LR: associativity_LR,
    associativity_expand: reverse_associativity_expand,
    idempotence: reverse_idempotence
}


class QuestionGenerator:

    def __init__(self, variables=("p", "q", "r", "s", "T", "F")):
        self.variables = variables
        self.reverse = reverse_rule_pairs
        self.reverse.update({v: k for k, v in reverse_rule_pairs.items()})
        self.allowed_ops = {k: [v for v in allowed_operations[k] if v in self.reverse] for k in allowed_operations}

    def _get_next_step(self, prev_expr, prev_rule):
        frontier = get_frontier(prev_expr, allowed_ops=self.allowed_ops)
        possible_next_rules = list({exp[1] for exp in frontier} - {prev_rule})
        if len(possible_next_rules) == 0:
            return None, None
        next_rule = random.choice(possible_next_rules)
        next_step = random.choice(list(filter(lambda exp: exp[1] == next_rule, frontier)))
        return next_step

    def generate(self, seed, max_depth=5, difficulty=None):
        solution = []
        if difficulty is None:
            difficulty = "mild" if max_depth < 5 else "medium"
            difficulty = difficulty if max_depth < 8 else "spicy"

        prev_expr, prev_rule = seed, None
        for d in range(max_depth):
            next_expr, next_rule = self._get_next_step(prev_expr, prev_rule)
            if next_expr is None or next_rule is None:
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
    q1 = qg.generate("p", max_depth=10)
    for k, v in q1.items():
        print("{}: {}".format(k, v))
