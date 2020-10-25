"""

checking student response deterministically

"""
from create_expressions_mistakes import *


from contextlib import redirect_stdout
import io



def check_correct_operation(e1, e2, ops, num_ops=1):
    def convert_to_logic_symbols(expr):
        logic_symbols = ['∧', '∨', '→', '↔', '~']
        new_expr = expr.replace('^', '∧')
        new_expr = new_expr.replace('v', '∨')
        new_expr = new_expr.replace('<->', '↔')
        new_expr = new_expr.replace('->', '→')
        new_expr = new_expr.replace('x', 'p')
        return new_expr

    # Remove all spaces in expressions
    e1, e2 = e1.replace(' ', ''), e2.replace(' ', '')

    e1 = convert_to_logic_symbols(e1)
    e2 = convert_to_logic_symbols(e2)

    f = io.StringIO()
    with redirect_stdout(f):

        try:
            trainer = LogicTreeTrainer(e1, op_seq=ops, op_pairs=False, expand=None)
        except:
            raise ValueError('Could not parse', e1)

        try:
            sntx_check = LogicTreeTrainer(e2)
        except:
            raise ValueError('Could not parse', e2)


        trainer.increment_ops(num_ops)
        trees = trainer.get_trees()

        if len(trees) < 1500:
            tree_strs = []
            for t in trees:
                for loc_parses in t.deep_parse_tree():
                    tree_strs.append(loc_parses)
        else:
            tree_strs = [t.parse_tree() for t in trees]

        if e2 in tree_strs:
            del trainer
            del tree_strs
            del f
            return True
        else:
            del trainer
            del tree_strs
            del f
            return False

# example: should return True
# print(check_correct_operation('T', '~F', 1))
