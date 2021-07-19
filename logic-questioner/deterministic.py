"""
checking student response deterministically
"""
from create_expressions_mistakes import *


from contextlib import redirect_stdout
import io



def check_correct_operation(e1, e2, ops, num_ops=1):
    print("called c_c_o()") #this does get called, but doesn't make LogicTreeTrainer object??
    if e1 == e2: #if the student entered the same thing as the previous step
        print("returned false and continued")
        return False
    ops = ops*num_ops
    def convert_to_logic_symbols(expr):
        logic_symbols = ['∧', '∨', '→', '↔', '~']
        new_expr = expr.replace('^', '∧')
        new_expr = new_expr.replace('v', '∨')
        new_expr = new_expr.replace('<->', '↔')
        new_expr = new_expr.replace('->', '→')
        new_expr = new_expr.replace('x', 'p')
        new_expr = new_expr.replace('¬', '~')
        return new_expr

    # Remove all spaces in expressions
    e1, e2 = e1.replace(' ', ''), e2.replace(' ', '')

    e1 = convert_to_logic_symbols(e1)
    e2 = convert_to_logic_symbols(e2)

    f = io.StringIO()
    print("this is f")
    print(f)
    #with statement: https://www.geeksforgeeks.org/with-statement-in-python/
    #redirect_stdout: https://docs.python.org/3/library/contextlib.html#:~:text=in%20version%203.4.-,contextlib,-.redirect_stdout(new_target)%C2%B6 
    #combination means that output in this block is directed to f instead of sys.stdout

    #!!!!CAREFUL WHEN CLEANING!!!!!!!
    #for dubugging, commented out with statement and un-indented until comment that says "until here"
    #with redirect_stdout(f): 
    print("got inside with statement")
    try:
        print("LogicTreeTrainer 1")
        trainer = LogicTreeTrainer(e1, expand=None, op_seq=ops, op_pairs=False)
    except:
        raise ValueError('Could not parse', e1)

    try:
        print("LogicTreeTrainer 2")
        sntx_check = LogicTreeTrainer(e2)
    except:
        raise ValueError('Could not parse', e2)


    trainer.increment_ops(num_ops)
    trees = trainer.get_trees()

    # if len(trees) < 1000 and ops[0] != 'Commutativity':
    #     tree_strs = [t.parse_tree() for t in trees]
    #     for t in trees:
    #         for loc_parses in t.deep_parse_tree():
    #             tree_strs.append(loc_parses)
    # else:
        # tree_strs = [t.parse_tree() for t in trees]

    tree_strs = [t.parse_tree() for t in trees]
    print(tree_strs)

    if e2 in tree_strs: #compare what the user entered to the possible next steps
        del trainer #clean up
        del tree_strs
        del f
        return True #the user was right
    else:
        del trainer #clean up
        del tree_strs
        del f
        return False #the user was wrong
    #until here

if __name__ == "__main__":
    print(check_correct_operation('T', '~F', ['LITERAL NEGATION'], 1))
