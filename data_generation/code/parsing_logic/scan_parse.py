from sly import Lexer
from sly import Parser

import re
import sys



class LogicLexer(Lexer):
    # Set of token names.   This is always required

    tokens = {P,Q,R,TRUE,FALSE,AND,OR,NOT,IMP,D_IMP,R_PAREN,L_PAREN}

    P = 'p'
    Q = 'q'
    R = 'r'
    TRUE = 'T'
    FALSE = 'F'
    AND = '\\^|∧'
    OR = 'v|∨'
    NOT = '~'
    D_IMP = '<->|↔'
    IMP = '->|→'
    R_PAREN = '\\('
    L_PAREN = '\\)'



"""

P = 'p'
Q = 'q'
R = 'r'
TRUE = 'T'
FALSE = 'F'
AND = '\\^|∧'
OR = 'v|∨'
NOT = '~'
IMP = '->|→'
D_IMP = '<->|↔'
R_PAREN = '\\('
L_PAREN = '\\)'

"""




class Expr:
    pass

class OrOp(Expr):

    def __str__(self):
        ops_str = ""
        for o in self.operands:
            ops_str += str(o) + " "
        return "OR " + ops_str

    def __init__(self, op, left, right):

        # if left.op == op:
        if isinstance(left, OrOp):
             left_ops = left.operands
        else:
             left_ops = (left,)
        # if right.op == op:
        if isinstance(right, OrOp):
             right_ops = right.operands
        else:
             right_ops = (right,)
        self.op = op
        self.operands = left_ops + right_ops

class AndOp(Expr):

    def __str__(self):
        ops_str = ""
        for o in self.operands:
            ops_str += str(o) + " "
        return "AND " + ops_str

    def __init__(self, op, left, right):

        # if left.op == op:
        if isinstance(left, AndOp):
             left_ops = left.operands
        else:
             left_ops = (left,)
        # if right.op == op:
        if isinstance(right, AndOp):
             right_ops = right.operands
        else:
             right_ops = (right,)
        self.op = op
        self.operands = left_ops + right_ops

class ImpOp(Expr):

    def __str__(self):
        return "-> "+str(self.left)+" "+str(self.right)

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class DblimpOp(Expr):

    def __str__(self):
        return "<-> "+str(self.left)+" "+str(self.right)

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class NotOp(Expr):

    def __str__(self):
        return "~ "+str(self.operand)

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

class Var(Expr):

    def __str__(self):
        return str(self.value)

    def __init__(self, value):
        self.value = value

class Lit(Expr):

    def __str__(self):
        return str(self.value)

    def __init__(self, value):
        self.value = value

class Pnth(Expr):

    def __str__(self):
        pass

    def __init__(self, exp):
        self.exp = exp




class LogicParser(Parser):
    # Get the token list from the lexer (required)
    tokens = LogicLexer.tokens


    @_('dblimparg')
    def expr(self, p):
        return p[0]

    @_('dblimparg D_IMP imparg')
    def dblimparg(self, p):
        return DblimpOp(p[1], p[0], p[2])

    @_('imparg')
    def dblimparg(self, p):
        return p[0]

    @_('imparg IMP orarg')
    def imparg(self, p):
        return ImpOp(p[1], p[0], p[2])

    @_('orarg')
    def imparg(self, p):
        return p[0]

    @_('orarg OR andarg')
    def orarg(self, p):
        return OrOp(p[1], p[0], p[2])

    @_('andarg')
    def orarg(self, p):
        return p[0]

    @_('andarg AND notarg')
    def andarg(self, p):
        return AndOp(p[1], p[0], p[2])

    @_('notarg')
    def andarg(self, p):
        return p[0]

    @_('NOT notarg')
    def notarg(self, p):
        return NotOp(p[0], p[1])

    # @_('NOT term')
    # def notarg(self, p):
    #     return NotOp(p[0], p[1])

    @_('term')
    def notarg(self, p):
        return p[0]

    @_('R_PAREN expr L_PAREN')
    def term(self, p):
        return Pnth(p[1])

    @_('P')
    def term(self, p):
        return Var(p[0])

    @_('Q')
    def term(self, p):
        return Var(p[0])

    @_('R')
    def term(self, p):
        return Var(p[0])

    @_('TRUE')
    def term(self, p):
        return Lit(p[0])

    @_('FALSE')
    def term(self, p):
        return Lit(p[0])



def checkSyntax(log_expr):
    lexer = LogicLexer()
    parser = LogicParser()
    try:
        result = parser.parse(lexer.tokenize(log_expr))
        if result:
            return True
        return False
    except:
        return False






if __name__ == '__main__':
    lexer = LogicLexer()
    parser = LogicParser()

    text = "pvq"
    result = parser.parse(lexer.tokenize(text))
    print(result)

    # sys.exit()
    while True:
        try:
            text = input('calc > ')
            result = parser.parse(lexer.tokenize(text))
            print(result)
        except EOFError:
            break

























# class LogicParser(Parser):
#     # Get the token list from the lexer (required)
#     tokens = LogicLexer.tokens
#
#     @_('orarg')
#     def expr(self, p):
#         return p.orarg
#
#     @_('orarg OR andarg')
#     def orarg(self, p):
#         return p.orarg + 'v' + p.andarg
#
#     @_('andarg')
#     def orarg(self, p):
#         return p.andarg
#
#     @_('andarg AND imparg')
#     def andarg(self, p):
#         return p.andarg + '^' + p.imparg
#
#     @_('imparg')
#     def andarg(self, p):
#         return p.imparg
#
#     @_('imparg IMP dblimparg')
#     def imparg(self, p):
#         return p.imparg + '->' + p.dblimparg
#
#     @_('dblimparg')
#     def imparg(self, p):
#         return p.dblimparg
#
#     @_('dblimparg D_IMP notarg')
#     def dblimparg(self, p):
#         return p.dblimparg + '<->' + p.notarg
#
#     @_('notarg')
#     def dblimparg(self, p):
#         return p.notarg
#
#     @_('NOT term')
#     def notarg(self, p):
#         return '~' + p.term
#
#     @_('term')
#     def notarg(self, p):
#         return p.term
#
#     @_('R_PAREN expr L_PAREN')
#     def term(self, p):
#         return '(' + p.expr + ')'
#
#     @_('P')
#     def term(self, p):
#         return 'p'
#
#     @_('Q')
#     def term(self, p):
#         return 'q'
#
#     @_('R')
#     def term(self, p):
#         return 'r'
#
#     @_('TRUE')
#     def term(self, p):
#         return 'T'
#
#     @_('FALSE')
#     def term(self, p):
#         return 'F'












#
# class LogicParser(Parser):
#     # Get the token list from the lexer (required)
#     tokens = LogicLexer.tokens
#
#     # precedence = (
#     #                 ('right', NOT),
#     #                 ('nonassoc', D_IMP),
#     #                 ('right', IMP),
#     #                 ('right', AND),
#     #                 ('right', OR),
#     # )
#
#     # precedence = (
#     #                 ('right', OR),
#     #                 ('right', AND),
#     #                 ('right', IMP),
#     #                 ('nonassoc', D_IMP),
#     #                 ('right', NOT)
#     # )
#
#     @_('orarg')
#     def expr(self, p):
#         return p[0]
#
#     @_('orarg OR andarg')
#     def orarg(self, p):
#         return OrOp(p[1], p[0], p[2])
#
#     @_('andarg')
#     def orarg(self, p):
#         return p[0]
#
#     @_('andarg AND imparg')
#     def andarg(self, p):
#         return AndOp(p[1], p[0], p[2])
#
#     @_('imparg')
#     def andarg(self, p):
#         return p[0]
#
#     @_('imparg IMP dblimparg')
#     def imparg(self, p):
#         return ImpOp(p[1], p[0], p[2])
#
#     @_('dblimparg')
#     def imparg(self, p):
#         return p[0]
#
#     @_('dblimparg D_IMP notarg')
#     def dblimparg(self, p):
#         return DblimpOp(p[1], p[0], p[2])
#
#     @_('notarg')
#     def dblimparg(self, p):
#         return p[0]
#
#     @_('NOT term')
#     def notarg(self, p):
#         return NotOp(p[0], p[1])
#
#     @_('term')
#     def notarg(self, p):
#         return p[0]
#
#     @_('R_PAREN expr L_PAREN')
#     def term(self, p):
#         return p[1]
#
#     @_('P')
#     def term(self, p):
#         return Var(p[0])
#
#     @_('Q')
#     def term(self, p):
#         return Var(p[0])
#
#     @_('R')
#     def term(self, p):
#         return Var(p[0])
#
#     @_('TRUE')
#     def term(self, p):
#         return Lit(p[0])
#
#     @_('FALSE')
#     def term(self, p):
#         return Lit(p[0])
#



# comment
