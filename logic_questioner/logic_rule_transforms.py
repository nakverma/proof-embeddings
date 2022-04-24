import itertools
from collections import OrderedDict

from lark import Lark, Tree, Token
from itertools import permutations


def is_token(node: object, token_type: str) -> bool:
    return type(node) == Token and node.type == token_type


def is_tree(node: object, tree_data: str) -> bool:
    return type(node) == Tree and node.data == tree_data


def negate(node: object) -> Tree:
    return Tree("literal", [Token("NOT", "~"), node])


def parenthesize(node: object) -> Tree:
    return Tree("paren_expr", [node])


def simplify_paren_expr(tree: Tree) -> object:
    assert tree.data == "paren_expr"
    while is_tree(tree.children[0], 'paren_expr'):
        tree = tree.children[0]
    if type(tree.children[0]) != Tree or is_tree(tree.children[0], 'literal') or len(tree.children[0].children) == 1:
        tree = tree.children[0]
    return tree


def safe_paren(node: object):
    return simplify_paren_expr(parenthesize(node))


def idempotence(tree: Tree):
    assert tree.data == "term" or tree.data == "expr"
    return Tree(tree.data, list({k: None for k in tree.children}.keys()))  # dicts remember insertion order now!


def reverse_idempotence(node):
    if is_tree(node, "term") or is_tree(node, "expr"):
        new_trees = [
            Tree(node.data, node.children[:i] + [c] + node.children[i:]) for i, c in enumerate(node.children)
        ]
    else:
        new_trees = [
            safe_paren(Tree("expr", [node, node])), safe_paren(Tree("term", [node, node]))
        ]
    return new_trees


def simplify_multiple_negation(tree: Tree):
    assert tree.data == "literal"
    ch = tree
    count = 0
    while is_tree(ch, "literal") or is_tree(ch, "paren_expr") and is_tree(ch.children[0], "literal"):
        count += 1 if is_tree(ch, "literal") else 0
        ch = ch.children[1] if is_tree(ch, "literal") else ch.children[0]
    tree = negate(ch) if count % 2 != 0 else ch
    return tree


def identity(tree: Tree):
    assert tree.data == "expr" or tree.data == "term"
    if tree.data == "expr":
        new_children = list(filter(lambda x: not is_token(x, "FALSE"), tree.children))
        new_children = new_children if len(new_children) > 0 else [Token("FALSE", "F")]
    elif tree.data == "term":
        new_children = list(filter(lambda x: not is_token(x, "TRUE"), tree.children))
        new_children = new_children if len(new_children) > 0 else [Token("TRUE", "T")]
    return Tree(tree.data, new_children)


def reverse_identity(node):
    tree = simplify_paren_expr(parenthesize(node))
    new_trees = []
    new_trees.extend([
        parenthesize(Tree("term", [Token("TRUE", "T"), tree])),
        parenthesize(Tree("term", [tree, Token("TRUE", "T")]))
    ])
    new_trees.extend([
        parenthesize(Tree("expr", [Token("FALSE", "F"), tree])),
        parenthesize(Tree("expr", [tree, Token("FALSE", "F")]))
    ])
    return new_trees


def domination(tree: Tree):
    assert tree.data == "expr" or tree.data == "term"
    new_tree = tree
    if tree.data == "expr" and any([is_token(c, "TRUE") for c in tree.children]):
        new_tree = Token("TRUE", "T")
    elif tree.data == "term" and any([is_token(c, "FALSE") for c in tree.children]):
        new_tree = Token("FALSE", "F")
    return new_tree


def commutativity(tree: Tree):  # p^q^r == q^p^r == ... r^p^q, returns list of permutations
    assert tree.data == "expr" or tree.data == "term" or tree.data == "eqn"
    new_trees = [Tree(tree.data, p) for p in itertools.permutations(tree.children)]
    return new_trees


def associativity_LR(tree: Tree):  # only for (a^b)^c or with V, i.e. 2-node 3-var ops. Somewhat Hacky. Expand later
    assert tree.data == "expr" or tree.data == "term"
    ch = tree.children
    new_trees = []
    for i, c in enumerate(ch):
        if is_tree(c, "paren_expr") and is_tree(c.children[0], tree.data):
            swap_ch = c.children[0].children
            if i > 0:
                for j in range(len(swap_ch)-1):
                    new_paren = parenthesize(Tree(tree.data, ch[:i] + swap_ch[:j+1]))
                    new_trees.append(Tree(tree.data, [new_paren] + swap_ch[j+1:] + ch[i+1:]))
            if i < len(ch)-1:
                for j in range(1, len(swap_ch)):
                    new_paren = parenthesize(Tree(tree.data, swap_ch[j:] + ch[i+1:]))
                    new_trees.append(Tree(tree.data, ch[:i] + swap_ch[:j] + [new_paren]))
    return new_trees


def associativity_expand(tree: Tree):  # remove all parenthesized expressions
    assert tree.data == "expr" or tree.data == "term"
    new_children = []
    for c in tree.children:
        if is_tree(c, "paren_expr") and is_tree(c.children[0], tree.data):
            new_children.extend(c.children[0].children)
        else:
            new_children.append(c)
    tree.children = new_children
    return tree


def reverse_associativity_expand(tree: Tree):  # add parentheses around arbitrary sequences of expressions
    assert tree.data == "expr" or tree.data == "term"
    ch = tree.children
    new_trees = []
    for i in range(len(ch)-1):
        for j in range(i+2, len(ch)+1):
            new_trees.append(Tree(tree.data, ch[:i] + [parenthesize(Tree(tree.data, ch[i:j]))] + ch[j:]))
    return new_trees


def impl_to_disj(tree: Tree):  # p->q == ~pVq
    assert tree.data == "dbl_expr"
    if len(tree.children) > 1:
        new_trees = []
        for i in range(len(tree.children) - 1):
            impl_fwd = Tree("expr", [simplify_multiple_negation(negate(tree.children[i])), tree.children[i+1]])
            impl_rev = Tree("expr", [tree.children[i+1], simplify_multiple_negation(negate(tree.children[i]))])
            new_trees.append(Tree("dbl_expr", tree.children[:i] + [safe_paren(impl_fwd)] + tree.children[i+2:]))
            new_trees.append(Tree("dbl_expr", tree.children[:i] + [safe_paren(impl_rev)] + tree.children[i+2:]))
        return new_trees
    return [tree]


def disj_to_impl(tree: Tree):  # ~pVq == p->q, convert any adjacent pair to implication
    assert tree.data == "expr"
    if len(tree.children) > 1:
        new_trees = []
        for i in range(len(tree.children) - 1):
            expr_fwd = Tree("dbl_expr", [simplify_multiple_negation(negate(tree.children[i])), tree.children[i+1]])
            expr_rev = Tree("dbl_expr", [simplify_multiple_negation(negate(tree.children[i+1])), tree.children[i]])
            new_trees.append(Tree("expr", tree.children[:i] + [safe_paren(expr_fwd)] + tree.children[i+2:]))
            new_trees.append(Tree("expr", tree.children[:i] + [safe_paren(expr_rev)] + tree.children[i+2:]))
        return new_trees
    return [tree]


def dblimpl_to_impl(tree: Tree):  # p<=>q == (p->q)^(q->p), only leftmost (commute for others)
    assert tree.data == "eqn"
    if len(tree.children) > 1:
        p, q = tree.children[0], tree.children[1]
        pq = Tree('paren_expr', [Tree('dbl_expr', [p, q])])
        qp = Tree('paren_expr', [Tree('dbl_expr', [q, p])])
        t = Tree('term', [pq, qp])
        if len(tree.children) == 2:
            tree = t
        else:
            tree.children = [t] + tree.children[2:]
    return tree


def impl_to_dblimpl(tree: Tree):  # (p->q)^(q->p) == p<=>q, any adjacent pair can be operated on
    assert tree.data == "term"
    if len(tree.children) == 1:
        return [tree]
    new_trees = []
    for i, c in enumerate(tree.children[:-1]):
        c2 = tree.children[i+1]
        if is_tree(c, "paren_expr") and is_tree(c2, "paren_expr"):
            if is_tree(c.children[0], "dbl_expr") and is_tree(c2.children[0], "dbl_expr"):
                ch0, ch1 = c.children[0].children[0], c.children[0].children[1]
                if ch0 == c2.children[0].children[1] and ch1 == c2.children[0].children[0]:
                    pre_exclude = tree.children[:i]
                    post_exclude = tree.children[i+2:]
                    tr_fwd, tr_bkwd = Tree("eqn", [ch0, ch1]), Tree("eqn", [ch1, ch0])
                    new_trees.append(Tree("term", pre_exclude + [tr_fwd] + post_exclude))
                    new_trees.append(Tree("term", pre_exclude + [tr_bkwd] + post_exclude))
    return new_trees if len(new_trees) > 0 else [tree]


def negation(tree: Tree):  # pv~p=T, p^~p=F
    assert tree.data == "expr" or tree.data == "term"
    if len(tree.children) == 1:
        return tree
    pos_dict = {c: i for i, c in enumerate(tree.children)}
    new_trees = []
    for i, c in enumerate(tree.children):
        neg_c = simplify_multiple_negation(negate(c))
        if neg_c in pos_dict and i < pos_dict[neg_c]:                     # avoid duplicates
            j = pos_dict[neg_c]
            tok = Token("TRUE", "T") if tree.data == "expr" else Token("FALSE", "F")
            new_trees.append(Tree(tree.data, tree.children[:i] + [tok] + tree.children[i+1:j] + tree.children[j+1:]))
            new_trees.append(Tree(tree.data, tree.children[:i] + tree.children[i+1:j] + tree.children[j+1:] + [tok]))
    return new_trees if len(new_trees) > 0 else [tree]


def reverse_negation(token: Token, additional_ids=('p', 'q', 'r', 's')):
    assert is_token(token, "TRUE") or is_token(token, "FALSE")
    new_trees = []
    for i in additional_ids:
        tok = Token("ID", i)
        if is_token(token, "TRUE"):
            new_trees.append(parenthesize(Tree("expr", [tok, negate(tok)])))
        else:
            new_trees.append(parenthesize(Tree("term", [tok, negate(tok)])))
    return new_trees


def demorgan(tree: Tree):  # ~(pvq) == ~p^~q, ~(p^q) == ~pV~q
    assert tree.data == "literal"
    if is_tree(tree.children[1], "paren_expr"):
        ch = tree.children[1].children[0]
        if (is_tree(ch, "expr") or is_tree(ch, "term")) and len(ch.children) > 1:
            dual = "expr" if ch.data == "term" else "term"
            new_ch = [simplify_multiple_negation(negate(c)) for c in ch.children]
            tree = parenthesize(Tree(dual, new_ch))
    return tree


def reverse_demorgan(tree: Tree):  # ~pV~q == ~(p^q), ~p^~q == ~(pvq)
    assert tree.data == "expr" or tree.data == "term"
    if len(tree.children) == 1:
        return [tree]
    new_trees = []
    dual = "expr" if tree.data == "term" else "term"
    for i, c in enumerate(tree.children[:-1]):
        new_ch = [simplify_multiple_negation(negate(c)) for c in tree.children[i:i+2]]
        tr = negate(Tree("paren_expr", [Tree(dual, new_ch)]))
        new_trees.append(Tree(tree.data, tree.children[:i] + [tr] + tree.children[i+2:]))
    return new_trees if len(new_trees) > 0 else [tree]


def absorption(tree: Tree):  # pV(p^q) == p, p^(pVq) == p
    assert tree.data == "expr" or tree.data == "term"
    ch_set = set(tree.children)
    dual = "expr" if tree.data == "term" else "term"
    for par in filter(lambda x: is_tree(x, "paren_expr"), tree.children):
        pc = par.children[0]
        if is_tree(pc, dual):
            for ch in pc.children:
                if ch in ch_set:
                    tree = ch  # change to append to possible tree list
                    return tree
    return tree


def reverse_absorption(node, additional_ids=('p', 'q', 'r', 's')):  # p == pv(p^ID), p == p^(pvID)
    new_trees = []
    if is_token(node, "ID"):
        additional_ids = [v for v in additional_ids if node.value != v]
    for v in additional_ids:
        new_trees.append(Tree("expr", [node, parenthesize(Tree("term", [node, v]))]))
        new_trees.append(Tree("term", [node, parenthesize(Tree("expr", [node, v]))]))
    return new_trees


def TF_negation(tree: Tree):  # ~T == F, ~F == T
    assert tree.data == "literal"
    if len(tree.children) == 2:
        if is_token(tree.children[1], "TRUE"):
            tree = Token("FALSE", "F")
        elif is_token(tree.children[1], "FALSE"):
            tree = Token("TRUE", "T")
    return tree


def distributivity(tree: Tree):  # pv(q^r) == (pvq)^(pvr) etc., assumes args in order. Hacky
    assert tree.data == "expr" or tree.data == "term"
    if len(tree.children) == 1:
        return [tree]
    new_trees = []
    for i, c in enumerate(tree.children):
        if is_tree(c, "paren_expr") and (is_tree(c.children[0], "expr") or is_tree(c.children[0], "term")):
            pre_exclude = tree.children[:i]
            post_exclude = tree.children[i+1:]  # These preserve the original order of arguments
            new_children = [
                Tree("paren_expr", [Tree(tree.data, [n1, n2])]) for n1 in pre_exclude for n2 in c.children[0].children
            ] + [
                Tree("paren_expr", [Tree(tree.data, [n1, n2])]) for n1 in c.children[0].children for n2 in post_exclude
            ]
            new_trees.append(Tree(c.children[0].data, new_children))
    return new_trees if len(new_trees) > 0 else [tree]


def reverse_distributivity(tree: Tree):  # (pvq)^(pvr) == pV(q^r) etc., assumes args in order. Hacky
    assert tree.data == "expr" or tree.data == "term"
    if len(tree.children) == 1:
        return [tree]

    def is_viable_node(n):
        return is_tree(n, "paren_expr") \
               and (is_tree(n.children[0], "expr") or is_tree(n.children[0], "term")) \
               and len(n.children[0].children) == 2

    new_trees = []
    viable_nodes = list(filter(lambda x: is_viable_node(x[1]), enumerate(tree.children)))
    for pos, val in enumerate(viable_nodes[:-1]):
        i, c1 = val
        for j, c2 in viable_nodes[pos+1:]:
            d1, d2 = c1.children[0], c2.children[0]  # children of paren_expr (i.e. the dual expression)
            for ch in d2.children:
                if ch in d1.children:
                    exclude = [c for k, c in enumerate(tree.children) if k not in (i, j)]
                    idxs = (d1.children.index(ch), d2.children.index(ch))
                    factored = [c for k, c in enumerate(d1.children) if k != idxs[0]]  # accommodate (ch, ch) e.g. (p^p)
                    factored += [c for k, c in enumerate(d2.children) if k != idxs[1]]
                    factored_tree = Tree(d1.data, [ch, parenthesize(Tree(tree.data, factored))])  # pV(q^r)
                    new_trees.append(Tree(tree.data, [factored_tree] + exclude))

    return new_trees if len(new_trees) > 0 else [tree]


def double_negate(tree: Tree):  # p == ~~p
    tree = negate(negate(simplify_paren_expr(Tree('paren_expr', [tree]))))
    return tree


operation_names = {                       # change to Enum?
    "Double Negation": [double_negate, simplify_multiple_negation],
    "Implication as Disjunction": [impl_to_disj, disj_to_impl],
    "Iff as Implication": [dblimpl_to_impl, impl_to_dblimpl],
    "Idempotence": [idempotence, reverse_idempotence],
    "Identity": [identity, reverse_identity],
    "Domination": [domination],
    "Commutativity": [commutativity],
    "Associativity": [associativity_LR, associativity_expand, reverse_associativity_expand],
    "Negation": [negation, TF_negation, reverse_negation],
    "Absorption": [absorption, reverse_absorption],
    "Distributivity": [distributivity, reverse_distributivity],
    "De Morgan's Law": [demorgan, reverse_demorgan],
    "Simplify Parentheses": [simplify_paren_expr]
}


def get_operation_name(op):
    for k, v in operation_names.items():
        if op in v:
            return k
    return None


allowed_operations = {
    'start': [],
    'eqn': [
        dblimpl_to_impl, commutativity, reverse_identity
    ],
    'dbl_expr': [
        impl_to_disj, reverse_identity
    ],
    'expr': [
        idempotence, identity, domination, commutativity, associativity_LR, associativity_expand,
        reverse_associativity_expand, negation, absorption, distributivity, reverse_distributivity, double_negate,
        reverse_demorgan, disj_to_impl, reverse_identity, reverse_idempotence
    ],
    'term': [
        idempotence, identity, domination, commutativity, associativity_LR, associativity_expand,
        reverse_associativity_expand, negation, absorption, distributivity, reverse_distributivity, double_negate,
        reverse_demorgan, impl_to_dblimpl, reverse_identity, reverse_idempotence
    ],
    'literal': [
        simplify_multiple_negation, TF_negation, demorgan, reverse_identity, reverse_idempotence
    ],
    'variable': [],
    'paren_expr': [
        double_negate, reverse_identity, reverse_idempotence, reverse_absorption
    ],
    'ID': [
        reverse_identity, reverse_idempotence, reverse_absorption
    ],
    "TRUE": [
        reverse_negation
    ],
    "FALSE": [
        reverse_negation
    ],
    "_LPAR": [],
    "_RPAR": [],
    "NOT": [],
    "_AND": [],
    "_OR": [],
    "_IMPL": [],
    "_DBLIMPL": []
}


search_operations = {
    'start': [],
    'eqn': [
        dblimpl_to_impl, commutativity
    ],
    'dbl_expr': [
        impl_to_disj
    ],
    'expr': [
        idempotence, identity, domination, commutativity, associativity_LR, associativity_expand,
        reverse_associativity_expand, negation, absorption, distributivity, reverse_distributivity, double_negate,
        reverse_demorgan, disj_to_impl
    ],
    'term': [
        idempotence, identity, domination, commutativity, associativity_LR, associativity_expand,
        reverse_associativity_expand, negation, absorption, distributivity, reverse_distributivity, double_negate,
        reverse_demorgan, impl_to_dblimpl
    ],
    'literal': [
        simplify_multiple_negation, TF_negation, demorgan
    ],
    'variable': [],
    'paren_expr': [
        double_negate, reverse_absorption
    ],
    'ID': [
        reverse_absorption
    ],
    "TRUE": [
        reverse_negation
    ],
    "FALSE": [
        reverse_negation
    ],
    "_LPAR": [],
    "_RPAR": [],
    "NOT": [],
    "_AND": [],
    "_OR": [],
    "_IMPL": [],
    "_DBLIMPL": []
}


if __name__ == "__main__":
    from expression_parser import ExpressionParser, TreeToString

    ep = ExpressionParser()
    tts = TreeToString()

    tr1 = ep.parse('(pvq)^(pvr)^(qvr)^(pvrvs)^(avb)^(avc)').children[0]
    tr2 = reverse_distributivity(tr1)
    print([tts.transform(t) for t in tr2])
    tr1 = ep.parse('(pvq)^(rvq)').children[0]
    tr2 = reverse_distributivity(tr1)
    print([tts.transform(t) for t in tr2])
    tr1 = ep.parse('(pvq)^(pvr)').children[0]
    tr2 = reverse_distributivity(tr1)
    print([tts.transform(t) for t in tr2])
    tr1 = ep.parse('(r^q)^(p^r)').children[0]
    tr2 = reverse_distributivity(tr1)
    print([tts.transform(t) for t in tr2])
    tr1 = ep.parse('pvq').children[0]
    tr2 = reverse_distributivity(tr1)
    print([tts.transform(t) for t in tr2])

    t1 = ep.parse('a^b^c^a^b^a').children[0]
    t1 = idempotence(t1)
    print(t1 if type(t1) == Token else tts.transform(t1), sep="\n")
    t3 = ep.parse('(a^b^c)v(a^b^c)').children[0]
    t3 = idempotence(t3)
    print(t3 if type(t3) == Token else tts.transform(t3), sep="\n")
