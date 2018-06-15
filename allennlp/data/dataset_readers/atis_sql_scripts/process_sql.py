from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


my_grammar = Grammar(r"""
    styled_text = bold_text / italic_text
    bold_text   = left text right
    left        = "(("
    right       = "))"
    italic_text = "''" text "''"
    text        = ~"[A-Z 0-9]*"i
    """)


class Visitor(NodeVisitor):
    grammar = my_grammar
    prod_acc = []

    def __init__(self):
        grammar = my_grammar
        prod_acc = []

        print(grammar.keys())
        for nonterm in grammar.keys():
            if nonterm != 'styled_text':
                self.__setattr__("visit_" + nonterm, self.add_prod_rule) 

    def add_prod_rule(self, node, children):
        rule = "{} =".format(node.expr_name)
        if children:
            for child in node.__iter__():
                rule += " {}".format(child.expr_name)
        else:
            rule += " {}".format(node.text)

        self.prod_acc = [rule] + self.prod_acc


    def visit_styled_text(self, node, children):
        self.add_prod_rule(node, children)
        return self.prod_acc

print(Visitor().parse('(( bold stuff ))'))

