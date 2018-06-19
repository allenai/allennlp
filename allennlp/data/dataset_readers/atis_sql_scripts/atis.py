from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import RegexNode

SQL_GRAMMAR = Grammar(r"""
    stmt            = query ";" ws
    query           = ws lparen?  ws "SELECT" ws "DISTINCT"? ws select_results ws "FROM" ws table_name ws WHERE rparen?  ws

    select_results  = col_refs / agg

    agg             = agg_func ws lparen ws col_ref ws rparen
    agg_func        = "MIN"

    col_refs        = col_ref (ws "," ws col_ref)* 
    name            = ~"[a-zA-Z]\w*"i
    ws              = ~"\s*"i
    table_name      = name
    column_name     = name
    col_ref         =  (table_name ws "." ws column_name) / column_name
    WHERE           = "WHERE" ws lparen? ws condition_paren (ws conj ws condition_paren)* ws rparen? ws
    lparen          = "(" 
    rparen          = ")"

    condition_paren = (lparen ws)? condition (ws rparen)?
    condition       = biexpr / in_clause / ternaryexpr

    in_clause       = (lparen ws)? col_ref ws "IN" ws query (ws rparen)?
    
    biexpr          = col_ref ws binaryop ws value 
    binaryop        = "+" / "-" / "*" / "/" / "=" / "<>" /
                      "<=" / ">" / "<" / ">" / "and" / "or"

    ternaryexpr     = col_ref ws "BETWEEN" ws value ws AND value ws

    value           = number / boolean / col_ref / string / agg_results

    agg_results     = ws lparen?  ws "SELECT" ws "DISTINCT"? ws agg ws "FROM" ws table_name ws WHERE rparen?  ws

    number          = ~"\d*\.?\d+"i
    string          = ~"\'.*?\'"i
    boolean         = "true" / "false"
    
    conj            = AND / OR
    AND             = "AND" ws 
    OR              = "OR" ws

    """) 

class SQLVisitor(NodeVisitor):
    grammar = SQL_GRAMMAR

    def __init__(self):
        self.prod_acc = []

        for nonterm in self.grammar.keys():
            if nonterm != 'stmt':
                self.__setattr__('visit_' + nonterm, self.add_prod_rule)

    def generic_visit(self, node, visited_children):
        self.add_prod_rule(node)

    def add_prod_rule(self, node, children=None):
        if node.expr.name and node.expr.name != 'ws':
            rule = '{} ='.format(node.expr.name)

            if isinstance(node, RegexNode):
                rule += '"{}"'.format(node.text)

            for child in node.__iter__():
                if child.expr.name != '':
                    rule += ' {}'.format(child.expr.name)
                else:
                    rule += ' {}'.format(child.expr._as_rhs())

            print('adding rule: {}'.format(rule))
            self.prod_acc = [rule] + self.prod_acc

    def visit_stmt(self, node, children):
        self.add_prod_rule(node)
        return self.prod_acc


with open('test.sql') as sql_file:
    for line in sql_file:
        sql_visitor = SQLVisitor()
        query = line.strip("\n")
        print("Parsing query: ", query)
        prod_rules = sql_visitor.parse(query)
        print(prod_rules)
        print(len(prod_rules))
        print("\n\n")
