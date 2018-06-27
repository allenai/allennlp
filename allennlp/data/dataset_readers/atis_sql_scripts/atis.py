import argparse
import os

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import RegexNode

SQL_GRAMMAR = Grammar(r"""
    stmt                = query ";" ws

    query               = ws lparen?  ws "SELECT" ws "DISTINCT"? ws select_results ws "FROM" ws table_refs ws where_clause rparen?  ws
    select_results      = agg / col_refs

    agg                 = agg_func ws lparen ws col_ref ws rparen
    agg_func            = "MIN" / "min" / "MAX" / "max" / "COUNT" / "count" 
    
    col_refs            = (col_ref (ws "," ws col_ref)*)  
    col_ref             = (table_name ws "." ws column_name) / column_name / asterisk
        
    table_refs          = table_name (ws "," ws table_name)* 
    table_name          = name

    column_name         = name

    where_clause        = "WHERE" ws lparen? ws condition_paren (ws conj ws condition_paren)* ws rparen? ws
    
    condition_paren     = not? (lparen ws)? condition_paren2 (ws rparen)?
    condition_paren2    = not? (lparen ws)? condition_paren3 (ws rparen)?
    condition_paren3    = not? (lparen ws)? condition (ws rparen)?
    condition           = in_clause / ternaryexpr / biexpr 

    in_clause       = (lparen ws)? col_ref ws "IN" ws query (ws rparen)?
    
    biexpr          = ( col_ref ws binaryop ws value) / (value ws binaryop ws value) / ( col_ref ws "LIKE" ws string)
    binaryop        = "+" / "-" / "*" / "/" / "=" / 
                      ">=" / "<=" / ">" / "<"  / "is" / "IS"

    ternaryexpr     = col_ref ws not? "BETWEEN" ws value ws and value ws
    
    value           = not? ws? pos_value
    pos_value       = ("ALL" ws query) / ("ANY" ws query) / number / boolean / col_ref / string / agg_results / "NULL"

    agg_results     = ws lparen?  ws "SELECT" ws "DISTINCT"? ws agg ws "FROM" ws table_name ws where_clause rparen?  ws

    number          = ~"\d*\.?\d+"i
    string          = ~"\'.*?\'"i
    boolean         = "true" / "false"

    name                = ~"[a-zA-Z]\w*"i
    ws                  = ~"\s*"i

    lparen              = "(" 
    rparen              = ")"
    conj            = and / or 
    and             = "AND" ws 
    or              = "OR" ws
    not             = ("NOT" ws ) / ("not" ws)
    asterisk        = "*"

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

            self.prod_acc = [rule] + self.prod_acc

    def visit_stmt(self, node, children):
        self.add_prod_rule(node)
        return self.prod_acc



parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
args = parser.parse_args()


processed_file = open(args.out_file, 'w')
if not args.in_file.endswith("sql"):
    num_queries = 0
    num_parsed = 0
    for root, dirs, files in os.walk(args.in_file):
        for filename in files:
            if filename.endswith("sql") and "rdb" not in root:
                full_path = os.path.join(root, filename)
                with open( full_path, "r") as sql_file:
                    for line in sql_file:
                        num_queries+= 1
                        sql_visitor = SQLVisitor()
                        query = line.strip("\n")
                        try:
                            prod_rules = sql_visitor.parse(query)
                            num_parsed += 1
                        except:
                            print(line)
                            processed_file.write(line)
                            pass
    print("Parsed {} out of {} queries, coverage {}".format(num_parsed, num_queries, num_parsed / num_queries))
    processed_file.write("Parsed {} out of {} queries, coverage {}".format(num_parsed, num_queries, num_parsed / num_queries))
    
    processed_file.close()

else:
    with open(args.in_file, 'r') as sql_file:
        with open(args.out_file, 'w') as processed_file:
            for line in sql_file:
                sql_visitor = SQLVisitor()
                query = line.strip("\n")
                prod_rules = sql_visitor.parse(query)

