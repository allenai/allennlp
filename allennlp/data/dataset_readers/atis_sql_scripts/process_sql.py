from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

sql_grammar = Grammar(r"""
    query    = select_cores orderby? limit?
    select_cores   = select_core (compound_op select_core)*
    select_core    = SELECT wsp select_results from_clause? where_clause? gb_clause?
    select_results = select_result (ws "," ws select_result)*
    select_result  = sel_res_all_star / sel_res_tab_star / sel_res_val / sel_res_col
    sel_res_tab_star = name ".*"
    sel_res_all_star = "*"
    sel_res_val    = expr (AS wsp name)?
    sel_res_col    = col_ref (AS wsp name)
    from_clause    = FROM join_source
    join_source    = ws single_source (ws "," ws single_source)*
    single_source  = source_table / source_subq
    source_table   = table_name (AS wsp name)?
    source_subq    = "(" ws query ws ")" (AS ws name)?
    where_clause   = WHERE wsp expr (AND expr)*
    gb_clause      = GROUP BY group_clause having_clause?
    group_clause   = grouping_term (ws "," grouping_term)*
    grouping_term  = ws expr
    having_clause  = HAVING expr
    orderby        = ORDER BY ordering_term (ws "," ordering_term)*
    ordering_term  = ws expr (ASC/DESC)?
    limit          = LIMIT expr (OFFSET expr)?
    col_ref        = (table_name ".")? column_name
    expr     = biexpr / unexpr / value
    biexpr   = value ws binaryop ws expr
    unexpr   = unaryop expr
    value    = parenval /
               number /
               boolean /
               col_ref /
               function /
               string /
               attr
    parenval = "(" ws expr ws ")"
    function = fname "(" ws arg_list? ws ")"
    arg_list = expr (ws "," ws expr)*
    number   = ~"\d*\.?\d+"i
    string   = ~"\'\w*\'"i
    attr     = ~"\w[\w\d]*"i
    fname    = ~"\w[\w\d]*"i
    boolean  = "true" / "false"
    compound_op = "UNION" / "union"
    binaryop = "+" / "-" / "*" / "/" / "=" / "<>" /
               "<=" / ">" / "<" / ">" / "and" / "or"
    unaryop  = "+" / "-" / "not"
    ws       = ~"\s*"i
    wsp      = ~"\s+"i
    name       = ~"[a-zA-Z]\w*"i
    table_name = name
    column_name = name
    ADD = wsp "ADD"
    ALL = wsp "ALL"
    ALTER = wsp "ALTER"
    AND = wsp "AND"
    AS = wsp "AS"
    ASC = wsp "ASC"
    BETWEEN = wsp "BETWEEN"
    BY = wsp "BY"
    CAST = wsp "CAST"
    COLUMN = wsp "COLUMN"
    DESC = wsp "DESC"
    DISTINCT = wsp "DISTINCT"
    E = "E"
    ESCAPE = wsp "ESCAPE"
    EXCEPT = wsp "EXCEPT"
    EXISTS = wsp "EXISTS"
    EXPLAIN = ws "EXPLAIN"
    EVENT = ws "EVENT"
    FORALL = wsp "FORALL"
    FROM = wsp "FROM"
    GLOB = wsp "GLOB"
    GROUP = wsp "GROUP"
    HAVING = wsp "HAVING"
    IN = wsp "IN"
    INNER = wsp "INNER"
    INSERT = ws "INSERT"
    INTERSECT = wsp "INTERSECT"
    INTO = wsp "INTO"
    IS = wsp "IS"
    ISNULL = wsp "ISNULL"
    JOIN = wsp "JOIN"
    KEY = wsp "KEY"
    LEFT = wsp "LEFT"
    LIKE = wsp "LIKE"
    LIMIT = wsp "LIMIT"
    MATCH = wsp "MATCH"
    NO = wsp "NO"
    NOT = wsp "NOT"
    NOTNULL = wsp "NOTNULL"
    NULL = wsp "NULL"
    OF = wsp "OF"
    OFFSET = wsp "OFFSET"
    ON = wsp "ON"
    OR = wsp "OR"
    ORDER = wsp "ORDER"
    OUTER = wsp "OUTER"
    PRIMARY = wsp "PRIMARY"
    QUERY = wsp "QUERY"
    RAISE = wsp "RAISE"
    REFERENCES = wsp "REFERENCES"
    REGEXP = wsp "REGEXP"
    RENAME = wsp "RENAME"
    REPLACE = ws "REPLACE"
    RETURN = wsp "RETURN"
    ROW = wsp "ROW"
    SAVEPOINT = wsp "SAVEPOINT"
    SELECT = ws "SELECT"
    SET = wsp "SET"
    TABLE = wsp "TABLE"
    TEMP = wsp "TEMP"
    TEMPORARY = wsp "TEMPORARY"
    THEN = wsp "THEN"
    TO = wsp "TO"
    UNION = wsp "UNION"
    USING = wsp "USING"
    VALUES = wsp "VALUES"
    VIRTUAL = wsp "VIRTUAL"
    WITH = wsp "WITH"
    WHERE = wsp "WHERE"
    """)

class SQLVisitor(NodeVisitor):
    grammar = sql_grammar 
    prod_acc = []

    def __init__(self):
        grammar = sql_grammar 
        prod_acc = []
        
        for nonterm in grammar.keys():
            if nonterm != 'query':
                self.__setattr__("visit_" + nonterm, self.add_prod_rule) 
        self.__setattr__("generic_visit", self.add_prod_rule)

    def add_prod_rule(self, node, children):
        '''
        if node.expr_name == "":
            return
        '''

        rule = "{} =".format(node.expr_name)
        if children:
            for child in node.__iter__():
                rule += " {}".format(child.expr_name)
        else:
            rule += " {}".format(node.text)

        self.prod_acc = [rule] + self.prod_acc  

    def visit_query(self, node, children):
        self.add_prod_rule(node, children)
        return self.prod_acc

print(SQLVisitor().parse("SELECT tcol FROM table"))
