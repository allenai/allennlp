from parsimonious.grammar import Grammar

# Changes:
# Added AVG, SUM to agg_func
# added optional AS to table_name
# Made where_clause optional in query
# Add query to pos_value (TODO check this, very unclear if it is the correct way to handle this)
# Added optional DISTINCT inside agg
# added <> binary op
# added biexpr to agg to support "SELECT TABLE.COLUMN / TABLE2.COLUMN2 FROM ..."
# added optional extra agg clauses connected to each other by a binaryop, fixes e.g SUM ( STATEalias0.POPULATION ) / SUM ( STATEalias0.AREA )
# Added optional nested brackets inside DISTINCT for aggregates (common in yelp)

# still TODO: 
# think about the  difference between variable and string matching
# tablename in (tablename1, tablename2)
# JOIN, seems hard.

# TODO:
# string set isn't an expr, it should only be in in_expr
# not all functions can take * as an argument.    

SQL_GRAMMAR2 = Grammar(
        r"""
        stmt             = (query ws ";") / (query ws)
        query            = (ws select_core groupby_clause ws orderby_clause ws limit) /
                           (ws select_core groupby_clause ws limit) /
                           (ws select_core orderby_clause ws limit) /
                           (ws select_core groupby_clause) /
                           (ws select_core orderby_clause) /
                           (ws select_core)

        select_core      = (select_with_distinct select_results from_clause where_clause) /
                           (select_with_distinct select_results from_clause) /
                           (select_with_distinct select_results where_clause) /
                           (select_with_distinct select_results)

        select_with_distinct = (SELECT DISTINCT) / SELECT
        select_results   = (ws select_result ws "," ws select_results) / (ws select_result)
        select_result    = sel_res_all_star / sel_res_tab_star / sel_res_val / sel_res_col

        sel_res_tab_star = name ".*"
        sel_res_all_star = "*"
        sel_res_val      = (expr AS wsp name) / expr
        sel_res_col      = col_ref (AS wsp name)

        from_clause      = FROM source
        source           = (ws single_source ws "," ws source) / (ws single_source)
        single_source    = source_table / source_subq
        source_table     = table_name (AS wsp name)
        source_subq      = ("(" ws query ws ")" AS ws name) / ("(" ws query ws ")")
        where_clause     = (WHERE wsp expr where_conj) / (WHERE wsp expr)
        where_conj       = (AND wsp expr where_conj) / (AND wsp expr)

        groupby_clause   = (GROUP BY group_clause having_clause) / (GROUP BY group_clause)
        group_clause     = (ws expr ws "," group_clause) / (ws expr)
        having_clause    = HAVING ws expr

        orderby_clause   = ORDER BY order_clause
        order_clause     = (ordering_term ws "," order_clause) / ordering_term
        ordering_term    = (ws expr ordering) / (ws expr)
        ordering         = ASC / DESC
        limit            = LIMIT ws number

        col_ref          = (table_name "." column_name) / column_name
        table_name       = name
        column_name      = name
        ws               = ~"\s*"i
        wsp              = ~"\s+"i
        name             = ~"[a-zA-Z]\w*"i

        expr             = in_expr / like_expr / between_expr / binary_expr / unary_expr / source_subq / value / string_set
        like_expr        = value wsp LIKE ws string
        in_expr          = (value wsp NOT IN wsp expr) / (value wsp IN wsp expr)
        between_expr     = value BETWEEN wsp value AND wsp value
        binary_expr      = value ws binaryop wsp expr
        unary_expr       = unaryop expr
        value            = parenval / number / boolean / function / col_ref / string
        parenval         = "(" ws expr ws ")"
        function         = (fname ws "(" ws DISTINCT ws arg_list_or_star ws ")") /
                           (fname ws "(" ws arg_list_or_star ws ")")

        arg_list_or_star = arg_list / "*"
        arg_list         = (expr ws "," ws arg_list) / expr
        number           = ~"\d*\.?\d+"i
        string_set       = ws "(" ws string_set_vals ws ")"
        string_set_vals  = (string ws "," ws string_set_vals) / string
        string           = ~"\'.*?\'"i
        fname            = "COUNT" / "SUM" / "MAX" / "MIN" / "AVG" / "ALL"
        boolean          = "true" / "false"
        binaryop         = "+" / "-" / "*" / "/" / "=" / "<>" / ">=" / "<=" / ">" / "<" / ">" / AND / OR
        binaryop_no_andor = "+" / "-" / "*" / "/" / "=" / "<>" / "<=" / ">" / "<" / ">"
        unaryop          = "+" / "-" / "not" / "NOT"

        SELECT   = ws "SELECT"
        FROM     = ws "FROM"
        WHERE    = ws "WHERE"
        AS       = ws "AS"
        AND      = (ws "AND") / (ws "and")
        OR       = (ws "OR") / (ws "or")
        DISTINCT = ws "DISTINCT"
        GROUP    = ws "GROUP"
        ORDER    = ws "ORDER"
        BY       = ws "BY"
        ASC      = ws "ASC"
        DESC     = ws "DESC"
        BETWEEN  = ws "BETWEEN"
        IN       = ws "IN"
        NOT      = ws "NOT"
        HAVING   = ws "HAVING"
        LIMIT    = ws "LIMIT"
        LIKE     = ws "LIKE"
        """
)