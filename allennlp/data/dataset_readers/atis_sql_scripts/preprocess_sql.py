import argparse
import sqlparse
from sqlparse.sql import Where, TokenList, Token, Parenthesis, Comparison, Identifier
from sqlparse.tokens import Whitespace, Punctuation, Keyword, Literal, String

from itertools import chain, repeat

OPS = ["AND", "OR", "=", ">", "<", ">=", "<=", "IN"] 

def annotate(token: Token) -> Token:
    """
    Takes a Token and annotates strings and nums, changes ``.`` operator in column refs
    to ``_``.
    """
    if not isinstance(token, TokenList) and token.ttype == Literal.Number.Integer:
        token.value = "num:" + token.value
        return token

    if not isinstance(token, TokenList) and token.ttype == String.Single:
        token.value = "string:" + token.value.replace(" ", "_")

    if token.ttype == Punctuation and token.value == ".":
        token.value = "_"
        return token
    
    if isinstance(token, Identifier):
        token.tokens = [annotate(token) for token in token.tokens]
        token.tokens[0].value = "string:" +  token.tokens[0].value
        token.value = "".join([token.value for token in token.tokens])
        return token

    elif isinstance(token, TokenList):
        token.tokens = [annotate(token) for token in token.tokens]
        token.value = "".join([token.value for token in token.tokens])
        return token

    else:
        return token


def prefix_condition(token_list: TokenList, start=None, end=None) -> TokenList:
    """
    Takes a TokenList, start, end indices that indicate the range in the token list
    that needs to be readjusted and reorders its tokens such that the tokens are in prefix order
    """

    stack = []
    prefix_token_list = []
    token_list.tokens = [token for token in token_list.tokens if token.ttype != Whitespace]

    if isinstance(token_list, Parenthesis) and token_list.tokens[1].value == "SELECT":
        print(token_list.value)
        return preprocess_select(token_list)

   
    '''
    in_idx = next( (idx for idx, token in enumerate(token_list.tokens) if (token.ttype == Keyword and token.value == "IN")), None)
    if in_idx:
        token_list.tokens = [token_list.tokens[in_idx]] + \
                            [token_list.tokens[in_idx-1]] + \
                            [preprocess_select(token_list.tokens[in_idx+1])]

        token_list.value = "(" + " ".join([token.value for token in token_list.tokens]) + ")"
        return token_list
    '''

    for token in token_list.tokens[start:end]:
        if token.value not in OPS:
            if isinstance(token, Parenthesis):
                token = prefix_condition(token, 1, -1)

            if isinstance(token, Comparison) and isinstance(token, TokenList):
                 token = prefix_condition(token) 

            prefix_token_list.append(token)
            prefix_token_list.append(Token(Whitespace, " "))

        elif token.value in OPS:
            if len(stack) == 0:
                stack.append(token)
            else:
                prefix_token_list.append(stack.pop())
                prefix_token_list.insert(0, Token(Punctuation, ")"))
                prefix_token_list.append(Token(Punctuation, "("))
                stack.append(token)
        
    while len(stack) > 0: 
        prefix_token_list.append(stack.pop())
        prefix_token_list.insert(0, Token(Punctuation, ")"))
        prefix_token_list.append(Token(Punctuation, "("))
    
    prefix_token_list.reverse()
    token_list.tokens = prefix_token_list
    token_list.value = "".join([token.value for token in prefix_token_list])
    return token_list
    

def preprocess_where(where_token: TokenList) -> TokenList:
    """
    Convert a TokenList that represents a Keyword of the Where type to (lisp-like) logical form
    """
    where_token.tokens = [token for token in where_token.tokens if token.ttype != Whitespace]
    
    where_token = prefix_condition(where_token, 1, None)
    where_token.tokens.insert(0, Token(Keyword, "WHERE"))
    where_token.value = "(" + " ".join([token.value for token in where_token.tokens]) + ")"

    return where_token

def preprocess_select(token_list: TokenList) -> TokenList:
    """
    Convert a TokenList representing a parenthesized SELECT statement from SQL to (lisp-like) logical form
    so it can be input into into the NLTK parser.
    """
    token_list.tokens = [token for token in token_list.tokens if token.ttype != Whitespace]

    distinct_idx = next( (idx for idx, token in enumerate(token_list.tokens) if (token.ttype == Keyword and token.value == "DISTINCT")), None)
    if distinct_idx:
        del token_list.tokens[distinct_idx]
        token_list.tokens[distinct_idx - 1].value = "SELECT_DISTINCT" 


    where_idx = next( (idx for idx, token in enumerate(token_list.tokens) if isinstance(token, Where)), None)
    if where_idx:
        token_list.tokens[where_idx] = preprocess_where(token_list.tokens[where_idx])

    from_idx = next(idx for idx, token in enumerate(token_list.tokens) if (token.ttype == Keyword and token.value == "FROM"))

    token_list.value = "(" + " ".join([token.value for token in token_list.tokens[1:from_idx]]) + \
            " (" + " ".join([token.value for token in token_list.tokens[from_idx:from_idx + 2]]) + ") " + \
            " ".join([token.value for token in token_list.tokens[from_idx + 2:]]) + ")"

    return token_list


parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
args = parser.parse_args()

with open(args.in_file, 'r') as sql_file:
    with open(args.out_file, 'w') as processed_file:
        for line in sql_file:
            parsed = sqlparse.parse(line)
            stmt = parsed[0].tokens[0]
            annotated_stmt = annotate(stmt)
            prefix_stmt = preprocess_select(annotated_stmt)
            prefix_stmt.value = prefix_stmt.value.rstrip("; ") + "\n"
            print(prefix_stmt.value)
            print("\n\n")
            processed_file.write(prefix_stmt.value)
