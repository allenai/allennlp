import argparse
import sqlparse
from sqlparse.sql import Where, TokenList, Token, Parenthesis, Comparison, Identifier
from sqlparse.tokens import Whitespace, Punctuation, Keyword, Literal

from itertools import chain, repeat

CONJ = ["AND", "OR", "="]

def rename(token: Token) -> Token:
    """
    Takes a Token and annotates strings and nums, changes ``.`` operator in column refs
    to ``_``.
    """
    if not isinstance(token, TokenList) and token.ttype == Literal.Number.Integer:
        token.value = "num:" + token.value
        return token

    if token.ttype == Punctuation and token.value == ".":
        token.value = "_"
        return token
    
    if isinstance(token, Identifier):
        token.tokens = [rename(token) for token in token.tokens]
        token.tokens[0].value = "string:" +  token.tokens[0].value
        token.value = "".join([token.value for token in token.tokens])
        return token

    elif isinstance(token, TokenList):
        token.tokens = [rename(token) for token in token.tokens]
        token.value = "".join([token.value for token in token.tokens])
        return token

    else:
        return token


def prefix_condition(token_list: TokenList, start=None, end=None) -> TokenList:
    """
    Takes a TokenList and reorders its tokens such that the tokens are in prefix order
    """

    stack = []
    prefix_token_list = []
    token_list.tokens = [token for token in token_list.tokens if token.ttype != Whitespace]

    for token in token_list.tokens[start:end]:
        if token.value not in CONJ:
            if isinstance(token, Parenthesis):
                token = prefix_condition(token, 1, -1)

            if isinstance(token, Comparison) and isinstance(token, TokenList):
                 token = prefix_condition(token) 

            prefix_token_list.append(token)
            prefix_token_list.append(Token(Whitespace, " "))

        elif token.value in CONJ:
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
    

def to_logical(token_list: TokenList) -> TokenList:
    """
    Convert a TokenList representing a parenthesized SELECT statement from SQL to (lisp-like) logical form
    so it can be input into into the NLTK parser.
    """
    token_list.tokens = [token for token in token_list.tokens if token.ttype != Whitespace]
    new_token_list = []
    

    for idx, token in enumerate(token_list.tokens):
        if isinstance(token, Where):
            token = prefix_condition(token, 1, None)
            token.tokens.insert(0, Token(Keyword, "WHERE"))
            token.value = "(" + "".join([where_token.value for where_token in token.tokens]) + ")"

        new_token_list.append(token)
    
    new_token_list = [new_token_list[0]] + list(chain(*zip(new_token_list[1:len(new_token_list)-2], repeat(Token(Whitespace, " "))))) + new_token_list[-2:]
    
    from_idx = next(idx for idx, token in enumerate(new_token_list) if (token.ttype == Keyword and token.value == "FROM"))
    new_token_list.insert(from_idx, Token(Punctuation, "("))

    where_idx = next(idx for idx, token in enumerate(new_token_list) if isinstance(token, Where))
    new_token_list.insert(where_idx - 1, Token(Punctuation, ")"))

    token_list.tokens = new_token_list
    token_list.value = "".join([token.value for token in new_token_list])
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
            renamed_stmt = rename(stmt)
            print(renamed_stmt.value)
            prefix_stmt = to_logical(renamed_stmt)
            prefix_stmt.value = prefix_stmt.value.rstrip("; ") + "\n"
            print(prefix_stmt.value)
            processed_file.write(prefix_stmt.value)
