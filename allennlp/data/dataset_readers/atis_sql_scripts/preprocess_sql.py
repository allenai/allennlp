import argparse
import sqlparse
from sqlparse.sql import Where, TokenList, Token, Parenthesis, Comparison
from sqlparse.tokens import Whitespace, Punctuation, Keyword

CONJ = ["AND", "OR", "="]

def prefix_condition(token_list, start=None, end=None):
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
    token_list.value = " ".join([token.value for token in prefix_token_list])
    return token_list
    

def to_prefix(token_list):
    new_token_list = [] 
    for token in token_list.tokens:
        if isinstance(token, Where):
            token = prefix_condition(token, 1, None)
            token.tokens.insert(0, Token(Keyword, "WHERE"))
            token.value = " ".join([where_token.value for where_token in token.tokens])
        elif isinstance(token, TokenList):
            token = to_prefix(token)
        new_token_list.append(token)

    token_list.tokens = new_token_list
    token_list.value = " ".join([token.value for token in new_token_list])
    return token_list


parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
args = parser.parse_args()

with open(args.in_file, 'r') as sql_file:
    with open(args.out_file, 'w') as processed_file:
        for line in sql_file:
            parsed = sqlparse.parse(line)
            stmt = parsed[0]
            prefix_stmt = to_prefix(stmt)
            processed_file.write(prefix_stmt.value + "\n")



