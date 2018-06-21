import sqlparse
from sqlparse.sql import Where, TokenList, Token, Parenthesis
from sqlparse.tokens import Whitespace, Punctuation

parsed = sqlparse.parse("""( SELECT DISTINCT flight.flight_id FROM flight WHERE flight.departure_time = 1200 OR ( flight.departure_time = 1800 AND flight.depature_time = 1900 ) );""")

CONJ = ["AND", "OR"]


def adjust_where(token_list):
    print("adjust where", token_list.tokens)
    if not any([token.value in CONJ for token in token_list.tokens]):
        return token_list

    stack = []
    prefix_token_list = []
    token_list.tokens = [token for token in token_list.tokens if token.ttype != Whitespace] 

    for token in token_list.tokens:
        if token.value not in CONJ:
            if isinstance(token, Parenthesis):
                token = adjust_where(token)
            prefix_token_list.append(token)

        elif token.value in CONJ:
            if len(stack) == 0:
                stack.append(token)
            else:
                # prefix_token_list.insert(0, Token(Punctuation, ")"))
                # prefix_token_list.append(Token(Punctuation, "("))
                prefix_token_list.append(stack.pop())
                stack.append(token)
        
    while len(stack) > 0: 
        print(prefix_token_list)
        #prefix_token_list.insert(0, Token(Punctuation, ")"))
        # prefix_token_list.append(Token(Punctuation, "("))
        prefix_token_list.append(stack.pop())
    
    prefix_token_list.append(token_list.tokens[0])
    prefix_token_list.reverse()
    token_list.value = " ".join([token.value for token in prefix_token_list])
    return token_list
    


def to_prefix(token_list):
    new_token_list = [] 
    for token in token_list.tokens:
        if isinstance(token, Where):
            token = adjust_where(token)
        elif isinstance(token, TokenList):
            token = to_prefix(token)
        new_token_list.append(token)

    token_list.tokens = new_token_list
    token_list.value = " ".join([token.value for token in new_token_list])
    return token_list

stmt = parsed[0]
prefix_stmt = to_prefix(stmt)
print(prefix_stmt.value)

