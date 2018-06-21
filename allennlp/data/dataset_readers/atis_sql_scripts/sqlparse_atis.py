import sqlparse
from sqlparse.sql import Where, TokenList, Token, Parenthesis, Comparison
from sqlparse.tokens import Whitespace, Punctuation, Keyword

parsed = sqlparse.parse("""( SELECT DISTINCT flight.flight_id FROM flight WHERE flight.departure_time = 1200 AND ( flight.arrival_time = 1234 OR flight.dept_time = 4321 ) ) ;""")

CONJ = ["AND", "OR", "="]


def prefix_condition(token_list, start=None, end=None):
    print('updated')
    print("prefix condition", token_list.tokens)
    '''
    if not any([token.value in CONJ for token in token_list.flatten()]):
        return token_list
    '''

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
        print('stack:', stack)
        print('prefix_token_list', prefix_token_list)

        
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

stmt = parsed[0]
prefix_stmt = to_prefix(stmt)
print(prefix_stmt.value)

