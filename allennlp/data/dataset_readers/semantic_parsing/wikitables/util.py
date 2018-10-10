from typing import Dict

def parse_example_line(lisp_string: str) -> Dict:
    """
    Training data in WikitableQuestions comes with examples in the form of lisp strings in the format:
        (example (id <example-id>)
                 (utterance <question>)
                 (context (graph tables.TableKnowledgeGraph <table-filename>))
                 (targetValue (list (description <answer1>) (description <answer2>) ...)))

    We parse such strings and return the parsed information here.
    """
    id_piece, rest = lisp_string.split(') (utterance "')
    example_id = id_piece.split('(id ')[1]
    question, rest = rest.split('") (context (graph tables.TableKnowledgeGraph ')
    table_filename, rest = rest.split(')) (targetValue (list')
    target_value_strings = rest.strip().split("(description")
    target_values = []
    for string in target_value_strings:
        string = string.replace(")", "").replace('"', '').strip()
        if string != "":
            target_values.append(string)
    return {'id': example_id,
            'question': question,
            'table_filename': table_filename,
            'target_values': target_values}
