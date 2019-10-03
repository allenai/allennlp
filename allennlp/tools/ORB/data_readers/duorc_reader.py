import json


def convert_duorc_to_evaluation_server_format(input_file_path: str):
    """
    Convert DuoRC to ORB format.
    """
    output_lines = []
    data = json.load(open(input_file_path))
    for annotation in data:
        passage_str = annotation["plot"]
        output_qa_pairs = []
        for qa_pair in annotation["qa"]:
            if not qa_pair["no_answer"]:
                output = {"question": qa_pair["question"], "answers": [[answer] for answer in qa_pair["answers"]],
                          "dataset": "duorc", "qid": qa_pair["id"]}
                output_qa_pairs.append(output)
        if len(output_qa_pairs) > 0:
            output_instance = {"context": passage_str, "qa_pairs": output_qa_pairs}
            output_format = json.dumps(output_instance)
            output_lines.append("{0}".format(output_format))
    return output_lines
