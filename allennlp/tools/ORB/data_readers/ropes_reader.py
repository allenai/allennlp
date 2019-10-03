import json


def convert_ropes_to_evaluation_server_format(input_file_path: str):
    """
    Convert ROPES to ORB format.
    """
    output_lines = []
    data = json.load(open(input_file_path))
    for annotation in data["data"]:
        for paragraph in annotation["paragraphs"]:
            passage_str = "{0} [SEP] {1}".format(paragraph["background"], paragraph["situation"])
            output_qa_pairs = []
            for qa_pair in paragraph["qas"]:
                answers = [[answer["text"]] for answer in qa_pair["answers"]]
                output = {"question": qa_pair["question"], "answers": answers,
                          "dataset": "squad", "qid": qa_pair["id"]}
                output_qa_pairs.append(output)
            output_instance = {"context": passage_str, "qa_pairs": output_qa_pairs}
            output_format = json.dumps(output_instance)
            output_lines.append("{0}".format(output_format))
    return output_lines
