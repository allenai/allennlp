import json
from typing import Any, Tuple, Dict


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return [str(answer["number"])]
    elif "spans" in answer and answer["spans"]:
        return answer["spans"]
    elif "date" in answer:
        return ["{0} {1} {2}".format(answer["date"]["day"], answer["date"]["month"], answer["date"]["year"])]
    else:
        raise ValueError(f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}")


def convert_drop_to_evaluation_server_format(input_file_path: str):
    """
    Convert DROP to ORB format.
    """
    output_lines = []
    data = json.load(open(input_file_path))
    for passage_id, annotation in data.items():
        passage_str = annotation["passage"]
        output_qa_pairs = []
        for qa_pair in annotation["qa_pairs"]:
            answer_spans = answer_json_to_strings(qa_pair["answer"])
            additional_answer_spans = [answer_json_to_strings(answer) for answer in qa_pair["validated_answers"]]
            output = {"question": qa_pair["question"], "answers": [answer_spans] + additional_answer_spans,
                      "dataset": "drop", "qid": qa_pair["query_id"]}
            output_qa_pairs.append(output)
        output_instance = {"context": passage_str, "qa_pairs": output_qa_pairs}
        output_format = json.dumps(output_instance)
        output_lines.append("{0}".format(output_format))

    return output_lines
