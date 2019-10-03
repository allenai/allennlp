import csv
import json
import uuid


def convert_newsqa_to_evaluation_server_format(stories_csv_per_split_path: str):
    """
    Convert NewsQA to ORB format.
    """
    data = csv.reader(open(stories_csv_per_split_path), delimiter=',', quotechar='"')
    output_instances = {}

    for annotation in data:
        if annotation[0] == 'story_id':
            continue
        story_id, story_text, question, answer_token_ranges = annotation

        candidate_answer_offsets = answer_token_ranges.split(",")
        answer_list = []
        for answer_char_offset in candidate_answer_offsets:
            start_index, end_index = list(map(int, answer_char_offset.split(":")))
            context_tokens = story_text.split(" ")
            answer_list.append([" ".join(context_tokens[start_index:end_index])])

        current_instance = {"question": question, "answers": answer_list,
                            "dataset": "newsqa", "qid": str(uuid.uuid1())}

        if story_id in output_instances:
            output_instances[story_id]["qa_pairs"].append(current_instance)
        else:
            output_instance = {"context": story_text, "qa_pairs": [current_instance]}
            output_instances[story_id] = output_instance

    output_lines = []
    for _, instance in output_instances.items():
        output_format = json.dumps(instance)
        output_lines.append("{0}".format(output_format))

    return output_lines
