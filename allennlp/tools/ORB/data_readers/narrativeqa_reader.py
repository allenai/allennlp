import csv
import json
import uuid


def convert_narrativeqa_to_evaluation_server_format(stories_csv: str, qapairs_csv: str):
    """
    Convert NarrativeQA to ORB format.
    """
    story_id_qa_map = {}
    summaries = csv.reader(open(stories_csv), delimiter=',', quotechar='"')
    for summary in summaries:
        document_id, _, story_text, _ = summary
        story_id_qa_map[document_id] = story_text

    data = csv.reader(open(qapairs_csv), delimiter=',', quotechar='"')
    output_instances_per_split = {"train": {}, "valid": {}, "test": {}}

    for annotation in data:
        if annotation[0] == 'document_id':
            continue
        document_id, split_type, question, answer1, answer2, _, _, _ = annotation

        current_instance = {"question": question, "answers": [[answer1], [answer2]], "dataset": "narrativeqa",
                            "qid": str(uuid.uuid1())}

        if document_id in output_instances_per_split[split_type]:
            output_instances_per_split[split_type][document_id]["qa_pairs"].append(current_instance)
        else:
            output_instance = {"context": story_id_qa_map[document_id], "qa_pairs": [current_instance]}
            output_instances_per_split[split_type][document_id] = output_instance

    output_lines_all_splits = []
    for split in ["train", "test", "valid"]:
        output_lines = []
        for _, instance in output_instances_per_split[split].items():
            output_format = json.dumps(instance)
            output_lines.append("{0}".format(output_format))
        output_lines_all_splits.append(output_lines)

    return output_lines_all_splits
