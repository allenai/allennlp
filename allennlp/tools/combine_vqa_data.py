import json


def combine_questions_and_answers(
    question_filename: str, answer_filename: str, result_filename: str
):
    with open(question_filename) as question_file:
        questions = json.load(question_file)
    with open(answer_filename) as answer_file:
        answers = json.load(answer_file)

    annotations_by_question_id = {}
    for annotation in answers["annotations"]:
        annotations_by_question_id[annotation["question_id"]] = annotation

    for question_dict in questions["questions"]:
        question_annotations = annotations_by_question_id[question_dict["question_id"]]
        question_dict["answers"] = question_annotations["answers"]

    with open(result_filename, "w") as result_file:
        json.dump(questions, result_file, indent=2)


if __name__ == "__main__":
    combine_questions_and_answers(
        "annotations/v2_OpenEnded_mscoco_train2014_questions.json",
        "annotations/v2_mscoco_train2014_annotations.json",
        "annotations/combined_train2014.json",
    )
    combine_questions_and_answers(
        "annotations/v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations/v2_mscoco_val2014_annotations.json",
        "annotations/combined_val2014.json",
    )
