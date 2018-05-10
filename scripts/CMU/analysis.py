import json

ID_INDEX = 0
G_TRUTH = 1
PREDICTION = 2
PREDICTION_PERCENT = 3
QUESTION_TYPE = 4
QUESTION = 5

CORRECT = "correct"
SIGMOID_ERROR = 'sigmoid-error'
SPAN_DETECTION_ERROR = 'span-detection-error'
WRONG_ENTITY_TYPE = 'wrong-entity-type'
WRONG_ANSWER_CHOSEN = 'wrong-answer-chosen'

ERROR_TYPES = [SIGMOID_ERROR, SPAN_DETECTION_ERROR, WRONG_ENTITY_TYPE, WRONG_ANSWER_CHOSEN]
SIGMOID_INDEX = ERROR_TYPES.index(SIGMOID_ERROR)
SPAN_INDEX = ERROR_TYPES.index(SPAN_DETECTION_ERROR)
WRONG_ENTITY_INDEX = ERROR_TYPES.index(WRONG_ENTITY_TYPE)
WRONG_ANSWER_INDEX = ERROR_TYPES.index(WRONG_ANSWER_CHOSEN)

classifier = {CORRECT: 0, SIGMOID_ERROR: 0, SPAN_DETECTION_ERROR: 0, WRONG_ENTITY_TYPE: 0, WRONG_ANSWER_CHOSEN: 0}


def reset_classifier():
    classifier[CORRECT] = 0
    classifier[SIGMOID_ERROR] = 0
    classifier[SPAN_DETECTION_ERROR] = 0
    classifier[WRONG_ENTITY_TYPE] = 0
    classifier[WRONG_ANSWER_CHOSEN] = 0

def error_type(g_truth, predict):
    error = {CORRECT: 0, SIGMOID_ERROR: 0, SPAN_DETECTION_ERROR: 0, WRONG_ENTITY_TYPE: 0, WRONG_ANSWER_CHOSEN: 0}
    missing_g_truth = g_truth - predict
    missing_predict = predict - g_truth
    if not missing_predict and not missing_predict:
        error[CORRECT] = 1
        return error

    g_truth_substr_matched = set([])
    predict_substr_matched = set([])
    # eliminate span errors
    for predict_entry in missing_predict:
        for g_truth_entry in missing_g_truth:
            if predict_entry in predict_substr_matched or g_truth_entry in g_truth_substr_matched:
                # either of them has been matched previously
                continue

            if predict_entry in g_truth_entry or g_truth_entry in predict_entry:
                g_truth_substr_matched.add(g_truth_entry)
                predict_substr_matched.add(predict_entry)

    missing_predict = missing_predict - predict_substr_matched
    missing_g_truth = missing_g_truth - g_truth_substr_matched

    if predict_substr_matched or g_truth_substr_matched:
        # put SPAN errors
        error[SPAN_DETECTION_ERROR] = 1

    elif len(missing_predict) == len(predict):
        # a combination of sigmoid error and wrong entity error
        error[WRONG_ANSWER_CHOSEN] = 1

    elif missing_predict or missing_g_truth:
        error[SIGMOID_ERROR] = 1

    return error


def convert_to_csv(filename):
    transformed = []
    with open(filename) as f:
        for index, line in enumerate(f, 1):
            elems = line.split('\t')
            assert(len(elems) == 6)
            transformed.append(elems)

    json_file = "%s.json" % (filename)
    with open(json_file, 'w') as f:
        for line in transformed:
            g_truth = set(line[G_TRUTH].split(","))
            predict = set(line[PREDICTION].split(","))
            error = error_type(g_truth, predict)
            try:
                assert(any(error.values()) == True)
            except Exception as e:
                print("Some value should be set", line)
                raise e

            for key in classifier:
                classifier[key] += error[key]

        f.write(json.dumps(classifier))
        reset_classifier()

convert_to_csv('kb_worlds/student_homework/drqa_results.txt.labels')
convert_to_csv('cross_kb_worlds/student_homework/drqa_results.txt.labels')
