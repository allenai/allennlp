import csv

ID_INDEX = 0
G_TRUTH = 1
PREDICTION = 2
PREDICTION_PERCENT = 3
QUESTION_TYPE = 4
QUESTION = 5

def convert_to_csv(filename):
    transformed = []
    with open(filename) as f:
        for index, line in enumerate(f, 1):
            elems = line.split('\t')
            assert(len(elems) == 6)
            transformed.append(elems)
            if index > 500:
                break

    csv_corr_file = "%s-correct.csv" % (filename)
    csv_wrong_file = "%s-wrong.csv" % (filename)
    csv_file = "%s.csv" % (filename)
    wrong = 0
    right = 0
    with open(csv_corr_file, 'w') as f_corr:
        with open(csv_wrong_file, 'w') as f_wrong:
            with open(csv_file, 'w') as f:
                corr_csv = csv.writer(f_corr, quoting=csv.QUOTE_ALL)
                wrong_csv = csv.writer(f_wrong, quoting=csv.QUOTE_ALL)
                all_csv = csv.writer(f, quoting=csv.QUOTE_ALL)
                for line in transformed:
                    g_truth = set(line[G_TRUTH].split(","))
                    predict = set(line[PREDICTION].split(","))
                    if g_truth == predict:
                        corr_csv.writerow(line)
                        value = "correct"
                        right += 1
                    else:
                        wrong_csv.writerow(line)
                        value = "wrong"
                        wrong += 1
                    line.append(value)
                    all_csv.writerow(line)
        print("Wrong-Right", wrong, right)

convert_to_csv('kb_worlds/student_homework/drqa_results.txt.labels')
convert_to_csv('cross_kb_worlds/student_homework/drqa_results.txt.labels')
