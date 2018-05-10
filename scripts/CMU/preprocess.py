import json
import os
import re
import uuid

total = 0.0
ignored = 0.0


class Preprocessor(object):
    def __init__(self):
        pass

    def _word_cleanup(self, lookups):
        result = []

        for word in lookups:
            try:
                assert word[0] == '['
            except AssertionError as e:
                print word
                raise e
            cleaned = ""
            for char in word[1:]:
                if char == '(' or char == ']':
                    break
                cleaned += char
            result.append(cleaned)

        return result

    def _get_indices(self, line, lookups):
        result = []
        for lookup in lookups:
            start = line.find(lookup)
            end = start + len(lookup)
            result.append((start, end))

        return result

    def _line_cleanup(self, line):
        ent_pattern = r"\[[a-zA-Z0-9 ]+\([a-zA-Z ]+\[\d+\]\)\]?"
        ref_pattern = r"\[[a-zA-Z0-9 ]+\]"
        entities = re.findall(ref_pattern + "|" + ent_pattern, line)
        entities_clean = self._word_cleanup(entities)
        for index, entity in enumerate(entities):
            line = line.replace(entities[index], entities_clean[index])

        return line.strip()

    def _remove_brace(self, line):
        open_braces = [index for index, c in enumerate(line) if c == '(']
        close_braces = [index for index, c in enumerate(line) if c == ')']
        ignore_ranges = list(zip(open_braces, close_braces))
        next_index = 0
        result = ""

        for ignore_range in ignore_ranges:
            result += line[next_index:ignore_range[0]]
            next_index = ignore_range[1] + 1

        result += line[next_index:]
        result = result.replace("', '", "\t")
        return result.strip("'").split("\t")

    def generate_qas(self, question, answers, context, qas=[]):
        qa = {"question": question, "answers": [], "id": str(uuid.uuid4())}
        for ans in answers:
            span_start = context.find(ans)
            if span_start == -1:
                continue
            entry = {
                'answer_start': span_start,
                'text': ans,
            }
            qa['answers'].append(entry)

        result = (len(qa['answers']) != 0)
        if result:
            qas.append(qa)

        return result

    def preprocess(self, filename):
        result = []
        context_so_far = ""
        entry = {"title": filename, "paragraphs": []}
        global ignored
        global total
        with open(filename) as f:
            context_changed = True
            qas = []
            paragraphs = entry["paragraphs"]
            para = {}
            for line in f:
                if "question_" in line:
                    if context_changed:
                        qas = []

                    line = line.split('\t')
                    question = self._line_cleanup(line[0])
                    answers = self._remove_brace(line[1])

                    context_so_far = " ".join(result)
                    updated = self.generate_qas(question, answers, context_so_far, qas)
                    context_changed = False
                else:
                    if not context_changed:
                        if len(qas) != 0:
                            para["context"] = context_so_far
                            para["qas"] = qas
                            paragraphs.append(para)
                        else:
                            ignored += 1
                            print("Ignoring paragraph", qas)
                        para = {}
                        total += 1

                    context_changed = True
                    line = self._line_cleanup(line) + "."
                    result.append(line)
        return entry


def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        json.dump(json_obj, f)


if __name__ == "__main__":
    p = Preprocessor()
    path = "/Users/prasoon/Desktop/train"
    files = os.listdir(path)
    student_train_json = {'data': []}
    bug_train_json = {'data': []}
    dept_train_json = {'data': []}
    meet_train_json = {'data': []}
    shop_train_json = {'data': []}

    for index, each_file in enumerate(files):
        if not os.path.isfile(each_file):
            print("Dir", each_file)
            inner_files = os.listdir(path + "/" + each_file)
            for filename in inner_files:
                if filename.startswith('student'):
                    train_json = student_train_json
                elif filename.startswith('bug'):
                    train_json = bug_train_json
                elif filename.startswith('department'):
                    train_json = dept_train_json
                elif filename.startswith('meetings'):
                    train_json = meet_train_json
                elif filename.startswith('shopping'):
                    train_json = shop_train_json
                else:
                    print("Ignored file", filename)
                    continue

                if len(train_json['data']) > 100:
                    continue

                real_path = path + "/" + each_file + "/" + filename
                print("Preprocessing:", index, filename)
                train_json['data'].append(p.preprocess(real_path))

    path += "/"
    print(ignored, "/", total)
    save_json(student_train_json, path + 'final/student.json')
    save_json(bug_train_json, path + 'final/bug.json')
    save_json(dept_train_json, path + 'final/department.json')
    save_json(meet_train_json, path + 'final/meetings.json')
    save_json(shop_train_json, path + 'final/shopping.json')
    train = {'data': student_train_json['data'] + bug_train_json['data'] + dept_train_json['data'] +
            meet_train_json['data']}
    dev = shop_train_json

    save_json(train, path + 'final/train.json')
    save_json(dev, path + 'final/dev.json')
