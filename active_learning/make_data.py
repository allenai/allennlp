import argparse
import json
import pickle
import torch
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    # home = "/home/ishita/Downloads/maluuba/allennlp/"
    home = "./"
    source_logits_file = "data/combined_logits.p"
    source_file = "data/newsqa/dev_dump.json"
    target_file = "data/newsqa/top_dev_dump.json"

    parser.add_argument('-s_dev', "--source_logits_file", default=home + source_logits_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)
    parser.add_argument('-s', "--source_file", default=home + source_file)
    parser.add_argument('-p', "--percent", default=1)
    parser.add_argument('-g', "--gpu", default=0)
    return parser.parse_args()

def process(args):
    df_logits = pickle.load(open(args.source_logits_file, 'rb'))
    ids = list(df_logits.id)
    start = list(df_logits.span_start_logits)
    end = list(df_logits.span_end_logits)
    e = []
    print(len(start))
    # print(start[0])
    # print(end[0])
    dtype = torch.cuda.FloatTensor
    if (args.gpu == 0):
        dtype = torch.FloatTensor

    for i in range(len(start)):
    # for i in range(50):

        # print(ids[i])
        e.append(get_scores_using_logits(start[i], end[i], dtype))
        # e.append(get_scores_using_softmax(start[i], end[i], dtype))
    #
    print("Sorting now")
    df = pd.DataFrame(list(zip(ids, e)), columns=['ids', 'entropy'])
    # print(df)
    dfsorted = df.sort_values('entropy')
    print(dfsorted)
    idx = int(args.percent * len(dfsorted) / 100)
    # print(idx)
    ids = dfsorted[:idx].ids
    # print(ids)
    return ids


def get_scores_using_logits(start, end, dtype):
    start = torch.from_numpy(np.array([start]).transpose())
    end = torch.from_numpy(np.array([end]))

    start = start.type(dtype)
    end = end.type(dtype)
    s_p = start
    e_p = end

    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)

    score_sum = torch.sum(score_mul)

    score_mul = score_mul / score_sum
    score_mul[score_mul<=0]=1

    y = torch.log(score_mul)

    y = score_mul * y

    # print("total ")
    total = -1 * torch.sum(y)
    # print(total)

    return total


'''
'''
def get_scores_using_softmax(start, end, dtype):
    start = torch.from_numpy(np.array([start]).transpose())
    end = torch.from_numpy(np.array([end]))

    start = start.type(dtype)
    end = end.type(dtype)

    s_p = torch.nn.functional.softmax(torch.autograd.Variable(start), dim=0).data
    e_p = torch.nn.functional.softmax(torch.autograd.Variable(end), dim=1).data

    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)

    score_sum = torch.sum(score_mul)

    score_mul = score_mul / score_sum
    score_mul[score_mul==0]=1

    y = torch.log(score_mul)

    y = score_mul * y

    # print("total ")
    total = -1 * torch.sum(y)
    # print(total)

    return total


def make_newdata(ids, args):

    ids = list(ids)

    file = open(args.source_file, 'rb')
    f = json.load(file)
    data = f['data']
    new_data = []
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            paragraph = []
            context = p['context']
            qas = p['qas']
            qas_new = []
            for q in qas:
                id = q['id']
                if id in ids:
                    # print("FOUND IT")
                    print(q['id'])
                    qas_new.append(q)
            if len(qas_new) != 0:
                paragraph.append({"context": context, "qas": qas_new})
            if len(paragraph) != 0:
                new_data.append({"paragraphs": paragraph, "title": "Hello"})
    jsonfinal = {"data": new_data, "version": "4"}

    with open(args.target_file, 'w') as fp:
        json.dump(jsonfinal, fp)


def main():
    args = get_args()
    ids = process(args)
    # print(ids)
    # print(ids)
    # ids = []
    make_newdata(ids, args)


if __name__ == "__main__":
    main()
