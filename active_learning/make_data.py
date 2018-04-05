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

        print(ids[i])
        # e.append(get_scores_using_logits(start[i], end[i], dtype))
        e.append(get_scores_using_softmax(start[i], end[i], dtype))
    #
    # df = pd.DataFrame(list(zip(ids, e)), columns=['ids', 'entropy'])
    # dfsorted = df.sort_values('entropy')
    # idx = int(args.percent * len(dfsorted) / 100)
    # ids = dfsorted[:idx].ids
    # return ids


def get_scores_using_logits(start, end, dtype):
    start = torch.from_numpy(np.array([start]).transpose())
    end = torch.from_numpy(np.array([end]))

    start = start.type(dtype)
    end = end.type(dtype)
    s_p = start
    e_p = end
    # s_p = torch.nn.functional.softmax(torch.autograd.Variable(start), dim=0).data
    # e_p = torch.nn.functional.softmax(torch.autograd.Variable(end), dim=1).data
    # print(s_p, e_p)
    # print(s_p.shape, e_p.shape)
    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)
    # print(score_mul)
    score_sum = torch.sum(score_mul)
    # print(score_sum)
    score_mul = score_mul / score_sum
    score_mul[score_mul<=0]=1
    # print(score_mul)
    y = torch.log(score_mul)
    # print("log")
    # print(y)
    y = score_mul * y
    # print("mul")
    # print(y)
    print("total ")
    print(-1 * torch.sum(y))


'''
'''
def get_scores_using_softmax(start, end, dtype):
    start = torch.from_numpy(np.array([start]).transpose())
    end = torch.from_numpy(np.array([end]))

    start = start.type(dtype)
    end = end.type(dtype)
    # s_p = start
    # e_p = end
    s_p = torch.nn.functional.softmax(torch.autograd.Variable(start), dim=0).data
    e_p = torch.nn.functional.softmax(torch.autograd.Variable(end), dim=1).data
    # print(s_p, e_p)
    # print(s_p.shape, e_p.shape)
    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)
    # print(score_mul)
    score_sum = torch.sum(score_mul)
    # print(score_sum)
    score_mul = score_mul / score_sum
    score_mul[score_mul==0]=1
    # print(score_mul)
    y = torch.log(score_mul)
    # print("log")
    # print(y)
    y = score_mul * y
    # print("mul")
    # print(y)
    print("total ")
    print(-1 * torch.sum(y))

    '''
    prob = []
    for i in range(len(start)):
        p = np.multiply(start[i], end[i:])
        for pr in p:
            prob.append(pr)
    prob = prob / (sum(prob) * 1.0)
    entro = []
    for p in prob:
        if p != 0.0:
            entro.append(p * np.log2(float(p)))
    if len(entro) == 0:
        e = 0
    else:
        e = -1 * sum(entro)
    '''
    return 0


def make_newdata(ids, args):
    file = open(args.source_file, 'rb')
    f = json.load(file)
    data = f['data']
    data = []
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            paragraph = []
            context = p['context']
            qas = p['qas']
            for q in qas:
                id = q[id]
                if id not in reqlist:
                    qas.remove(q)
            if len(qas) != 0:
                paragraph.append({"qas": qas, "context": context})
            if len(paragraph) != 0:
                data.append({"paragraphs": paragraph, "title": "Hello"})
    jsonfinal = {"data": data, "version": "3"}

    with open(args.target_file, 'w') as fp:
        json.dump(jsonfinal, fp)


def main():
    args = get_args()
    ids = process(args)
    # print(ids)
    # make_newdata(ids, args)


if __name__ == "__main__":
    main()
    '''
    x = torch.FloatTensor(torch.randn(2,4))
    a = np.array([[1,2,3]])
    b = np.array([[4,5,6]])
    a = torch.from_numpy(a.transpose())
    b = torch.from_numpy(b)
    # b = torch.transpose(b)
    x = torch.matmul(a, b)
    # print(a, b, x)
    print(x)
    x = x.type(torch.FloatTensor)
    # print(torch.min(x))
    # x = x - (torch.min(x))
    # print(x)

    x_sum = torch.sum(x)
    x = x/x_sum
    y = torch.log(x)
    print(y)

    y = x * y
    print(y)
    z = torch.triu(y)
    print(z)
    print(-1 * torch.sum(z))
    
    '''
    # z_sum = torch.sum(z)
    # print(z_sum)
    # print(z / z_sum)
