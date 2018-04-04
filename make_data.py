import pandas as pd 
import numpy as np 
import pickle
import argparse
import os

def main():
    args = get_args()
    ids = process(args)
    # print(ids)
    make_newdata(ids,args)

def get_args():
	parser = argparse.ArgumentParser()
	home = "/home/ishita/Downloads/maluuba/allennlp/"
	source_logits_file = "data/dev_eval_per_epoch/0/combined_logits.p"
	source_file = "data/active_data/train_small.json"
	target_file = "data/active_newdata/train_small.json"

	parser.add_argument('-s_dev', "--source_logits_file", default=home+source_logits_file)
	parser.add_argument('-t', "--target_file", default=home+target_file)
	parser.add_argument('-s', "--source_file", default=home+source_file)
	parser.add_argument('-p', "--percent", default=1)
	return parser.parse_args()



def getentropy(start,end):
	prob =[]
	for i in range(len(start)):
		p = np.multiply(start[i],end[i:])
		for pr in p:
			prob.append(pr)
	prob = prob/(sum(prob)*1.0)
	entro =[]
	for p in prob:
		if  p!=0.0:
			entro.append(p*np.log2(float(p)))
	if len(entro)==0:
		e = 0
	else:
		e = -1*sum(entro)
	return e

def process(args):
	df_logits = pickle.load(open(args.source_logits_file,'r'))
	ids = list(df_logits.id)
	start = list(df_logits.span_start_logits)
	end = list(df_logits.span_end_logits)
	e =[]
	for i in range(len(start)):
		e.append(getentropy(start[i],end[i]))

	df = pd.DataFrame(list(zip(ids, e)), columns=['ids', 'entropy'])
	dfsorted = df.sort_values('entropy')
	idx = int(args.percent*len(dfsorted)/100)
	ids = dfsorted[:idx].ids
	return ids

def make_newdata(ids,args):
	file = open(args.source_file, 'r')
	f = json.load(file)
	data = f['data']
	data=[]
	for item in data:
		paragraphs = item['paragraphs']
		for p in paragraphs:
			paragraph=[]
			context = p['context']
			qas = p['qas']
			for q in qas:
				id = q[id]
				if id not in reqlist:
					qas.remove(q)
			if len(qas) != 0:
				paragraph.append({"qas":qas,"context":context})
			if len(paragraph)!=0: 
				data.append({"paragraphs":paragraph,"title":"Hello"})
	jsonfinal = {"data": data,"version" : "3"}

	with open(args.target_file, 'w') as fp:
		json.dump(jsonfinal, fp)
if __name__ =="__main__":
    main()