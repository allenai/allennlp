import pandas as pd 
import numpy as np 
import pickle
import argparse
import os

def main():
    args = get_args()
    ids = process(args)
    make_newdata(ids,args)

def get_args():
	parser = argparse.ArgumentParser()
	home = os.path.expanduser("~/Downloads//maluuba/allennlp/")
	source_dev_file = "active/input/result_dev.json"
	source_eval_file = "active/input/result_eval.json"
	source_file = "data/active_data/train_small.json"
	target_file = "data/active_newdata/train_small.json"

	parser.add_argument('-s_dev', "--source_dev_file", default=source_dev_file)
	parser.add_argument('-s_eval', "--source_eval_file", default=source_eval_file)
	parser.add_argument('-t', "--target_file", default=target_file)
	parser.add_argument('-s', "--source_file", default=source_file)
	parser.add_argument('-p', "--percent", default=1)
	return parser.parse_args()



def getentropy(start,end):
	prob =[]
	for i in range(len(start)):
		p = np.multiply(start[i],end[i:])			
		prob +=[pr for pr in p]
	prob = prob/(sum(prob)*1.0)
	entro =[]
	for p in prob:
		entro.append(p*np.log2(float(p)))
	e = -1*sum(entro)
	return e

def process(args):
	df_dev = pd.read_json(args.source_dev_file)
	df_eval = pd.read_json(args.source_eval_file)
	ids = list(df_dev.columns)
	ids = ids[:-1]
	start = df_eval.yp
	end = df_eval.yp2

	e =[]
	for i in range(len(start)):
		e.append(getentropy(start[i][0],end[i][0]))

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