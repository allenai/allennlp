import pandas as pd 
import os
import pickle
directory = "/home/ishita/Downloads/maluuba/allennlp/data/dev_eval_per_epoch"
final = []
for subdir, dirs, files in os.walk(directory):
	for filename in files:
		if filename.endswith(".json"):
			file = open(os.path.join(subdir,filename),'r')
			file = file.read()
			file = file.split('][')
			for i in range(len(file)):
				ele = file[i]
				if i!=len(file)-1:
					ele = ele+']'
				if i!=0:
					ele = '['+ele
				df = pd.read_json(ele)
				final.append(df)
result = pd.concat(final)
f = open('combined_logits.p','wb')
pickle.dump(result,f)