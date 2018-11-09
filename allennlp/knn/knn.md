## Details of Lucene experiment 

- elasticsearch version used: 6.3.2. After downloading this version, run it from commandline: 
```
./elasticsearch
```

If you want to make the server available over the network, make the proper changes in the config file: 
```
network.host: 0.0.0.0
http.port: 8080
discovery.type: single-node
```

To dump the questions as json, run the python experiment `nearest_questions`. 

Then create an index `squad_questions``: 
```
curl -X PUT "localhost:9200/squad_questions"
```

which should make it available on the following uri: 
```
http://localhost:9200/squad_questions
```

Install `elasticsearch_loader` to move the questions to the elasticsearch: 
```
pip install elasticsearch_loader
```

And copy the files: 
```
elasticsearch_loader --index squad_questions --es-host http://bronte.cs.illinois.edu:8080  --type text json /Users/daniel/ideaProjects/allennlp/allennlp/knn/data/lucene_questions.json
```

if anything happens middle of the experiment and you want to delete the existing index, you can use commandline: 
```
curl -XDELETE 'http://localhost:9200/squad_questions'
```
