import json

from allennlp.data.dataset_readers import SquadReader
from allennlp.data.tokenizers import WordTokenizer

# the class for loaded the instances from lucene and use them for QA
class NearestNeighborQuestionExtractor():
    # uses lucene to extract nearest neghbor instances
    lucene_path = ""

    qTerms = {"which", "what", "where", "who", "whom", "how", "when", "why", "whose"}

    from elasticsearch import Elasticsearch
    es = Elasticsearch(['http://bronte.cs.illinois.edu'], port=8080)

    tokenizer = WordTokenizer()

    def retrieve_best_questions(self):
        pass

    def get_elasticsearch_output(self, question, title: str = ""):
        tokens = [t.text.lower() for t in self.tokenizer.tokenize(question)]

        # filter the wh-terms in the question
        question_terms = [t for t in tokens if t in self.qTerms]

        question_terms_bigrams = []
        for i, t in enumerate(tokens):
            if tokens[i] in self.qTerms or tokens[i + 1] in self.qTerms:
                question_terms_bigrams.append(tokens[i] + " " + tokens[i + 1])

        print(question_terms)
        print(question_terms_bigrams)

        res = self.es.search(index="squad_questions", doc_type="text", body={"query": {"match": {"question": question}}}, size=50)
        print("%d documents found:" % res['hits']['total'])
        output = []
        for doc in res['hits']['hits']:
            question_text = doc['_source']["question"].lower()
            question_tokens = doc['_source']["question_tokens"]
            passage_tokens = doc['_source']["passage_tokens"]
            span_start = doc['_source']["span_start"]
            span_end = doc['_source']["span_end"]
            question_title = doc['_source']["title"]
            print(doc['_source']["question"])
            if question_title != title:
                output.append((question_text, question_tokens, passage_tokens, span_start, span_end))


        # def rank(item):
            # if it contains bigram, should be ranked one,
            # if it contains unigram, should be ranked two,
            # otherwise the rank should be three
        # if item[0]:

        if len(question_terms) > 0:
            sorted([('abc', 121), ('abc', 231), ('abc', 148), ('abc', 221)], key=lambda x: x[1])
        else:
            output = output[0:20]

        return output

# function used to create a json dump, to be indexed in lucene
def createQuestionJson():
    reader = SquadReader(lazy=True)
    outputs = []
    instances = reader.read("/Users/daniel/ideaProjects/allennlp/allennlp/knn/data/squad-train-v1.1.json")
    for i, ins in enumerate(instances):
        if i % 10000 == 1:
            print(f"processesed {i} . . . ")
        # print(ins)
        question_tokens = [t.text for t in ins.fields["question"].tokens]
        passage_tokens = [t.text for t in ins.fields["passage"].tokens]
        question = " ".join(question_tokens)
        passage_title = str(ins.fields["passage_title"].metadata)
        span_start = ins.fields["span_start"].sequence_index
        span_end = ins.fields["span_end"].sequence_index
        output = {"question": question, "title": passage_title, "question_tokens": question_tokens,
                  "passage_tokens": passage_tokens,  "span_start": span_start, "span_end": span_end}
        # print(output)
        outputs.append(output)
    with open("/Users/daniel/ideaProjects/allennlp/allennlp/knn/data/lucene_questions.json", "w") as myfile:
        json.dump(outputs, myfile)

if __name__ == '__main__':
    extractor = NearestNeighborQuestionExtractor()
    extractor.get_elasticsearch_output("When did the Scholastic Magazine of Notre dame begin publishing?")
    # createQuestionJson()

