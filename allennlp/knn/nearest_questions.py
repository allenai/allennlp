import json

from allennlp.data.tokenizers import WordTokenizer
from diskcache import Cache
import json

# the class for loaded the instances from lucene and use them for QA
class NearestNeighborQuestionExtractor():
    qTerms = {"which", "what", "where", "who", "whom", "how", "when", "why", "whose"}

    cache = Cache('nearest_questions.cache')

    # uses lucene to extract nearest neghbor instances
    from elasticsearch import Elasticsearch
    es = Elasticsearch(['http://bronte.cs.illinois.edu'], port=8080)

    tokenizer = WordTokenizer()

    def retrieve_best_questions(self, question, title: str = "", topK: int = 5):
        key = str.encode(question + title + str(topK))
        if key in self.cache:
            output_bytes = self.cache[key]
            return json.loads(output_bytes.decode())
        else:
            output = self.retrieve_best_questions_from_elastic_search(question, title, topK)
            self.cache.add(key, str.encode(json.dumps(output)))
            return output

    def retrieve_best_questions_from_elastic_search(self, question, title, topK: int = 5):
        tokens = [t.text.lower() for t in self.tokenizer.tokenize(question)]

        # filter the wh-terms in the question
        question_terms = [t for t in tokens if t in self.qTerms]

        question_terms_bigrams = []
        for i, t in enumerate(tokens):
            if i + 1 < len(tokens) and (tokens[i] in self.qTerms or tokens[i + 1] in self.qTerms):
                question_terms_bigrams.append(tokens[i] + " " + tokens[i + 1])

        # print(question_terms)
        # print(question_terms_bigrams)

        def rank(question_text, score):
            # if it contains bigram, should be ranked one,
            # if it contains unigram, should be ranked two,
            # otherwise the rank should be three
            if any(x in question_text for x in question_terms_bigrams):
                return 200
            elif any(x in question_text for x in question_terms):
                return 100
            else:
                return score # lucene score

        res = self.es.search(index="squad_questions", doc_type="text", body={"query": {"match": {"question": question}}}, size=50)
        # print("%d documents found:" % res['hits']['total'])
        output = []
        for doc in res['hits']['hits']:
            question_text = doc['_source']["question"].lower()
            question_tokens = doc['_source']["question_tokens"]
            passage_tokens = doc['_source']["passage_tokens"]
            span_start = doc['_source']["span_start"]
            span_end = doc['_source']["span_end"]
            question_title = doc['_source']["title"]
            score = doc['_score']
            r = rank(question_text, score)
            # print(doc['_source']["question"], r)
            if question_title != title:
                output.append((question_text, r, question_tokens, passage_tokens, span_start, span_end))

        if len(question_terms) > 0:
            # print("rerank the items")
            # sorted(output, key=lambda x: x[1])
            output.sort(key=lambda x: -x[1])

        output = output[0:topK]

        if False:
            for x in output:
                print(x[0], x[1])

        return output

    def close(self):
        self.cache.close()

# function used to create a json dump, to be indexed in lucene
def createQuestionJson():
    from allennlp.data.dataset_readers import SquadReader
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
    extractor.retrieve_best_questions("When did the Scholastic Magazine of Notre dame begin publishing?", "", 5)
    # createQuestionJson()

