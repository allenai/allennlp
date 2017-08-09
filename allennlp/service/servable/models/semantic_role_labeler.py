from typing import Dict

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField, IndexField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import SemanticRoleLabeler
from allennlp.service.servable import Servable, JSONDict

import spacy

class SemanticRoleLabelerServable(Servable):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 model: SemanticRoleLabeler) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.model = model
        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        sentence = inputs["sentence"]
        tokens = self.tokenizer.tokenize(sentence)
        text = TextField(tokens, token_indexers=self.token_indexers)

        results = {"verbs": []}  # type: JSONDict
        spacy_doc = self.nlp(sentence)
        for i, word in enumerate(spacy_doc):
            if word.pos_ == "VERB":
                verb_indicator = IndexField(i, text)
                output = self.model.tag(text, verb_indicator)
                results["verbs"].append({
                        "index": i,
                        "verb": word.text,
                        "tags": output["tags"],
                        "class_probabilities": output["class_probabilities"].tolist()
                })

        return results

    @classmethod
    def from_params(cls, params: Params) -> 'SemanticRoleLabelerServable':
        tokenizer = Tokenizer.from_params(params.pop("tokenizer"))

        token_indexers = {}
        token_indexer_params = params.pop('token_indexers')
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)

        vocab_dir = params.pop('vocab_dir')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = params.pop("model")
        assert model_params.pop("type") == "semantic_role_labeler"
        model = SemanticRoleLabeler.from_params(vocab, model_params)

        return SemanticRoleLabelerServable(tokenizer=tokenizer,
                                           token_indexers=token_indexers,
                                           model=model)
