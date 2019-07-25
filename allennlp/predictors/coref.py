from typing import List, Dict

from overrides import overrides
from copy import deepcopy
import numpy
from spacy.tokens import Doc

from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import SequenceLabelField, SpanField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("coreference-resolution")
class CorefPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenise our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)

    def predict(self, document: str) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        .. code-block:: js

            {
            "document": [tokenised document text]
            "clusters":
              [
                [
                  [start_index, end_index],
                  [start_index, end_index]
                ],
                [
                  [start_index, end_index],
                  [start_index, end_index],
                  [start_index, end_index],
                ],
                ....
              ]
            }

        Parameters
        ----------
        document : ``str``
            A string representation of a document.

        Returns
        -------
        A dictionary representation of the predicted coreference clusters.
        """
        return self.predict_json({"document" : document})

    def predict_tokenized(self, tokenized_document: List[str]) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        Parameters
        ----------
        tokenized_document : ``List[str]``
            A list of words representation of a tokenized document.

        Returns
        -------
        A dictionary representation of the predicted coreference clusters.
        """
        instance = self._words_list_to_instance(tokenized_document)
        return self.predict_instance(instance)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        predicted_clusters = []
        """cluster_assignment = [-1]*len(outputs['top_spans'])
        for i in range(len(outputs['top_spans'])):
            antecedent = outputs['predicted_antecedents'][i]
            if antecedent > -1 and cluster_assignment[antecedent] > -1:
                cluster_assignment[i] = cluster_assignment[antecedent]
                predicted_clusters[cluster_assignment[i]].append(tuple(outputs['top_spans'][i]))
            elif antecedent > -1:
                cluster_assignment[antecedent] = len(predicted_clusters)
                cluster_assignment[i] = len(predicted_clusters)
                predicted_clusters.append([tuple(outputs['top_spans'][antecedent]), tuple(outputs['top_spans'][i])])
        print(outputs['top_spans'])
        print(outputs['predicted_antecedents'])"""
        predicted_clusters = outputs['clusters']
        instances = []
        print_flag = True
        for clust in predicted_clusters:
            new_instance = deepcopy(instance)
            span_labels = [0 if (span.span_start, span.span_end) in clust else -1 for span in instance['spans']]
            new_instance.add_field('span_labels',
                                   SequenceLabelField(span_labels, instance['spans']), self._model.vocab)
            new_instance['metadata'].metadata['clusters'] = [clust]
            instances.append(new_instance)
        if len(instances) == 0:
            new_instance = deepcopy(instance)
            span_labels = [-1]*len(instance['spans'])
            new_instance.add_field('span_labels',
                                   SequenceLabelField(span_labels, instance['spans']), self._model.vocab)
            new_instance['metadata'].metadata['clusters'] = []
            instances.append(new_instance)
        return instances


    @staticmethod
    def replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
        """
        Uses a list of coreference clusters to convert a spacy document into a
        string, where each coreference is replaced by its main mention.
        """
        # Original tokens with correct whitespace
        resolved = list(tok.text_with_ws for tok in document)

        for cluster in clusters:
            # The main mention is the first item in the cluster
            mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
            mention_span = document[mention_start: mention_end]

            # The coreferences are all items following the first in the cluster
            for coref in cluster[1:]:
                final_token = document[coref[1]]
                # In both of the following cases, the first token in the coreference
                # is replaced with the main mention, while all subsequent tokens
                # are masked out with "", so that they can be elimated from
                # the returned document during "".join(resolved).

                # The first case attempts to correctly handle possessive coreferences
                # by inserting "'s" between the mention and the final whitespace
                # These include my, his, her, their, our, etc.

                # Disclaimer: Grammar errors can occur when the main mention is plural,
                # e.g. "zebras" becomes "zebras's" because this case isn't
                # being explictly checked and handled.

                if final_token.tag_ in ["PRP$", "POS"]:
                    resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                else:
                # If not possessive, then replace first token with main mention directly
                    resolved[coref[0]] = mention_span.text + final_token.whitespace_
                # Mask out remaining tokens
                for i in range(coref[0] + 1, coref[1] + 1):
                    resolved[i] = ""

        return "".join(resolved)

    def coref_resolved(self, document: str) -> str:
        """
        Produce a document where each coreference is replaced by the its main mention

        Parameters
        ----------
        document : ``str``
            A string representation of a document.

        Returns
        -------
        A string with each coference replaced by its main mention
        """

        spacy_document = self._spacy(document)
        clusters = self.predict(document).get("clusters")

        # Passing a document with no coreferences returns its original form
        if not clusters:
            return document

        return self.replace_corefs(spacy_document, clusters)

    def _words_list_to_instance(self, words: List[str]) -> Instance:
        """
        Create an instance from words list represent an already tokenized document,
        for skipping tokenization when that information already exist for the user
        """
        spacy_document = Doc(self._spacy.vocab, words=words)
        for pipe in filter(None, self._spacy.pipeline):
            pipe[1](spacy_document)

        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]  # pylint: disable=not-an-iterable
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"document": "string of document text"}``
        """
        document = json_dict["document"]
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance
