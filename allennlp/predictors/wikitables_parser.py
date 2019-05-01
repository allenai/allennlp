from overrides import overrides
import torch

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('wikitables-parser')
class WikiTablesParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
        question_text = json_dict["question"]
        table_rows = json_dict["table"].split('\n')

        # We are directly passing the raw table rows here. The code in ``TableQuestionContext`` will do some
        # minimal processing to extract dates and numbers from the cells.
        # pylint: enable=protected-access
        instance = self._dataset_reader.text_to_instance(question_text,  # type: ignore
                                                         table_rows)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        We need to override this because of the interactive beam search aspects.
        """
        # pylint: disable=protected-access,not-callable
        instance = self._json_to_instance(inputs)

        # Get the rules out of the instance
        index_to_rule = [production_rule_field.rule
                         for production_rule_field in instance.fields['actions'].field_list]
        rule_to_index = {rule: i for i, rule in enumerate(index_to_rule)}

        # A sequence of strings to force, then convert them to ints
        initial_tokens = inputs.get("initial_sequence", [])

        # Want to get initial_sequence on the same device as the model.
        initial_sequence = torch.tensor([rule_to_index[token] for token in initial_tokens],
                                        device=next(self._model.parameters()).device)

        # Replace beam search with one that forces the initial sequence
        original_beam_search = self._model._beam_search
        interactive_beam_search = original_beam_search.constrained_to(initial_sequence)
        self._model._beam_search = interactive_beam_search

        # Now get results
        results = self.predict_instance(instance)

        # And add in the choices. Need to convert from idxs to rules.
        results["choices"] = [
                [(probability, action)
                 for probability, action in zip(pa["action_probabilities"], pa["considered_actions"])]
                for pa in results["predicted_actions"]
        ]

        results["beam_snapshots"] = {
                # For each batch_index, we get a list of beam snapshots
                batch_index: [
                        # Each beam_snapshots consists of a list of timesteps,
                        # each of which is a list of pairs (score, sequence).
                        # The sequence is the *indices* of the rules, which we
                        # want to convert to the string representations.
                        [(score, [index_to_rule[idx] for idx in sequence])
                         for score, sequence in timestep_snapshot]
                        for timestep_snapshot in beam_snapshots
                ]
                for batch_index, beam_snapshots in interactive_beam_search.beam_snapshots.items()
        }

        # Restore original beam search
        self._model._beam_search = original_beam_search

        return results
