from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaForMaskedLM
from pytorch_transformers.modeling_xlnet import XLNetConfig, XLNetLMHeadModel
from pytorch_transformers.modeling_bert import BertConfig, BertForMaskedLM
from pytorch_transformers.modeling_utils import SequenceSummary
from pytorch_transformers.tokenization_gpt2 import bytes_to_unicode
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import CrossEntropyLoss

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1


class BertForMultiChoiceMaskedLM(BertForMaskedLM):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None, all_masked_index_ids = None, label = None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            # choosing prediction with all_masked_index_ids
            masked_lm_loss = 0
            for i,e in enumerate(all_masked_index_ids):
                masked_lm_loss += \
                    loss_fct(prediction_scores[i,e[0][0][0],[e[0][0][1],e[1][0][1],e[2][0][1]]].unsqueeze(0),label[i].unsqueeze(0))

            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class RobertaForMultiChoiceMaskedLM(RobertaForMaskedLM):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, position_ids=None,
                head_mask=None, all_masked_index_ids = None, label = None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # choosing prediction with all_masked_index_ids
            masked_lm_loss = 0
            for i, choices in enumerate(all_masked_index_ids):
                masked_lm_loss += \
                    loss_fct(prediction_scores[i, choices[0][0][0], [c[0][1] for c in choices]].unsqueeze(0), label[i].unsqueeze(0))

            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

@Model.register("transformer_masked_lm")
class TransformerMaskedLMModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 unfreeze_pooler: bool = False,
                 top_layer_only: bool = True,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 on_load: bool = False,
                 loss_on_all_vocab: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._loss_on_all_vocab = loss_on_all_vocab

        self._pretrained_model = pretrained_model
        if 'roberta' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            if loss_on_all_vocab:
                self._transformer_model = RobertaForMaskedLM.from_pretrained(pretrained_model)
            else:
                self._transformer_model = RobertaForMultiChoiceMaskedLM.from_pretrained(pretrained_model)
        elif 'xlnet' in pretrained_model:
            self._padding_value = 5  # The index of the XLNet padding token
            self._transformer_model = XLNetLMHeadModel.from_pretrained(pretrained_model)
        elif 'bert' in pretrained_model:
            if loss_on_all_vocab:
                self._transformer_model = BertForMaskedLM.from_pretrained(pretrained_model)
            else:
                self._transformer_model = BertForMultiChoiceMaskedLM.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
        else:
            assert (ValueError)

        ## TODO ask oyvind about this code ...
        for name, param in self._transformer_model.named_parameters():
            if layer_freeze_regexes and requires_grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            else:
                grad = requires_grad
            if grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if unfreeze_pooler:
            try:
                self._transformer_model.pooler.dense.weight.requires_grad = True
                self._transformer_model.pooler.dense.bias.requires_grad = True
            except:
                pass

        transformer_config = self._transformer_model.config
        transformer_config.num_labels = 1
        self._output_dim = self._transformer_model.config.hidden_size

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2

    def forward(self,
                phrase: Dict[str, torch.LongTensor],
                masked_labels: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = phrase['tokens']
        masked_labels[(masked_labels == 0)] = -1

        batch_size = input_ids.size(0)
        # TODO make this a param
        num_choices = len(metadata[0]['choice_text_list'])

        question_mask = (input_ids != self._padding_value).long()

        # Segment ids are not used by RoBERTa
        if 'roberta' in self._pretrained_model or 'bert' in self._pretrained_model:
            if self._loss_on_all_vocab:
                outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids), masked_lm_labels=masked_labels)
            else:
                outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids), masked_lm_labels=masked_labels,
                                                  all_masked_index_ids=[e['all_masked_index_ids'] for e in metadata],label=label)
            loss, predictions = outputs[:2]
        elif 'xlnet' in self._pretrained_model:
            transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                          attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self.sequence_summary(transformer_outputs[0])
        else:
            assert (ValueError)

        # TODO move this terrible loopy code i wrote to martix operations :)  ...
        output_dict = {}
        label_logits = torch.zeros(batch_size,num_choices)
        for e,example in enumerate(metadata):
            for c,choice in enumerate(example['all_masked_index_ids']):
                for t in choice:
                    label_logits[e,c] +=  predictions[e,t[0],t[1]]

        self._accuracy(label_logits, label)
        output_dict["loss"] = loss

        if self._debug > 0:
            print(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'EM': self._accuracy.get_metric(reset),
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


