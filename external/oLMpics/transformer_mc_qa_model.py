from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
from pytorch_transformers.modeling_xlnet import XLNetConfig, XLNetModel
from pytorch_transformers.modeling_bert import BertConfig, BertModel
from pytorch_transformers.modeling_utils import SequenceSummary
from pytorch_transformers.tokenization_gpt2 import bytes_to_unicode
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1


@Model.register("transformer_mc_qa")
class TransformerMCQAModel(Model):
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
                 probe_type: str = None,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._pretrained_model = pretrained_model
        if 'roberta' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'xlnet' in pretrained_model:
            self._padding_value = 5  # The index of the XLNet padding token
            self._transformer_model = XLNetModel.from_pretrained(pretrained_model)
            self.sequence_summary = SequenceSummary(self._transformer_model.config)
        elif 'bert' in pretrained_model:
            self._transformer_model = BertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        else:
            assert (ValueError)

        if probe_type == 'MLP':
            layer_freeze_regexes = ["embeddings", "encoder"]

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


        transformer_config = self._transformer_model.config
        transformer_config.num_labels = 1
        self._output_dim = self._transformer_model.config.hidden_size

        #if 'roberta' in pretrained_model:
        #    self._classifier = RobertaClassificationHead(transformer_config)
        #else:

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, 1)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()
        #self._classifier.apply(self._transformer_model.init_weights)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")

        # Segment ids are not used by RoBERTa
        if 'roberta' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                      # token_type_ids=util.combine_initial_dims(segment_ids),
                                                      attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
            #cls_output = pooled_output
        elif 'xlnet' in self._pretrained_model:
            transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                          attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self.sequence_summary(transformer_outputs[0])

        elif 'bert' in self._pretrained_model:
            last_layer, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                          token_type_ids=util.combine_initial_dims(segment_ids),
                                                          attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        else:
            assert (ValueError)





        if self._debug > 0:
            print(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)
        label_logits = label_logits.view(-1, num_choices)

        output_dict = {}
        output_dict['label_logits'] = label_logits

        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            loss = self._loss(label_logits, label)
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


