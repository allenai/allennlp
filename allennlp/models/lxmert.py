import logging
from typing import Any, Dict, List, Tuple

import difflib
from overrides import overrides
import numpy as np
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from lib.allennlp.modules.lxmert import LxmertEncoder

logger = logging.getLogger(__name__)


@Model.register("nlvr2_lxmert")
class Nlvr2Lxmert(Model):
    """
    Model for the NLVR2 task based on the LXMERT paper (Tan et al. 2019).
    Parameters
    ----------
    vocab: ``Vocabulary``
    lxmert_encoder: ``LxmertEncoder``
    wrong_output_file: ``str``, optional
        Name of the file in which to write the identifiers of
        validation examples answered incorrectly by the model (if None,
        nothing is written)
    correct_output_file: ``str``, optional
        Name of the file in which to write the identifiers of
        validation examples answered incorrectly by the model (if None,
        nothing is written)
    tokens_namespace: ``str``, optional
        Name of the key for the token IDs in the TokenIndexers
        dictionary used by the Dataset Reader. This key is used to
        access the relevant token IDs in the TextField passed to the
        model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        lxmert_encoder: LxmertEncoder,
        wrong_output_file: str = None,
        correct_output_file: str = None,
        tokens_namespace: str = "tokens",
    ) -> None:
        super().__init__(vocab)
        self._lxmert_encoder = lxmert_encoder
        self._tokens_namespace = tokens_namespace
        self.hidden_dim = self._lxmert_encoder.get_output_dim()
        self._wrong_output_file = (
            open(wrong_output_file, "w") if wrong_output_file else None
        )
        self._correct_output_file = (
            open(correct_output_file, "w") if correct_output_file else None
        )
        self.set_output_file_names(wrong_output_file, correct_output_file)
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.hidden_dim * 2, eps=1e-12),
            torch.nn.Linear(self.hidden_dim * 2, 1),
        )
        self.output_projection.apply(
            self._lxmert_encoder.encoder.model.init_bert_weights
        )
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.consistency_wrong_map = {}
        self._denotation_accuracy = CategoricalAccuracy()

    def set_output_file_names(
        self, wrong_file_name: str, correct_file_name: str
    ) -> None:
        if self._wrong_output_file:
            self._wrong_output_file.close()
        if self._correct_output_file:
            self._correct_output_file.close()
        self._wrong_output_file = open(wrong_file_name, "w")
        self._correct_output_file = open(correct_file_name, "w")

    def consistency(self, reset: bool) -> float:
        num_consistent_groups = 0
        for key in self.consistency_wrong_map:
            if self.consistency_wrong_map[key] == 0:
                num_consistent_groups += 1
        value = float(num_consistent_groups) / len(self.consistency_wrong_map)
        if reset:
            self.consistency_wrong_map = {}
        return value

    def forward(
        self,
        sentence: List[str],
        visual_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        image_id: List[List[str]],
        identifier: List[str],
        sentence_field: Dict[str, torch.Tensor],
        group_predictions: List[bool] = [False],
        denotation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, img_num, object_num, feature_size = visual_features.size()
        assert object_num == 36 and feature_size == 2048
        sentence_mask = util.get_text_field_mask(sentence_field).float()
        tokens = (
            sentence_field[self._tokens_namespace]
            .unsqueeze(1)
            .repeat(1, 2, 1)
            .view(batch_size * 2, -1)
        )
        sentence_mask = (
            sentence_mask.unsqueeze(1).repeat(1, 2, 1).view(batch_size * 2, -1)
        )
        if all(group_predictions):
            # Assume batch_size == 1 if we're in group prediction mode
            assert batch_size == 1
            # Each instance has 2 images, so the total number of images
            # should be even
            assert img_num % 2 == 0
            visual_features = visual_features.view(
                img_num // 2, 2, object_num, features_size
            )
            box_coordinates = box_coordinates.view(img_num // 2, 2, object_num, 4)
        else:
            assert img_num == 2
            visual_features = visual_features.view(
                batch_size * 2, object_num, feature_size
            )
            box_coordinates = box_coordinates.view(batch_size * 2, object_num, 4)
        _, cross_encoding = self._lxmert_encoder(
            tokens, sentence_mask, visual_features, box_coordinates,
        )
        cross_encoding = cross_encoding.contiguous().view(batch_size, -1)
        outputs = {}
        outputs["logits"] = self.output_projection(cross_encoding)
        predictions = torch.cat(
            (torch.zeros_like(outputs["logits"]), outputs["logits"]), dim=1
        )
        outputs["predictions"] = predictions
        if denotation is not None:
            outputs["loss"] = self.loss(
                outputs["logits"].view(-1), denotation.float().view(-1)
            ).sum()
            if (
                not self.training
                and self._wrong_output_file is not None
                and self._correct_output_file is not None
            ):
                for i in range(batch_size):
                    if (
                        predictions[i][denotation[i].long().item()]
                        > predictions[i][1 - denotation[i].long().item()]
                    ):
                        self._correct_output_file.write(
                            identifier[i]
                            + " "
                            + str(denotation[i].long().item())
                            + " "
                            + sentence[i]
                            + "\n"
                        )
                    else:
                        self._wrong_output_file.write(
                            identifier[i]
                            + " "
                            + str(denotation[i].long().item())
                            + " "
                            + sentence[i]
                            + "\n"
                        )
            self._denotation_accuracy(predictions, denotation)
            # Update group predictions for consistency computation
            predicted_binary = predictions.argmax(1)
            for i in range(len(identifier)):
                ident_parts = identifier[i].split("-")
                group_id = "-".join([ident_parts[0], ident_parts[1], ident_parts[-1]])
                if group_id not in self.consistency_wrong_map:
                    self.consistency_wrong_map[group_id] = 0
                if predicted_binary[i].item() != denotation[i].item():
                    self.consistency_wrong_map[group_id] += 1
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
