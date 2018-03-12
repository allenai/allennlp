from typing import Dict, List

from allennlp.models import Model
import torch


class Ensemble(Model):

    def __init__(self,
                 submodels: List[Model]) -> None:
        assert len(submodels) > 0
        vocab = submodels[0].vocab
        for submodel in submodels:
            assert submodel.vocab == vocab, "Vocabularies in ensemble differ"

        super(Ensemble, self).__init__(vocab, None)
        self.submodels = submodels

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        raise NotImplementedError
