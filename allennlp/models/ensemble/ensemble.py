from typing import Dict, List

import torch

from allennlp.common.params import Params
from allennlp.models import Model


class Ensemble(Model):
    """
    An ensemble runs multiple instances of a model and selects an answer from the subresults via some
    ensembling strategy.
    """

    def __init__(self,
                 submodels: List[Model]) -> None:
        vocab = submodels[0].vocab
        for submodel in submodels:
            assert submodel.vocab == vocab, "Vocabularies in ensemble differ"

        super(Ensemble, self).__init__(vocab, None)
        self.submodels = submodels

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        raise NotImplementedError

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1) -> 'Model':
        """
        Ensembles don't have vocabularies or weights, so they override _load.
        """
        model_params = config.get('model')

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        Model.remove_pretrained_embedding_params(model_params)
        model = Model.from_params(None, model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model
