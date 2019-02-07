from typing import Dict, List
import torch
from torch.nn.modules import Linear

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.decoders import Decoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import BLEU, Perplexity


@Model.register("vae")
class VAE(Model):
    """
    This ``VAE`` class is a :class:`Model` which implements a simple VAE as first described
    in https://arxiv.org/pdf/1511.06349.pdf (Bowman et al., 2015).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    encoder : ``Seq2VecEncoder``, required
        The encoder model of which to pass the source tokens
    decoder : ``Model``, required
        The variational decoder model of which to pass the the latent variable
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 decoder: Decoder,
                 latent_dim: int,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(VAE, self).__init__(vocab)

        self._encoder = encoder
        self._decoder = decoder

        self._latent_dim = latent_dim

        self._encoder_output_dim = self._encoder.rnn.get_output_dim()
        self._latent_to_mean = Linear(self._encoder_output_dim, self._latent_dim)
        self._latent_to_logvar = Linear(self._encoder_output_dim, self._latent_dim)

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token) # pylint: disable=protected-access
        self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._ppl = Perplexity()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make forward pass for both training/validation/test time.
        """
        mean, logvar, latent = self.variational_encoding(source_tokens)
        output_dict = self._decoder(target_tokens, latent)
        target_mask = get_text_field_mask(target_tokens)
        output_dict["loss"] = self._get_loss(output_dict["logits"], target_tokens["tokens"],
                                             target_mask, mean, logvar)

        if not self.training:
            if "predictions" in output_dict:
                best_predictions = output_dict["predictions"]
            else:
                top_k_predictions = self._decoder.forward_beam_search(latent)
                best_predictions = top_k_predictions["predictions"][:, 0, :]
            self._bleu(best_predictions, target_tokens["tokens"])
            self._ppl(output_dict["logits"], get_text_field_mask(target_tokens)[:, 1:])

        return output_dict

    def variational_encoding(self,
                             source_tokens: Dict[str, torch.LongTensor]) -> List[torch.Tensor]:
        final_state = self._encoder(source_tokens)
        mean = self._latent_to_mean(final_state)
        logvar = self._latent_to_logvar(final_state)
        latent = self._reparameterize(mean, logvar)

        return mean, logvar, latent

    def _get_loss(self,
                  logits: torch.Tensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  mean: torch.Tensor,
                  logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get reconstruction and kld losses.
        Notice explanation in simple_seq2seq.py for creating relavant targets and mask
        """
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        reconstruction_loss_per_token = sequence_cross_entropy_with_logits(logits, relevant_targets,
                                                                           relevant_mask, average=None)
        # In VAE, we want both loss terms to be comparable, for this reason, we compare kld to each
        # predicted token NLL. For this reason we return to the following mean.
        reconstruction_loss = (reconstruction_loss_per_token * (relevant_mask.sum(1).float() + 1e-13)).mean()
        kld = self._kld_loss(mean, logvar)

        return {"reconstruction": reconstruction_loss, "kld": kld}

    @staticmethod
    def _reparameterize(mean: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
        """
        Creating the latent vector using the reparameterization trick
        """
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def _kld_loss(mean: torch.Tensor,
                  logvar: torch.Tensor) -> torch.Tensor:
        """
        Analytically calcualting kld loss, as shown in https://arxiv.org/pdf/1312.6114.pdf (Kingma et al., 2014)
        """

        return torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1 - logvar, 1))

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        if self._ppl and not self.training:
            all_metrics.update(self._ppl.get_metric(reset=reset))

        return all_metrics
 