# pylint: disable=protected-access
from typing import List
import numpy
import torch
from allennlp.interpret.attackers import Attacker
from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.tokenizers import Token

@Attacker.register('hotflip')
class Hotflip(Attacker):
    """
    Runs the HotFlip style attack at the word-level https://arxiv.org/abs/1712.06751.
    We use the first-order taylor approximation described in https://arxiv.org/abs/1903.06620,
    in the function first_order_taylor(). Constructing this object is expensive due to the
    construction of the embedding matrix.
    """
    def __init__(self, predictor: Predictor):
        super().__init__(predictor)
        self.vocab = self.predictor._model.vocab
        self.token_embedding = self._construct_embedding_matrix()

    def _construct_embedding_matrix(self):
        """ For HotFlip, we need a word embedding matrix to search over. The below is neccessary for
        models such as ELMo, character-level models, or for models that use a projection layer
        after their word embeddings.
        We run all of the tokens from the vocabulary through the TextFieldEmbedder, and save the final
        output embedding. We then group all of those output embeddings into an "embedding matrix".
        """
        # Gets all tokens in the vocab and their corresponding IDs
        all_tokens = list(self.vocab._token_to_index["tokens"].keys())
        all_inputs = \
        {"tokens":torch.LongTensor([x for x in self.vocab._index_to_token["tokens"].keys()]).unsqueeze(0)}

        # handle when a model uses character-level inputs, e.g., ELMo or a CharCNN
        if "token_characters" in self.predictor._dataset_reader._token_indexers:
            indexer = self.predictor._dataset_reader._token_indexers["token_characters"]
            tokens = [Token(x) for x in all_tokens]
            max_token_length = max(len(x) for x in all_tokens)
            indexed_tokens = indexer.tokens_to_indices(tokens, self.vocab, "token_characters")
            padded_tokens = indexer.pad_token_sequence(indexed_tokens,
                                                       desired_num_tokens={"token_characters": len(tokens)},
                                                       padding_lengths={"num_token_characters": max_token_length})
            all_inputs['token_characters'] = torch.LongTensor(padded_tokens['token_characters']).unsqueeze(0)

            if "elmo" in self.predictor._dataset_reader._token_indexers:
                elmo_tokens = []
                indexer = self.predictor._dataset_reader._token_indexers["elmo"]
                for tok in all_tokens:
                    tmp = indexer.tokens_to_indices([Token(text=tok)], self.vocab, "sentence")["sentence"]
                    elmo_tokens.append(tmp[0])
                all_inputs["elmo"] = torch.LongTensor(elmo_tokens).unsqueeze(0)

        # find the TextFieldEmbedder
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                embedder = module
        # pass all tokens through the fake matrix and create an embedding out of it.
        embedding_matrix = embedder(all_inputs).squeeze()
        return Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                         embedding_dim=embedding_matrix.shape[1],
                         weight=embedding_matrix,
                         trainable=False)

    def attack_from_json(self,
                         inputs: JsonDict = None,
                         input_field_to_attack: str = 'tokens',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = ["@@NULL@@"]): # pylint disable=unused-argument,dangerous-default-value

        """
        Replaces one token at a time from the input until the model's prediction changes.
        `input_field_to_attack` is for example `tokens`, it says what the input
        field is called. `grad_input_field` is for example `grad_input_1`, which
        is a key into a grads dictionary.

        The method computes the gradient w.r.t. the tokens, finds
        the token with the maximum gradient (by L2 norm), and replaces it will
        another token based on the first-order Taylor approximation of the loss.
        This process is iteratively repeated until the prediction changes.
        Once a token is replaced, it is not flipped again.

        TODO (@Eric-Wallace) add functionality for ignore_tokens in the future.
        """
        original_instances = self.predictor.inputs_to_labeled_instances(inputs)
        original_tokens = list(original_instances[0][input_field_to_attack].tokens)
        final_tokens = []
        new_prediction = None
        for new_instances in original_instances:
            new_instances = [new_instances]
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)

            # get a list of fields that we want to check to see if they change
            # (we want to change model predictions)
            fields_to_compare = {}
            for key in new_instances[0].fields:
                if key not in inputs.keys() and key != input_field_to_attack:
                    fields_to_compare[key] = test_instances[0][key]

            current_tokens = new_instances[0][input_field_to_attack].tokens
            grads, outputs = self.predictor.get_gradients(new_instances)
            flipped = []
            while True:
                # Compute L2 norm of all grads.
                grad = grads[grad_input_field]
                grads_mag = [numpy.sqrt(g.dot(g)) for g in grad]

                # only flip a token once
                for index in flipped:
                    grads_mag[index] = -1

                # we flip the token with highest gradient norm
                index_of_token_to_flip = numpy.argmax(grads_mag)
                if grads_mag[index_of_token_to_flip] == -1:
                    break
                flipped.append(index_of_token_to_flip)

                # Get new token using taylor approximation
                input_tokens = new_instances[0][input_field_to_attack]._indexed_tokens["tokens"]
                original_id_of_token_to_flip = input_tokens[index_of_token_to_flip]
                new_id_of_flipped_token = \
                    first_order_taylor(grad[index_of_token_to_flip],
                                       self.token_embedding.weight,
                                       original_id_of_token_to_flip)
                # flip token
                new_instances[0][input_field_to_attack].tokens[index_of_token_to_flip] = \
                    Token(self.vocab._index_to_token["tokens"][new_id_of_flipped_token])
                new_instances[0].indexed = False

                # Get model predictions on new_instances, and then label the instances
                grads, outputs = self.predictor.get_gradients(new_instances) # predictions
                for key in outputs:
                    if isinstance(outputs[key], torch.Tensor):
                        outputs[key] = outputs[key].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[key], list):
                        outputs[key] = outputs[key][0]
                # add labels to new_instances
                self.predictor.predictions_to_labeled_instances(new_instances[0], outputs)

                # if the prediction has changed, then stop
                label_change = False
                for field in fields_to_compare:
                    if field in new_instances[0].fields:

                        equal = new_instances[0][field].__eq__(fields_to_compare[field])
                    else:
                        equal = outputs[field] == fields_to_compare[field]
                    if not equal:
                        label_change = True
                        break
                # if the prediction has changed, we want to return the new answer
                # for visualization in the demo.
                if label_change:
                    new_prediction = get_new_prediction(outputs)
                    break

            final_tokens.append(current_tokens)
        return \
            sanitize({"final": final_tokens, "original": original_tokens, "new_prediction": new_prediction})

# Get the model's new prediction given its outputs
def get_new_prediction(outputs):
    new_prediction = None
    if "probs" in outputs: # sentiment analysis
        new_prediction = outputs["probs"]
    elif "label_probs" in outputs: # textual entailment
        new_prediction = outputs["label_probs"]
    elif "best_span_str" in outputs: # bidaf
        new_prediction = outputs["best_span_str"]
    elif "answer" in outputs: # NAQANet
        ans_type = outputs["answer"]["answer_type"]
        if ans_type == "count":
            new_prediction = outputs["answer"]["count"]
        else:
            new_prediction = outputs["answer"]["value"]
    return new_prediction

def first_order_taylor(grad: numpy.ndarray,
                       embedding_matrix: torch.nn.parameter.Parameter,
                       token_idx: int):
    """
    the below code is based on
    https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
    research/adversarial/adversaries/brute_force_adversary.py

    Replaces the current token_idx with another token_idx to increase the loss. In particular,
    this function uses the grad, alongside the embedding_matrix to select
    the token that maximizes the first-order taylor approximation of the loss.
    """
    grad = torch.from_numpy(grad)
    embedding_matrix = embedding_matrix.cpu()
    word_embeds = \
        torch.nn.functional.embedding(torch.LongTensor([token_idx]), embedding_matrix)
    word_embeds = word_embeds.detach().unsqueeze(0)
    grad = grad.unsqueeze(0).unsqueeze(0)
    # solves equation (3) here https://arxiv.org/abs/1903.06620
    new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix))
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embeds)).unsqueeze(-1)
    neg_dir_dot_grad = -1 * (prev_embed_dot_grad - new_embed_dot_grad)
    _, best_at_each_step = neg_dir_dot_grad.max(2)
    return best_at_each_step[0].data[0].detach().cpu().item() # return the best candidate
