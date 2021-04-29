import json
import torch


def load_words(fname, vocab, namespace):
    word_ids = []
    with open(fname) as f:
        words = json.load(f)
        for w in words:
            if (
                w.lower() in vocab._token_to_index[namespace]
                and w.title() in vocab._token_to_index[namespace]
            ):
                word_ids.append(vocab.get_token_index(w.lower(), namespace))
                word_ids.append(vocab.get_token_index(w.title(), namespace))
    return torch.LongTensor(word_ids)


def load_word_pairs(fname, vocab, namespace):
    word_ids1 = []
    word_ids2 = []
    with open(fname) as f:
        words = json.load(f)
        for w1, w2 in words:
            if (
                w1.lower() in vocab._token_to_index[namespace]
                and w2.lower() in vocab._token_to_index[namespace]
                and w1.title() in vocab._token_to_index[namespace]
                and w2.title() in vocab._token_to_index[namespace]
            ):
                word_ids1.append(vocab.get_token_index(w1.lower(), namespace))
                word_ids1.append(vocab.get_token_index(w1.title(), namespace))

                word_ids2.append(vocab.get_token_index(w2.lower(), namespace))
                word_ids2.append(vocab.get_token_index(w2.title(), namespace))
    return torch.LongTensor(word_ids1), torch.LongTensor(word_ids2)


# allennlp train training_config/pair_classification/snli_roberta.jsonnet --include-package allennlp_models -s /tmp/snli -r
def wrap_snli_embedder_with_hard_bias_mitigator(model, definitional_pairs, equalize_pairs):
    fn = (
        model._text_field_embedder.token_embedder_tokens.transformer_model.embeddings.word_embeddings.forward
    )

    def new_embedder(*args, **kwargs):
        emb = fn(*args, **kwargs)

        with torch.no_grad():
            definitional_emb1 = fn(definitional_pairs[0])
            definitional_emb2 = fn(definitional_pairs[1])

            equalize_emb1 = fn(equalize_pairs[0])
            equalize_emb2 = fn(equalize_pairs[1])

        bias_direction = PairedPCABiasDirection()(definitional_emb1, definitional_emb2)
        bias_mitigated_emb = HardBiasMitigator()(emb, bias_direction, equalize_emb1, equalize_emb2)

        return bias_mitigated_emb[: emb.size(0)]

    model._text_field_embedder.token_embedder_tokens.transformer_model.embeddings.word_embeddings.forward = (
        new_embedder
    )
