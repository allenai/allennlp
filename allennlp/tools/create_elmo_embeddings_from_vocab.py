import argparse
import gzip
import os

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.modules.elmo import _ElmoCharacterEncoder


def main(
    vocab_path: str,
    elmo_config_path: str,
    elmo_weights_path: str,
    output_dir: str,
    batch_size: int,
    device: int,
    use_custom_oov_token: bool = False,
):
    """
    Creates ELMo word representations from a vocabulary file. These
    word representations are _independent_ - they are the result of running
    the CNN and Highway layers of the ELMo model, but not the Bidirectional LSTM.
    ELMo requires 2 additional tokens: <S> and </S>. The first token
    in this file is assumed to be an unknown token.

    This script produces two artifacts: A new vocabulary file
    with the <S> and </S> tokens inserted and a glove formatted embedding
    file containing word : vector pairs, one per line, with all values
    separated by a space.
    """

    # Load the vocabulary words and convert to char ids
    with open(vocab_path, "r") as vocab_file:
        tokens = vocab_file.read().strip().split("\n")

    # Insert the sentence boundary tokens which elmo uses at positions 1 and 2.
    if tokens[0] != DEFAULT_OOV_TOKEN and not use_custom_oov_token:
        raise ConfigurationError("ELMo embeddings require the use of a OOV token.")

    tokens = [tokens[0]] + ["<S>", "</S>"] + tokens[1:]

    indexer = ELMoTokenCharactersIndexer()
    indices = indexer.tokens_to_indices([Token(token) for token in tokens], Vocabulary())["tokens"]
    sentences = []
    for k in range((len(indices) // 50) + 1):
        sentences.append(
            indexer.as_padded_tensor_dict(
                indices[(k * 50) : ((k + 1) * 50)], padding_lengths={"tokens": 50}
            )
        )

    last_batch_remainder = 50 - (len(indices) % 50)
    if device != -1:
        elmo_token_embedder = _ElmoCharacterEncoder(elmo_config_path, elmo_weights_path).cuda(
            device
        )
    else:
        elmo_token_embedder = _ElmoCharacterEncoder(elmo_config_path, elmo_weights_path)

    all_embeddings = []
    for i in range((len(sentences) // batch_size) + 1):
        batch = torch.stack(sentences[i * batch_size : (i + 1) * batch_size])
        if device != -1:
            batch = batch.cuda(device)

        token_embedding = elmo_token_embedder(batch)["token_embedding"].data

        # Reshape back to a list of words of shape (batch_size * 50, encoding_dim)
        # We also need to remove the <S>, </S> tokens appended by the encoder.
        per_word_embeddings = (
            token_embedding[:, 1:-1, :].contiguous().view(-1, token_embedding.size(-1))
        )

        all_embeddings.append(per_word_embeddings)

    # Remove the embeddings associated with padding in the last batch.
    all_embeddings[-1] = all_embeddings[-1][:-last_batch_remainder, :]

    embedding_weight = torch.cat(all_embeddings, 0).cpu().numpy()

    # Write out the embedding in a glove format.
    os.makedirs(output_dir, exist_ok=True)
    with gzip.open(os.path.join(output_dir, "elmo_embeddings.txt.gz"), "wb") as embeddings_file:
        for i, word in enumerate(tokens):
            string_array = " ".join(str(x) for x in list(embedding_weight[i, :]))
            embeddings_file.write(f"{word} {string_array}\n".encode("utf-8"))

    # Write out the new vocab with the <S> and </S> tokens.
    _, vocab_file_name = os.path.split(vocab_path)
    with open(os.path.join(output_dir, vocab_file_name), "w") as new_vocab_file:
        for word in tokens:
            new_vocab_file.write(f"{word}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate CNN representations for a vocabulary using ELMo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        help="A path to a vocabulary file to generate representations for.",
    )
    parser.add_argument(
        "--elmo_config", type=str, help="The path to a directory containing an ELMo config file."
    )
    parser.add_argument(
        "--elmo_weights", type=str, help="The path to a directory containing an ELMo weight file."
    )
    parser.add_argument(
        "--output_dir", type=str, help="The output directory to store the serialised embeddings."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use.")
    parser.add_argument("--device", type=int, default=-1, help="The device to run on.")
    parser.add_argument(
        "--use_custom_oov_token",
        type=bool,
        default=False,
        help="AllenNLP requires a particular OOV token."
        "To generate embeddings with a custom OOV token,"
        "add this flag.",
    )

    args = parser.parse_args()
    main(
        args.vocab_path,
        args.elmo_config,
        args.elmo_weights,
        args.output_dir,
        args.batch_size,
        args.device,
        args.use_custom_oov_token,
    )
