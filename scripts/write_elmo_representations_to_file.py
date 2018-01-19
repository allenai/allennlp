"""
Given a pre-processed input text file, this script dumps all of the internal
layers used to compute ELMo representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (http://docs.h5py.org/en/latest/) where each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.

In the default setting, each sentence is keyed in the output file by the line number
in the original text file.  Optionally, by specifying --use_sentence_key
the first token in each sentence is assumed to be a unique sentence key
used in the output file.
"""

import argparse

import torch
from torch.autograd import Variable
import h5py

from allennlp.data.dataset import Dataset
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm


indexer = ELMoTokenCharactersIndexer()

def batch_to_ids(batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Dataset(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']


def batch_to_embeddings(batch, elmo_bilm, device):
    # returns (batch_size, 3, num_times, 1024) embeddings and (batch_size, num_times) mask
    character_ids = batch_to_ids(batch)
    if device >= 0:
        character_ids = character_ids.cuda(device=device)

    bilm_output = elmo_bilm(character_ids)
    layer_activations = bilm_output['activations']
    mask_with_bos_eos = bilm_output['mask']

    without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
            for layer in layer_activations]
    # without_bos_eos is a 3 element list of (batch_size, num_times, dim) arrays
    activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
    mask = without_bos_eos[0][1]

    return activations, mask


def write_batch(batch, keys, elmo_bilm, device, fout):
    embeddings, mask = batch_to_embeddings(batch, elmo_bilm, device)
    for i, key in enumerate(keys):
        length = int(mask[i, :].sum())
        sentence_embeds = embeddings[i, :, :length, :].data.cpu().numpy()

        ds = fout.create_dataset(key,
                sentence_embeds.shape, dtype='float32',
                data=sentence_embeds
        )


def main(options_file: str,
         weight_file: str,
         input_file: str,
         output_file: str,
         batch_size: int,
         device: int,
         use_sentence_key: bool = False):

    elmo_bilm = _ElmoBiLm(options_file, weight_file)
    if device >= 0:
        elmo_bilm.cuda(device=device)

    with open(input_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
        batch = []
        keys = []
        line_no = 0

        for line in fin:
            tokens = line.strip().split()

            if use_sentence_key:
                keys.append(tokens[0])
                batch.append(tokens[1:])
            else:
                keys.append(str(line_no))
                batch.append(tokens)
            line_no += 1

            if len(batch) >= batch_size:
                # run biLM and save to file
                write_batch(batch, keys, elmo_bilm, device, fout)
                batch = []
                keys = []

        if len(batch) > 0:
            write_batch(batch, keys, elmo_bilm, device, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, help='A path to a vocabulary file to generate '
                                                       'representations for.')
    parser.add_argument('--options_file', type=str, help='The path to the ELMo options file.')
    parser.add_argument('--weight_file', type=str, help='The path to the ELMo weight file.')
    parser.add_argument('--input_file', type=str, help='The input text file')
    parser.add_argument('--output_file', type=str, help='The output hdf5 file')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size to use.')
    parser.add_argument('--device', type=int, default=-1, help='The device to run on.')
    parser.add_argument('--use_sentence_key', default=False, action='store_true')

    args = parser.parse_args()
    main(args.options_file,
         args.weight_file,
         args.input_file,
         args.output_file,
         args.batch_size,
         args.device,
         args.use_sentence_key)
