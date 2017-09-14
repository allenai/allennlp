import time
import torch
import random
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask

from contextlib import contextmanager


@contextmanager
def time_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def sort_batch(batch_size, seq_length, hidden_dim):
    tensor = torch.rand([batch_size, seq_length, hidden_dim])
    for i in range(batch_size - 1):
        tensor[i, random.randint(1, seq_length - 1):, :] = 0
    tensor = Variable(tensor)
    sequence_lengths = get_lengths_from_binary_sequence_mask((tensor[:, :, 0] != 0).long())
    with time_context("Sort batch only - 100 runs, batch: {}, seq_length: {}, "
                      "hidden_dim: {}".format(batch_size, seq_length, hidden_dim)):
        for _ in range(100):
            sort_batch_by_length(tensor, sequence_lengths)


def pack_and_unpack_sequence(batch_size, seq_length, hidden_dim):
    tensor = Variable(torch.rand([batch_size, seq_length, hidden_dim]))
    for i in range(batch_size - 1):
        tensor[i, random.randint(1, seq_length-1):, :] = 0
    sequence_lengths = get_lengths_from_binary_sequence_mask((tensor[:, :, 0] != 0).long())
    sorted_tensor, sorted_lengths, _ = sort_batch_by_length(tensor, sequence_lengths)
    with time_context("Pad and Pack only - 100 runs, batch: {}, seq_length: {}, "
                      "hidden_dim: {}".format(batch_size, seq_length, hidden_dim)):
        for _ in range(100):
            seq = pack_padded_sequence(sorted_tensor, lengths=sorted_lengths.data.tolist(), batch_first=True)
            pad_packed_sequence(seq, batch_first=True)

if __name__ == "__main__":
    sort_batch(32, 40, 100)
    pack_and_unpack_sequence(32, 40, 100)

    sort_batch(120, 40, 100)
    pack_and_unpack_sequence(120, 40, 100)

    sort_batch(32, 100, 100)
    pack_and_unpack_sequence(32, 100, 100)

    sort_batch(32, 600, 100)
    pack_and_unpack_sequence(32, 600, 100)

    # Large embeddings
    sort_batch(32, 40, 400)
    pack_and_unpack_sequence(32, 40, 400)

    sort_batch(120, 40, 400)
    pack_and_unpack_sequence(120, 40, 400)

    sort_batch(32, 100, 400)
    pack_and_unpack_sequence(32, 100, 400)

    sort_batch(32, 600, 400)
    pack_and_unpack_sequence(32, 600, 400)
