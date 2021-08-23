import bisect
import random
from collections import abc
from typing import Sequence, Optional, Union


class ShuffledSequence(abc.Sequence):
    """
    Produces a shuffled view of a sequence, such as a list.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.
    """

    def __init__(self, inner_sequence: Sequence, indices: Optional[Sequence[int]] = None):
        self.inner = inner_sequence
        self.indices: Sequence[int]
        if indices is None:
            self.indices = list(range(len(inner_sequence)))
            random.shuffle(self.indices)
        else:
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            return self.inner[self.indices[i]]
        else:
            return ShuffledSequence(self.inner, self.indices[i])

    def __contains__(self, item) -> bool:
        for i in self.indices:
            if self.inner[i] == item:
                return True
        return False


class SlicedSequence(ShuffledSequence):
    """
    Produces a sequence that's a slice into another sequence, without copying the elements.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.
    """

    def __init__(self, inner_sequence: Sequence, s: slice):
        super().__init__(inner_sequence, range(*s.indices(len(inner_sequence))))


class ConcatenatedSequence(abc.Sequence):
    """
    Produces a sequence that's the concatenation of multiple other sequences, without
    copying the elements.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.
    """

    def __init__(self, *sequences: Sequence):
        self.sequences = sequences
        self.cumulative_sequence_lengths = [0]
        for sequence in sequences:
            self.cumulative_sequence_lengths.append(
                self.cumulative_sequence_lengths[-1] + len(sequence)
            )

    def __len__(self):
        return self.cumulative_sequence_lengths[-1]

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            if i < 0:
                i += len(self)
            if i < 0 or i >= len(self):
                raise IndexError("list index out of range")
            sequence_index = bisect.bisect_right(self.cumulative_sequence_lengths, i) - 1
            i -= self.cumulative_sequence_lengths[sequence_index]
            return self.sequences[sequence_index][i]
        else:
            return SlicedSequence(self, i)

    def __contains__(self, item) -> bool:
        return any(s.__contains__(item) for s in self.sequences)
