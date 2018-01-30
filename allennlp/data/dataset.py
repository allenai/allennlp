"""
A :class:`~Batch` represents a collection of ``Instance`` s to be fed
through a model.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Union, Iterator, Iterable

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Batch(Iterable):
    """
    A batch of Instances. In addition to containing the instances themselves,
    it contains helper functions for converting the data into tensors.
    """
    def __init__(self, instances: Iterable[Instance]) -> None:
        """
        A Batch just takes an iterable of instances in its constructor and hangs onto them
        in a list.
        """
        super().__init__()

        self.instances: List[Instance] = ensure_list(instances)
        self._check_types()

    def _check_types(self) -> None:
        """
        Check that all the instances have the same types.
        """
        all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__
                                                                for k, v in x.fields.items()}
                                                               for x in self.instances]
        # Check all the field names and Field types are the same for every instance.
        if not all([all_instance_fields_and_types[0] == x for x in all_instance_fields_and_types]):
            raise ConfigurationError("You cannot construct a Batch with non-homogeneous Instances.")

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this batch.  Each ``Instance``
        has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this batch into arrays of consistent length, or to set
        model parameters, etc.
        """
        padding_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        all_instance_lengths: List[Dict[str, Dict[str, int]]] = [instance.get_padding_lengths()
                                                                 for instance in self.instances]
        if not all_instance_lengths:
            return {**padding_lengths}
        all_field_lengths: Dict[str, List[Dict[str, int]]] = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)
        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x[padding_key] if padding_key in x else 0 for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value
        return {**padding_lengths}

    def as_tensor_dict(self,
                       padding_lengths: Dict[str, Dict[str, int]] = None,
                       cuda_device: int = -1,
                       for_training: bool = True,
                       verbose: bool = False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # This complex return type is actually predefined elsewhere as a DataArray,
        # but we can't use it because mypy doesn't like it.
        """
        This method converts this ``Batch`` into a set of pytorch Tensors that can be passed
        through a model.  In order for the tensors to be valid tensors, all ``Instances`` in this
        batch need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the tensors for each field in each instance into a set of
        batched tensors for each field.

        Parameters
        ----------
        padding_lengths : ``Dict[str, Dict[str, int]]``
            If a key is present in this dictionary with a non-``None`` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").
        cuda_device : ``int``
            If cuda_device >= 0, GPUs are available and Pytorch was compiled with CUDA support, the
            tensor will be copied to the cuda_device specified.
        for_training : ``bool``, optional (default=``True``)
            If ``False``, we will pass the ``volatile=True`` flag when constructing variables,
            which disables gradient computations in the graph.  This makes inference more efficient
            (particularly in memory usage), but is incompatible with training models.
        verbose : ``bool``, optional (default=``False``)
            Should we output logging information when we're doing this padding?  If the batch is
            large, this is nice to have, because padding a large batch could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious (and really slow).

        Returns
        -------
        tensors : ``Dict[str, DataArray]``
            A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
            This is a `batch` of instances, so, e.g., if the instances have a "question" field and
            an "answer" field, the "question" fields for all of the instances will be grouped
            together into a single tensor, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of tensors, for batched computation. Additionally,
            for complex ``Fields``, the value of the dictionary key is not necessarily a single
            tensor.  For example, with the ``TextField``, the output is a dictionary mapping
            ``TokenIndexer`` keys to tensors. The number of elements in this sub-dictionary
            therefore corresponds to the number of ``TokenIndexers`` used to index the
            ``TextField``.  Each ``Field`` class is responsible for batching its own output.
        """
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular field and padding key.  If we were, we use that
        # instead of the instance-based one.
        if verbose:
            logger.info("Padding batch of size %d to lengths %s", len(self.instances), str(padding_lengths))
            logger.info("Getting max lengths from instances")
        instance_padding_lengths = self.get_padding_lengths()
        if verbose:
            logger.info("Instance max lengths: %s", str(instance_padding_lengths))
        lengths_to_use: Dict[str, Dict[str, int]] = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_lengths[field_name].get(padding_key) is not None:
                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]

        # Now we actually pad the instances to tensors.
        field_tensors: Dict[str, list] = defaultdict(list)
        if verbose:
            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))
        for instance in self.instances:
            for field, tensors in instance.as_tensor_dict(lengths_to_use, cuda_device, for_training).items():
                field_tensors[field].append(tensors)

        # Finally, we combine the tensors that we got for each instance into one big tensor (or set
        # of tensors) per field.  The `Field` classes themselves have the logic for batching the
        # tensors together, so we grab a dictionary of field_name -> field class from the first
        # instance in the batch.
        field_classes = self.instances[0].fields
        final_fields = {}
        for field_name, field_tensor_list in field_tensors.items():
            final_fields[field_name] = field_classes[field_name].batch_tensors(field_tensor_list)
        return final_fields

    def __iter__(self) -> Iterator[Instance]:
        return iter(self.instances)

    def index_instances(self, vocab: Vocabulary) -> None:
        for instance in self.instances:
            instance.index_fields(vocab)
