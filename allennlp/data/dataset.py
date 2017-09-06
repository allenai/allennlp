"""
A :class:`~Dataset` represents a collection of data suitable for feeding into a model.
For example, when you train a model, you will likely have a *training* dataset and a *validation* dataset.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Union

import numpy
import tqdm

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dataset:
    """
    A collection of :class:`~allennlp.data.instance.Instance` objects.
    The ``Instances`` have ``Fields``, and the fields
    could be in an indexed or unindexed state - the ``Dataset`` has methods around indexing the
    data and converting the data into arrays.
    """
    def __init__(self, instances: List[Instance]) -> None:
        """
        A Dataset just takes a list of instances in its constructor.  It's important that all
        subclasses have an identical constructor to this (though possibly with different Instance
        types).  If you change the constructor, you also have to override all methods in this base
        class that call the constructor, such as `truncate()`.
        """
        all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__
                                                                for k, v in x.fields.items()}
                                                               for x in instances]
        # Check all the field names and Field types are the same for every instance.
        if not all([all_instance_fields_and_types[0] == x for x in all_instance_fields_and_types]):
            raise ConfigurationError("You cannot construct a Dataset with non-homogeneous Instances.")

        self.instances = instances

    def truncate(self, max_instances: int):
        """
        If there are more instances than ``max_instances`` in this dataset, we truncate the
        instances to the first ``max_instances``.  This `modifies` the current object, and returns
        nothing.
        """
        if len(self.instances) > max_instances:
            self.instances = self.instances[:max_instances]

    def index_instances(self, vocab: Vocabulary):
        """
        Converts all ``UnindexedFields`` in all ``Instances`` in this ``Dataset`` into
        ``IndexedFields``.  This modifies the current object, it does not return a new object.
        """
        logger.info("Indexing dataset")
        for instance in tqdm.tqdm(self.instances):
            instance.index_fields(vocab)

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this dataset.  Each ``Instance``
        has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this dataset into arrays of consistent length, or to set
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

    def as_array_dict(self,
                      padding_lengths: Dict[str, Dict[str, int]] = None,
                      verbose: bool = True) ->Dict[str, Union[numpy.ndarray, Dict[str, numpy.ndarray]]]:
        # This complex return type is actually predefined elsewhere as a DataArray,
        # but we can't use it because mypy doesn't like it.
        """
        This method converts this ``Dataset`` into a set of numpy arrays that can be passed through
        a model.  In order for the numpy arrays to be valid arrays, all ``Instances`` in this
        dataset need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the arrays for each field in each instance into a set of
        batched arrays for each field.

        Parameters
        ----------
        padding_lengths : ``Dict[str, Dict[str, int]]``
            If a key is present in this dictionary with a non-``None`` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").
        verbose : ``bool``, optional (default=``True``)
            Should we output logging information when we're doing this padding?  If the dataset is
            large, this is nice to have, because padding a large dataset could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious.

        Returns
        -------
        data_arrays : ``Dict[str, DataArray]``
            A dictionary of data arrays, keyed by field name, suitable for passing as input to a
            model.  This is a `batch` of instances, so, e.g., if the instances have a "question"
            field and an "answer" field, the "question" fields for all of the instances will be
            grouped together into a single array, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of arrays, for batched computation. Additionally,
            for TextFields, the value of the dictionary key is no longer a single array, but another
            dictionary mapping TokenIndexer keys to arrays. The number of elements in this
            sub-dictionary therefore corresponds to the number of ``TokenIndexers`` used to index
            the Field.
        """
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular field and padding key.  If we were, we use that
        # instead of the instance-based one.
        if verbose:
            logger.info("Padding dataset of size %d to lengths %s", len(self.instances), str(padding_lengths))
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

        # Now we actually pad the instances to numpy arrays.
        field_arrays: Dict[str, list] = defaultdict(list)
        if verbose:
            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))
        for instance in self.instances:
            for field, arrays in instance.as_array_dict(lengths_to_use).items():
                field_arrays[field].append(arrays)

        # Finally, we combine the arrays that we got for each instance into one big array (or set
        # of arrays) per field.  The `Field` classes themselves have the logic for batching the
        # arrays together, so we grab a dictionary of field_name -> field class from the first
        # instance in the dataset.
        field_classes = self.instances[0].fields
        final_fields = {}
        for field_name, field_array_list in field_arrays.items():
            final_fields[field_name] = field_classes[field_name].batch_arrays(field_array_list)
        return final_fields
