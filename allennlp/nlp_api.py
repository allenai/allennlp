from typing import Callable, Dict, List, Union

import torch

from .common.checks import ConfigurationError
from .common import Params


class NlpApi:
    """
    The ``NlpApi`` provides a high-level API for building NLP models.  You don't have to use it
    for building a model with AllenNLP, but we think you should, because it's useful.

    ``NlpApi`` abstracts away particular decisions that are frequently necessary in NLP, like how
    exactly words get represented as vectors, or what kind of RNN should be used when processing
    sequences of word vectors.  When building a model, you just call a function like
    :func:`embed_input`, or :func:`get_recurrent_layer`, and leave the decision of `how` exactly
    things are embedded or encoded for later.  This makes it easy to define a `class` of models in
    your model code, and easily run controlled experiments with these models later.  If you come up
    with a new RNN variant (say, multiplicative LSTMs, or recurrent additive networks), or a new
    way to represent words, you can easily run experiments on all models that use this API without
    changing any model code - you just change the state in the ``NlpApi``, which can be done from
    parameters in an experiment configuration file.

    Note that in the typical, intended usage of this class, you won't be actually instantiating
    this class yourself - AllenNLP will instantiate it :func:`from_params` given a parameter file
    that you provide to our experiment framework.  Though you can instantiate this object yourself
    and bypass our experiment code (or just ignore this class altogether and build models without
    it), if you really want to.  TODO(matt): link to some documentation here.

    Parameters
    ----------
    embedders : ``Dict[str, Union[torch.nn.Module, List[torch.nn.Module]]]``, optional (default=``{}``)
        A mapping from names to token embedders.  The names should typically correspond to fields
        in your input, like "passage_tokens", or "question_tokens".  Because our
        :class:`~allennlp.data.fields.TextField` returns a `list` of arrays, one for each
        :class:`~allennlp.data.token_indexers.TokenIndexer`, you should typically specify a `list`
        of ``Modules`` here, one to process each input array and produce a vector for each token.
        E.g., if you convert passage words into two different word ids, plus a sequence of
        character ids, you should have three modules for "passage_tokens" to do three separate
        embeddings given the different arrays (one of which could actually be a CNN encoder).  The
        three separate arrays then get concatenated and returned.

        Alternatively, instead of giving a list of ``Modules``, you could give a single ``Module``
        that knows how to process the whole list of inputs, returning a single ``Tensor``.

        This abstraction is key to allowing you to easily switch between word embeddings and
        character-level encoders, or token-in-context embeddings produced by some pre-trained
        model, or any combination of these.  Instead of instantiating your own embedding layers, or
        character-level CNNs, you should call :func:`embed_text_field` to do that for you, so it is
        easy to try new representations later.  See the documentation of that function for a little
        more detail.
    recurrent_layers : ``Dict[str, torch.nn.Module]``, optional (default=``{}``)
        A mapping from names to recurrent layers, like LSTMs and GRUs.  The name here corresponds
        to some part of your `model`, like "passage_encoder", not the `type` of the recurrent
        layer.
    default_recurrent_layer : ``Callable[[], torch.nn.Module]``, optional (default=``None``)
        A function that returns a new recurrent layer.  Giving this makes it so you can use unique
        names for all recurrent layers in your model, but still only have to specify one layer
        type.  See :func:`get_recurrent_layer` for more detail.
    recurrent_cells : Dict[str, torch.nn.Module], optional (default=``{}``)
        A mapping from names to recurrent cells, for building, e.g., a TreeLSTM.  Similarly to
        ``recurrent_layers``, the names correspond to places in your model where these cells get
        used, not to anything about what kind of cell you're using.
    default_recurrent_cell : ``Callable[[], torch.nn.Module]``, optional (default=``None``)
        A function that returns a new recurrent cell.  Giving this makes it so you can use unique
        names for all recurrent cells in your model, but still only have to specify one cell type.
        See :func:`get_recurrent_cell` for more detail.
    """
    def __init__(self,
                 embedders: Dict[str, Union[torch.nn.Module, List[torch.nn.Module]]] = None,
                 recurrent_layers: Dict[str, torch.nn.Module] = None,
                 default_recurrent_layer: Callable[[], torch.nn.Module] = None,
                 recurrent_cells: Dict[str, torch.nn.Module] = None,
                 default_recurrent_cell: Callable[[], torch.nn.Module] = None):
        if embedders is None:
            embedders = {}
        self._embedders = embedders
        if recurrent_layers is None:
            recurrent_layers = {}
        self._recurrent_layers = recurrent_layers
        self._default_recurrent_layer = default_recurrent_layer
        if recurrent_cells is None:
            recurrent_cells = {}
        self._recurrent_cells = recurrent_cells
        self._default_recurrent_cell = default_recurrent_cell

    def embed_text_field(self,
                         input_tensors: List[torch.Tensor],
                         name: str = "default",
                         fallback_behavior: str = "crash") -> torch.Tensor:
        """
        Given an input corresponding to a :class:`~allennlp.data.fields.TextField`, return an
        embedding of the tokens in that field.  The ``name`` here is typically the name of an input
        field, like "passage", or "question".  And we will use the ``embedders`` specified for the
        given ``name`` in the constructor to actually perform the embedding.

        ``TextFields`` are represented in AllenNLP as `lists` of tensors, where each item in the
        list is one representation of the tokens in the ``TextField`` (e.g., word IDs, or character
        IDs per word).  This lets us be flexible in specifying the model, changing the
        representation used for ``TextFields`` in your data, and the way that those representations
        are embedded, without changing any model code at all.  The recommended way to leverage this
        abstraction in a model like Bidirectional Attention Flow, which does passage-based question
        answering, is something like this::

            # In the model code.  This assumes a data type that has four ``Fields``: "passage",
            # "question", "span_begin", and "span_end".
            def forward(self, inputs):
                passage_tokens = inputs['passage']
                embedded_passage = nlp_api.embed_text_field(passage_tokens)
                question_tokens = inputs['question']
                embedded_question = nlp_api.embed_text_field(question_tokens)

        Note that there's nothing in here that says `how` exactly the embedding happens, which
        means you don't have to make that decision in your model code.  Instead, you make it later,
        in an experiment configuration file, and can change it easily to try out different word
        representations.  TODO(matt): give a pointer to documentation on how to configure this
        correctly.  That code and associated documentation isn't written yet, though.

        Parameters
        ----------
        input_tensors : List[torch.Tensor]
            The tensors produced by a ``TextField`` in your data.
        name : str
            The name of the embedding you want to use.  Typically this should correspond to the
            name of the input ``TextField``, but it doesn't have to.  It `does` have to match a
            name that was a key in the ``embedders`` dictionary passed to the constructor of
            ``NlpApi``.  If the name is not present in that dictionary, the behavior is determined
            by the ``fallback_behavior`` parameter.
        fallback_behavior : ``str``, optional (default=``"crash"``)
            Determines what to do when ``name`` is not a key in ``embedders``.  There are two
            options:

            - ``"crash"``: raise an error.  This is the default.  The intention is to help you find
              bugs - if you specify a particular embedding name when constructing your model
              without giving a fallback behavior, you probably wanted to use a particular set of
              modules, so we crash if they are not provided.
            - ``"use [name]``: In this case, we `reuse` the token embedders with the given name,
              essentially replacing the ``name`` given as an argument to this function with the
              name in the fallback behavior.  You should use this fallback behavior to share
              embedding layers across inputs.

            An alternative usage of this API to the BiDAF example given above, that allows for more
            flexibility in using different embedded representations for different fields, would be
            something like this::

                def forward(self, inputs):
                    passage_tokens = inputs['passage']
                    embedded_passage = nlp_api.embed_text_field(passage_tokens,
                                                                "passage",
                                                                fallback_behavior="use default")
                    question_tokens = inputs['question']
                    embedded_question = nlp_api.embed_text_field(question_tokens,
                                                                 "question",
                                                                 fallback_behavior="use passage")

            This would let you specify `different` embedding behaviors for the "passage" and
            "question" fields in your configuration file, but still falling back to using a shared
            default embedding if no special configuration was given for "passage" or "question"
            embeddings.
        """
        if name not in self._embedders:
            if fallback_behavior == "crash":
                raise ConfigurationError("No embedders specified for name: %s" % name)
            elif fallback_behavior.startswith("use "):
                name_to_use = fallback_behavior[4:]
                self._embedders[name] = self._embedders[name_to_use]
            else:
                raise ConfigurationError("Unrecognized fallback behavior: %s" % fallback_behavior)
        embedders = self._embedders[name]

        if isinstance(embedders, torch.nn.Module):
            return embedders.forward(input_tensors)
        assert len(embedders) == len(input_tensors), \
                "Mismatch between number of TokenIndexers (%d) and number of embedding modules (%d) " \
                % (len(embedders), len(input_tensors))
        embedded_arrays = []

        for embedder, input_array in zip(embedders, input_tensors):
            embedded_arrays.append(embedder.forward(input_array))
        if len(input_tensors) == 1:
            return embedded_arrays[0]
        else:
            return torch.cat(embedded_arrays, -1)

    def get_recurrent_layer(self,
                            name: str = "default",
                            fallback_behavior: str = "crash") -> torch.nn.Module:
        """
        This method abstracts away the decision of which particular recurrent layer you want to use
        when building your model, and allows for easy experimentation with different recurrent
        layers.  All we do is a simple re-direction, where you specify a recurrent layer by name
        (e.g., "passage_encoder" or "question_encoder"), and we return the layer with that name
        that was passed to the constructor of ``NlpApi``.  If there was no layer with the name
        you gave, there are a few things we could do, determined by ``fallback_behavior``.

        Parameters
        ----------
        name : ``str``, optional (default=``"default"``)
            The name of the recurrent layer, e.g., "sentence_encoder".  Multiple calls to this
            method with the same name will return the same recurrent layer.  If ``name`` is not a
            key in the ``recurrent_layers`` passed to :func:`__init__`, the behavior is defined by
            the ``fallback_behavior`` parameter.
        fallback_behavior : ``str``, optional (default=``"crash"``)
            Determines what to do when ``name`` is not a key in ``recurrent_layers``.  There are
            three options:

            - ``"crash"``: raise an error.  This is the default.  The intention is to help you find
              bugs - if you specify a particular encoder name when constructing your model without
              giving a fallback behavior, you probably wanted to use a particular recurrent layer,
              so we crash if it is not provided.
            - ``"new default layer"``: In this case, we return a new encoder created with the
              ``default_recurrent_layer`` function passed to the constructor.  If this is your
              fallback behavior, you will get a `new` layer by default when you give a new name.
              You should use this for things like stacked recurrent layers.
            - ``"use [name]``: In this case, we `reuse` the layer with the given name, essentially
              replacing the ``name`` given as an argument to this function with the name in the
              fallback behavior.  You should use this fallback behavior for recurrent layers that
              should be shared between input fields by default, e.g., for encoding a passage and a
              question using the same RNN.

            For example, in the Bidirectional Attention Flow reading comprehension model, there are
            several biLSTMs used in various contexts: one to encode the question, one to encode the
            passage, and several stacked layers later on in the model that are applied to modified
            passage representations.  The recommended approach to setting ``fallback_behavior`` for
            these RNNs is something like this::

                # in __init__ for the BiDAF model code
                self.question_encoder = nlp_api.get_recurrent_layer("question_encoder", "new default layer")
                self.passage_encoder = nlp_api.get_recurrent_layer("passage_encoder", "use passage_encoder")
                self.modeling_layers = [nlp_api.get_recurrent_layer("modeling_layer_$i", "new default layer")
                                        for i in num_modeling_layers]

            Done this way, all of these layers can be modified individually in a parameter file, if
            you want to heavily tweak things, or you can just specify a single default layer type
            to use for all recurrent layers, and by default the question and passage encoders will
            be shared, and all other layers will have their own parameters.
        """
        if name in self._recurrent_layers:
            return self._recurrent_layers[name]
        elif fallback_behavior == "crash":
            raise ConfigurationError("No recurrent layer specified for name: %s" % name)
        elif fallback_behavior == "new default layer":
            self._recurrent_layers[name] = self._default_recurrent_layer()
            return self._recurrent_layers[name]
        elif fallback_behavior.startswith("use "):
            name_to_use = fallback_behavior[4:]
            self._recurrent_layers[name] = self._recurrent_layers[name_to_use]
            return self._recurrent_layers[name]
        else:
            raise ConfigurationError("Unrecognized fallback behavior: %s" % fallback_behavior)

    def get_recurrent_cell(self,
                           name: str = "default",
                           fallback_behavior: str = "crash") -> torch.nn.Module:
        """
        This method abstracts away the decision of which particular recurrent cells you want to use
        when building your model, and allows for easy experimentation with different recurrent
        cells.  This method is basically identical to :func:`get_recurrent_layer`, but for cells
        instead of full RNNs.  See the documentation there for more detail about how this function
        works.

        Parameters
        ----------
        name : ``str``, optional (default=``"default"``)
            The name of the recurrent cell, e.g., "question_tree_cell".
        fallback_behavior : ``str``, optional (default=``"crash"``)
            Determines what to do when ``name`` is not a key in ``recurrent_cells``.  There are
            three options: "crash", "new default cell", and "use [name]".  They behave identically
            to :func:`get_recurrent_layer`, so see the descriptions there.
        """
        if name in self._recurrent_cells:
            return self._recurrent_cells[name]
        elif fallback_behavior == "crash":
            raise ConfigurationError("No recurrent cell specified for name: %s" % name)
        elif fallback_behavior == "new default cell":
            self._recurrent_cells[name] = self._default_recurrent_cell()
            return self._recurrent_cells[name]
        elif fallback_behavior.startswith("use "):
            name_to_use = fallback_behavior[4:]
            self._recurrent_cells[name] = self._recurrent_cells[name_to_use]
            return self._recurrent_cells[name]
        else:
            raise ConfigurationError("Unrecognized fallback behavior: %s" % fallback_behavior)

    @classmethod
    def from_params(cls, params: Params):
        # TODO(matt): actually implement this.
        raise NotImplementedError
