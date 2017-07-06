from typing import Callable, Dict

from torch.nn import Module

from .common.checks import ConfigurationError
from .common import Params


class NlpApi:
    """
    The ``NlpApi`` provides a high-level API for building NLP models.  You don't have to use it
    for building a model with AllenNLP, but we think you should, because it's useful.

    ``NlpApi`` abstracts away particular decisions that are frequently necessary in NLP, like how
    exactly words get represented as vectors, or what kind of RNN or other encoder should be used
    when processing sequences of word vectors.  When building a model, you just call a function
    like :func:`get_token_embedder`, or :func:`get_sentence_encoder`, and leave the decision of
    `how` exactly things are embedded or encoded for later.  This makes it easy to define a `class`
    of models in your model code, and easily run controlled experiments with these models later.
    If you come up with a new RNN variant (say, multiplicative LSTMs, or recurrent additive
    networks), or a new way to represent words, you can easily run experiments on all models that
    use this API without changing any model code - you just change the state in the ``NlpApi``,
    which can be done from parameters in an experiment configuration file.

    Key abstractions:

    1. ``get_token_embedder()``.  This method returns a ``Module`` that converts raw ``TextField``
       input into a ``Tensor`` with shape ``(batch_size, num_tokens, embedding_dim)``.  This
       abstraction is key to allowing you to easily switch between word embeddings and
       character-level encoders, or token-in-context embeddings produced by some pre-trained model,
       or any combination of these.  This is, to us, the most useful abstraction in ``NlpApi`` -
       there are a `lot` of options for how you get word repesentations in NLP, and this lets you
       really easily experiment with those options.  Instead of instantiating your own embedding
       layers, or character-level CNNs, you should call :func:`get_token_embedder` to do that for
       you, so it is easy to try new representations later.
    2. ``get_context_encoder()``.  This method returns a ``Module`` that transforms sequences of
       vectors into new sequences of vectors, encoding context from other elements in the sequence.
       These ``Modules`` take as input a ``Tensor`` with shape ``(batch_size, num_tokens,
       embedding_dim)``, and return a ``Tensor`` with the same shape.  These are typically
       recurrent layers that return a vector for every input vector they receive.  Having this as
       an abstraction lets you easily switch between LSTMs, bi-LSTMs, GRUs, or some new recurrent
       layer that someone comes up with.
    3. ``get_sentence_encoder()``.  This method returns a ``Module`` that transforms sequences of
       vectors into a single vector.  The difference between this method and
       :func:`get_context_encoder` is the return value: ``get_context_encoder`` returns a
       `sequence` of vectors, while ``get_sentence_encoder`` returns a `single` vector.  Applicable
       modules here are CNNs, tree-structured recursive networks, RNNs where you just take the last
       hidden state or do some pooling operation over the sequence, etc.

    All of these functions behave similarly: you give the function a name and a fallback behavior.
    The name should correspond to a location in your model, such as "passage_tokens",
    "question_encoder", "modeling_layer_2", etc.  Each time you call these APIs in different parts
    of your model code, you should typically pass a unique name to the function.  When constructing
    this ``NlpApi`` object, you map those names to concrete ``Modules`` that will be used to
    perform the function.  The ``fallback_behavior`` parameter makes it so you can be less verbose
    in constructing that mapping, allowing you to have multiple unique names in your model
    correspond to the same key in the mapping.  This parameter determines what to do when a name
    you requested (like "modeling_layer_2") is not in the mapping you provided.  There are three
    options:

        - ``"crash"``: raise an error.  This is typically the default.  The intention is to help
          you find bugs - if you specify a particular name when constructing your model without
          giving a fallback behavior, you must also provide a mapping for that name in the
          ``NlpApi``.
        - ``"new default"``: In this case, we return a new module created with the ``default_*``
          function passed to the constructor.  If this is your fallback behavior, you will get a
          `new` module by default when you give a new name.  You should use this for things like
          stacked context encoders, or for the first time you get a sentence encoder you want to be
          shared (see example below).
        - ``"use [name]``: In this case, we `reuse` the module with the given name, essentially
          replacing the ``name`` given as an argument to this function with the name in the
          fallback behavior.  You should use this fallback behavior for modules that should be
          shared between input fields by default, e.g., for encoding a passage and a question using
          the same RNN, or for embedding several different TextFields (again, see example below).

    For example, if we're writing model code for the Bidirectional Attention Flow model (BiDAF) for
    reading comprehension on the Stanford Question Answering Dataset (SQuAD), we might have code
    that looks something like this::

        def forward(self, question, passage, answer_span_begin=None, answer_span_end=None):
            # This is our first token embedder, so we get a new default.
            question_embedder = nlp_api.get_token_embedder("question",
                                                           fallback_behavior="new default")
            embedded_question = question_embedder(question)
            # We typically want to just have one shared embedding layer, so we use the question
            # embedder by default.
            passage_embedder = nlp_api.get_token_embedder("passage",
                                                          fallback_behavior="use question")
            embedded_passage = passage_embedder(passage)

            # Again here, this is our first context encoder, so we get a new default.
            question_encoder = nlp_api.get_context_encoder("question_encoder",
                                                            fallback_behavior="new default")
            encoded_question = question_encoder(embedded_question)
            # And we want to share this base context encoder across both the question and the
            # passage, normally, so we "use question_encoder" as our fallback behavior.
            passage_encoder = nlp_api.get_context_encoder("passage_encoder",
                                                          fallback_behavior="use question_encoder")
            encoded_passage = passage_encoder(embedded_passage)

            # ... do bidirectional attention piece, merge passage and question representations
            modeled_passage = merged_passage_representation
            for i in range(self.num_modeling_layers):
                # It would be odd to share stacked RNNs - these should be "new default" for a
                # fallback behavior.  They will all be identical modules, with separate parameters,
                # defined by the ``default_context_encoder`` function.  If you want them to have
                # different structures (e.g., with more or less capacity as you get deeper), you
                # can give them specific structures by passing in a mapping for each encoder name
                # individually.
                modeling_layer = nlp_api.get_context_encoder("modeling_layer_%d" % i,
                                                             fallback_behavior="new default")
                modeled_passage = modeling_layer(modeled_passage)

            # ... predict span begin and span end, compute loss, return outputs

    Then, when constructing the ``NlpApi`` object, you might give arguments that look roughly
    like the following::

        # Actual module instantiations here are made up, but this is the basic idea.
        default_token_embedder = lambda: TokenEmbedder(pretrained_vectors=glove,
                                                       character_dim=8)
        default_context_encoder = lambda: BiLSTM(hidden_dim=256)
        context_encoders = {'question_encoder': MyNewFancyRNN(hidden_dim=512)}

        # These arguments can all be omitted, because they are defaults.
        default_sentence_encoder = None
        token_embedders = {}
        sentence_encoders = {}

    This would result in both the passage and the question using the same embedding layer
    (instantiated from the ``default_token_embedder`` function), the question and passage sharing
    an initial BiGRU encoding layer, and the modeling layers each having their own BiLSTM (each
    instantiated from the ``default_context_encoder`` function).

    Because you almost always just want to use a single embedding module for all of the TextFields
    in your model, we give a default name for ``get_token_embedder``: ``"default"``.  This means
    that you can simplify the code above a little bit::

        def forward(self, question, passage, answer_span_begin=None, answer_span_end=None):
            # By default, we'll look for the "default" key in the ``token_embedders`` mapping.
            embedder = nlp_api.get_token_embedder()
            # We know we always just want one embedding, so we just reuse it.
            embedded_question = embedder(question)
            embedded_passage = embedder(passage)

            # Again here, we know we always want to reuse this encoder, so we just share it
            # explicitly.  This is less flexible later, but makes this code simpler.
            encoder = nlp_api.get_context_encoder("base_encoder", fallback_behavior="new default")
            encoded_question = encoder(embedded_question)
            encoded_passage = encoder(embedded_passage)

            # the rest of the code is the same as above

    And the parameters would change a bit to look like this::

        # Actual module instantiations here are made up, but this is the basic idea.
        token_embedders = {"default": TokenEmbedder(pretrained_vectors=glove, character_dim=8)}
        default_context_encoder = lambda: BiLSTM(hidden_dim=256)

        # These arguments can all be omitted, because they are defaults.
        default_token_embedder = None
        default_sentence_encoder = None
        sentence_encoders = {}
        context_encoders = {}

    Note that in practice, we'll likely instantiate the ``NlpApi`` object for you from a parameter
    file.  You can see documentation for how to specify those parameters [here](TODO(mattg)).

    Parameters
    ----------
    token_embedders : ``Dict[str, Module]``, optional (default=``{}``)
    default_token_embedder : ``Callable[[], Module]``, optional (default=``None``)
    context_encoders : ``Dict[str, Module]``, optional (default=``{}``)
    default_context_encoder : ``Callable[[], Module]``, optional (default=``None``)
    sentence_encoders : ``Dict[str, Module]``, optional (default=``{}``)
    default_sentence_encoder : ``Callable[[], Module]``, optional (default=``None``)
    """
    def __init__(self,
                 token_embedders: Dict[str, Module] = None,
                 default_token_embedder: Callable[[], Module] = None,
                 context_encoders: Dict[str, Module] = None,
                 default_context_encoder: Callable[[], Module] = None,
                 sentence_encoders: Dict[str, Module] = None,
                 default_sentence_encoder: Callable[[], Module] = None):
        if token_embedders is None:
            token_embedders = {}
        self._token_embedders = token_embedders
        self._default_token_embedder = default_token_embedder
        if context_encoders is None:
            context_encoders = {}
        self._context_encoders = context_encoders
        self._default_context_encoder = default_context_encoder
        if sentence_encoders is None:
            sentence_encoders = {}
        self._sentence_encoders = sentence_encoders
        self._default_sentence_encoder = default_sentence_encoder

    def get_token_embedder(self, name: str = "default", fallback_behavior: str = "crash") -> Module:
        """
        This method abstracts away the decision of how word representations are obtained in your
        model.  Use this method to get a module that can be applied to arrays from a ``TextField``,
        to get a sequence of vectors in return, with shape ``(batch_size, num_tokens,
        embedding_dim)``.

        See the class docstring for usage info, and a description of the parameters.
        """
        return self._get_module_from_dict(self._token_embedders, self._default_token_embedder,
                                          name, fallback_behavior, "token embedder")

    def get_context_encoder(self, name: str, fallback_behavior: str = "crash") -> Module:
        """
        This method abstracts away the decision of which particular context encoder you want to use
        when building your model, and allows for easy experimentation with different context
        encoders.

        Use this method to get a module that takes as input sequences of vectors with shape
        ``(batch_size, sequence_length, encoding_dim)``, and returns sequences of vectors with the
        same shape, having modified each element in the sequence by incorporating context from
        surrounding elements.

        See the class docstring for usage info, and a description of the parameters.
        """
        return self._get_module_from_dict(self._context_encoders, self._default_context_encoder,
                                          name, fallback_behavior, "context encoder")

    def get_sentence_encoder(self, name: str, fallback_behavior: str = "crash") -> Module:
        """
        This method abstracts away the decision of which particular sentence encoder you want to
        use when building your model.  Use this method to get a module that can be applied to
        sequences of vectors with shape ``(batch_size, sequence_length, encoding_dim)`` and returns
        a single vector, with shape ``(batch_size, result_encoding_dim)``.

        See the class docstring for usage info, and a description of the parameters.
        """
        return self._get_module_from_dict(self._sentence_encoders, self._default_sentence_encoder,
                                          name, fallback_behavior, "token embedder")

    @staticmethod
    def _get_module_from_dict(module_dict: Dict[str, Module],
                              default_module_fn: Callable[[], Module],
                              name: str,
                              fallback_behavior: str,
                              module_type: str) -> Module:
        if name in module_dict:
            return module_dict[name]
        elif fallback_behavior == "crash":
            raise ConfigurationError("No %s module specified for name: %s" % (module_type, name))
        elif fallback_behavior == "new default":
            module_dict[name] = default_module_fn()
            return module_dict[name]
        elif fallback_behavior.startswith("use "):
            name_to_use = fallback_behavior[4:]
            module_dict[name] = module_dict[name_to_use]
            return module_dict[name]
        else:
            raise ConfigurationError("Unrecognized fallback behavior: %s" % fallback_behavior)

    @classmethod
    def from_params(cls, params: Params):
        # TODO(mattg): actually implement this.
        raise NotImplementedError
