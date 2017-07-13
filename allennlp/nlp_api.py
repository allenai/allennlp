from typing import Callable, Dict

from torch.nn import Module

from .common import Params


class NlpApi:
    """
    The ``NlpApi`` provides a high-level API for building NLP models.  You don't have to use it
    for building a model with AllenNLP, but we think you should, because it's useful.

    ``NlpApi`` abstracts away particular decisions that are frequently necessary in NLP, like how
    exactly words get represented as vectors, or what kind of RNN or other encoder should be used
    when processing sequences of word vectors.  When building a model, you just call a function
    like :func:`get_token_embedder`, or :func:`get_seq2vec_encoder`, and leave the decision of
    `how` exactly things are embedded or encoded for later.  This makes it easy to define a `class`
    of models in your model code, and easily run controlled experiments within this class later.
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
    2. ``get_seq2seq_encoder()``.  This method returns a ``Module`` that transforms sequences of
       vectors into new sequences of vectors, encoding context from other elements in the sequence.
       These ``Modules`` take as input a ``Tensor`` with shape ``(batch_size, num_tokens,
       embedding_dim)``, and return a ``Tensor`` with the same shape.  These are typically
       recurrent layers that return a vector for every input vector they receive.  Having this as
       an abstraction lets you easily switch between LSTMs, bi-LSTMs, GRUs, or some new recurrent
       layer that someone comes up with.
    3. ``get_seq2vec_encoder()``.  This method returns a ``Module`` that transforms sequences of
       vectors into a single vector.  The difference between this method and
       :func:`get_seq2seq_encoder` is the return value: both ``get_seq2vec_encoder`` and
       ``get_seq2seq_encoder`` return ``Modules`` that `encode` sequences, but ``Modules`` from
       this method return a `single vector`, instead of a sequence of vectors.  Applicable modules
       here are CNNs, tree-structured recursive networks, RNNs where you just take the last hidden
       state or do some pooling operation over the sequence, etc.

    All of these functions behave similarly, and they function largely as python dictionaries, just
    doing some simple redirection.  You give the function a name, and the function returns a
    ``Module`` or ``None``.  The name should correspond to a location in your model, such as
    "passage_tokens", "question_encoder", "modeling_layer_2", etc.  Each time you call these APIs
    in different parts of your model code, you should typically pass a unique name to the function.
    When constructing this ``NlpApi`` object, you map those names to concrete ``Modules`` that will
    be used to perform the function.  In your model, you should check whether the object you got
    was ``None``, and fall back to some other option if it was.

    For example, if we're writing model code for the Bidirectional Attention Flow model (BiDAF) for
    reading comprehension on the Stanford Question Answering Dataset (SQuAD), we might have code
    that looks something like this::

        def forward(self, question, passage, answer_span_begin=None, answer_span_end=None):
            # We ask for the "base_embedder" token embedder, and if there was nothing specified, we
            # use the ``token_embedder_fn`` passed to the construct to create a new embedding
            # module.  Note that basic python short-circuits ``or`` statements,
            # ``get_new_token_embedder()`` will not get evaluated if there's a "base_embedder".
            embedder = nlp_api.get_token_embedder("base_embedder") or nlp_api.get_new_token_embedder()
            # We know we always want to share the embedding layer, so we just reuse it.
            embedded_question = embedder(question)
            embedded_passage = embedder(passage)

            # This is the first encoder in BiDAF, and it's typically shared between both the
            # question and the passage.  We ask for the "base_encoder", but fall back to a new
            # encoder created by the ``seq2seq_encoder_fn`` passed to the constructor.
            encoder = nlp_api.get_seq2seq_encoder("base_encoder") or nlp_api.get_new_seq2seq_encoder()
            encoded_question = encoder(embedded_question)
            encoded_passage = encoder(embedded_passage)

            # ... do bidirectional attention piece, merge passage and question representations ...

            modeled_passage = merged_passage_representation
            for i in range(self.num_modeling_layers):
                # Here we get a number of stacked RNNs.  If you want to, you can specify different
                # architectures, widths, or anything else, by giving different encoders as keys for
                # "modeling_layer_1", "modeling_layer_2", etc.  Or, you could keep things simple
                # and just not provide any of those keys, so this code will just create a new
                # encoder for each stacked layer using ``seq2seq_encoder_fn``.
                # Note that pytorch has depth as just a parameter to the RNN in some cases, you
                # might be able to get away with just a single ``encoder`` here with more depth,
                # though that's less flexible later.
                modeling_layer = nlp_api.get_seq2seq_encoder("modeling_layer_%d" % i) or \
                        nlp_api.get_new_seq2seq_encoder()
                modeled_passage = modeling_layer(modeled_passage)

            # ... predict span begin and span end, compute loss, return outputs ...

    To experiment with embedding and encoding options using this model code, and the ``NlpApi``
    object, you could instantiate the ``NlpApi`` object with these parameters::

        # Actual module instantiations here are made up, but this is the basic idea.
        token_embedder_fn = lambda: TokenEmbedder(pretrained_vectors=glove, character_dim=8)
        seq2seq_encoder_fn = lambda: BiLstm(hidden_dim=256)

        # These arguments can all be omitted, because they are unused, and the defaults work.
        token_embedders = {}
        seq2vec_encoders = {}
        seq2seq_encoders = {}
        seq2vec_encoder_fn = None

    This would result in using pretrained glove vectors concatenated with a character-level
    representation to represent words, and using BiLSTMs with width 256 for every encoder (note
    again, though, that this is hypothetical - that's not how you actually instantiate those
    layers).

    Or you could use these parameters to get different behavior (unused arguments omitted this
    time)::

        token_embedder_fn = lambda: MyFancyTransferLearningWordRepresentationModule()
        seq2seq_encoders = {'base_encoder': RecurrentAdditiveNetwork(some_params)}
        seq2seq_encoder_fn = lambda: BiGru(hidden_dim=512)

    Now, without changing the model code at all, I am experimenting with my fancy new transfer
    learning algorithm, trying out a different encoder for the initial encoding of the question and
    passage, and just tweaking the RNN in the modeling layer a bit (GRUs instead of LSTMs, a
    little wider).

    If I wanted the model to allow for even more flexibility, I could use the API a little bit
    differently::

        def forward(self, question, passage, answer_span_begin=None, answer_span_end=None):
            # This is our first token embedder, so we get a new one if there's no "question"
            # embedder.
            question_embedder = nlp_api.get_token_embedder("question") or \
                    nlp_api.get_new_token_embedder()
            embedded_question = question_embedder(question)
            # Now, we ask for an embedder with a different name, but re-use `question_embedder` if
            # there is nothing specified.
            passage_embedder = nlp_api.get_token_embedder("passage") or question_embedder
            embedded_passage = passage_embedder(passage)

            # We do a similar thing for the encoding layers.  Instead of requesting a shared
            # "base_encoder", we use two different names, but share them if you didn't specify two
            # different encoders when instantiating the ``NlpApi`` object.
            question_encoder = nlp_api.get_seq2seq_encoder("question_encoder") or \
                    nlp_api.get_new_seq2seq_encoder()
            encoded_question = question_encoder(embedded_question)
            passage_encoder = nlp_api.get_seq2seq_encoder("passage_encoder") or question_encoder
            encoded_passage = passage_encoder(embedded_passage)

            # ... the rest of the code is the same as above ...

    This is more flexible, because I can now use separate word representations for the question and
    the passage, if I want to (e.g., maybe I have a question-specific representation that I
    pre-trained on a large collection of questions, and I want to try it out), and I can use
    different encoders for the question and the passage (the passage is a whole lot longer, so
    maybe something that has more hidden state to remember context is warranted).

    To get this different behavior, all I have to do is pass different parameters when constructing
    ``NlpApi``::

        token_embedders = {'question': MyFancyQuestionRepresentation(),
                           'passage': TokenEmbedder(pretrained_vectors=glove, character_dim=8)}
        seq2seq_encoders = {'question_encoder': MultiplicativeLstm(hidden_dim=128),
                            'passage_encoder': MultiplicativeLstm(hidden_dim=512)}
        seq2seq_encoder_fn = lambda: BiLstm(hidden_dim=256)  # for the modeling layers

    This flexibility doesn't cost us anything, either, except a slightly more complicated
    ``forward`` method - I can still get the same behavior as in the first, simple example::

        token_embedder_fn = lambda: TokenEmbedder(pretrained_vectors=glove, character_dim=8)
        seq2seq_encoder_fn = lambda: BiLstm(hidden_dim=256)

    Those parameters will still result in a single, shared token embedding layer and a shared
    BiLSTM encoder.

    Our recommendation is to err on the side of flexibility; unless you're certain that you will
    `always` want to share a particular layer, just query the API with a new name, and fall back to
    sharing the layer if you don't get anything back.

    Note that in practice, we'll likely instantiate the ``NlpApi`` object for you from a parameter
    file.  You can see documentation for how to specify those parameters [here](TODO(mattg)).

    Parameters
    ----------
    token_embedders : ``Dict[str, Module]``, optional (default=``{}``)
    token_embedder_fn : ``Callable[[], Module]``, optional (default=``None``)
    seq2seq_encoders : ``Dict[str, Module]``, optional (default=``{}``)
    seq2seq_encoder_fn : ``Callable[[], Module]``, optional (default=``None``)
    seq2vec_encoders : ``Dict[str, Module]``, optional (default=``{}``)
    seq2vec_encoder_fn : ``Callable[[], Module]``, optional (default=``None``)
    """
    def __init__(self,
                 token_embedders: Dict[str, Module] = None,
                 token_embedder_fn: Callable[[], Module] = None,
                 seq2seq_encoders: Dict[str, Module] = None,
                 seq2seq_encoder_fn: Callable[[], Module] = None,
                 seq2vec_encoders: Dict[str, Module] = None,
                 seq2vec_encoder_fn: Callable[[], Module] = None) -> None:
        self._token_embedders = token_embedders or {}
        self._token_embedder_fn = token_embedder_fn
        self._seq2seq_encoders = seq2seq_encoders or {}
        self._seq2seq_encoder_fn = seq2seq_encoder_fn
        self._seq2vec_encoders = seq2vec_encoders or {}
        self._seq2vec_encoder_fn = seq2vec_encoder_fn

    def get_token_embedder(self, name: str) -> Module:
        """
        This method abstracts away the decision of how word representations are obtained in your
        model.  Use this method to get a module that can be applied to arrays from a ``TextField``,
        to get a sequence of vectors in return, with shape ``(batch_size, num_tokens,
        embedding_dim)``.

        See the class docstring for usage info.
        """
        return self._token_embedders.get(name)

    def get_new_token_embedder(self):
        """
        Returns a new token embedder ``Module``, instantiated from the ``token_embedder_fn` passed
        to the constructor.  See the class docstring for usage info.
        """
        return self._token_embedder_fn()

    def get_seq2seq_encoder(self, name: str) -> Module:
        """
        This method abstracts away the decision of how exactly vector sequences get transformed
        into other vector sequences when building your model, and allows for easy experimentation
        with different encoders.

        Use this method to get a module that takes as input sequences of vectors with shape
        ``(batch_size, sequence_length, encoding_dim)``, and returns sequences of vectors with the
        same shape, having modified each element in the sequence by incorporating context from
        surrounding elements.

        See the class docstring for usage info.
        """
        return self._seq2seq_encoders.get(name)

    def get_new_seq2seq_encoder(self):
        """
        Returns a new ``Module`` that encodes sequences of vectors into new sequences of vectors,
        instantiated from the ``seq2seq_encoder_fn` passed to the constructor.  See the class
        docstring for usage info.
        """
        return self._seq2seq_encoder_fn()

    def get_seq2vec_encoder(self, name: str) -> Module:
        """
        This method abstracts away the decision of how exactly vector sequences get encoded as
        vectors when building your model, and allows for easy experimentation with different
        encoders.

        This method abstracts away the decision of which particular sentence encoder you want to
        use when building your model.  Use this method to get a module that can be applied to
        sequences of vectors with shape ``(batch_size, sequence_length, encoding_dim)`` and returns
        a single vector, with shape ``(batch_size, result_encoding_dim)``.

        See the class docstring for usage info.
        """
        return self._seq2vec_encoders.get(name)

    def get_new_seq2vec_encoder(self):
        """
        Returns a new ``Module`` that encodes sequences of vectors into single vectors,
        instantiated from the ``seq2vec_encoder_fn` passed to the constructor.  See the class
        docstring for usage info.
        """
        return self._seq2vec_encoder_fn()

    @classmethod
    def from_params(cls, params: Params):
        # TODO(mattg): actually implement this.
        raise NotImplementedError
