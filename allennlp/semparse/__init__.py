"""
This module contains code relating to processing logical forms for semantic parsing.  The main
functionality provided by this code is:

    1. A way of declaring a type system for use in processing logical forms.  This determines which
    (lisp-like) logical forms are valid in your logical form language, and it enables us to parse
    statements in that language into logical expressions.  We rely heavily on NLTK for the logic
    processing here.  This is the ``type_declarations`` module.

    2. A way of specifying extra "context" that's available to the parser / type system.
    Currently, this just consists of a ``KnowledgeGraph`` that can represent table entities, or
    question-specific entities.  These entities can be made available to the type system to allow
    context-specific production rules.  This is the ``contexts`` module.

    3. A way to bundle the two things above together in the context of a particular training /
    testing instance.  This bundling is called a ``World``, and this is the primary means that a
    model should use to interact with the type system.  The ``World`` has methods to get all
    possible actions in any state, all context entities that need to be linked, and so on.  This is
    the ``worlds`` module.

Note that the main reason you would use this code is to get the set of actions that are available
at any point during constrained decoding.  If you have some other means of doing that, you might
not need this code at all.
"""

# We have some circular dependencies with the data code.  Importing this loads all of the data code
# before we try to import stuff under semparse.  This means that we can't do
# `from allennlp.semparse import [whatever]` in the `data` module, but it resolves the other
# dependency issues.  If you want to import semparse stuff from the data code, just use a more
# complete path, like `from allennlp.semparse.worlds import WikiTablesWorld`.
from allennlp.data.tokenizers import Token as _
from allennlp.semparse.common.errors import ParsingError, ExecutionError
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage,
                                                                predicate, predicate_with_side_args)
from allennlp.semparse.worlds.world import World
from allennlp.semparse.action_space_walker import ActionSpaceWalker
