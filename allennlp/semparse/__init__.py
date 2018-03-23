"""
This module contains classes related to handling data for semantic parsing. That includes knowledge graphs, type
declarations, and world representations for all the domains for which we want to build semantic parsers.
"""

# We have some circular dependencies with the data code.  Importing this loads all of the data code
# before we try to import stuff under semparse.  This means that we can't do
# `from allennlp.semparse import [whatever]` in the `data` module, but it resolves the other
# dependency issues.  If you want to import semparse stuff from the data code, just use a more
# complete path, like `from allennlp.semparse.worlds import WikiTablesWorld`.
from allennlp.data.tokenizers import Token as _
from allennlp.semparse.knowledge_graphs.knowledge_graph import KnowledgeGraph
from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.action_space_walker import ActionSpaceWalker
