"""
This module contains classes related to handling data for semantic parsing. That includes knowledge graphs, type
declarations, and world representations for all the domains for which we want to build semantic parsers.
"""

from allennlp.semparse.knowledge_graphs.knowledge_graph import KnowledgeGraph
from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.action_space_walker import ActionSpaceWalker
