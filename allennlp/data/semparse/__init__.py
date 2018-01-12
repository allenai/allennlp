"""
This module contains classes related to handling data for semantic parsing. That includes knowledge graphs, type
declarations, and world representations for all the domains for which we want to build semantic parsers.
"""

from allennlp.data.semparse.knowledge_graphs.knowledge_graph import KnowledgeGraph
from allennlp.data.semparse.worlds.world import ParsingError, World
