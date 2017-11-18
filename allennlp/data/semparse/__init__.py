"""
This module contains classes related to handling data for semantic parsing. That includes knowledge graphs, type
declarations, and world representations for all the domains for which we want to build semantic parsers.
"""
from allennlp.data.semparse.knowledge_graphs import KnowledgeGraph, TableKnowledgeGraph
from allennlp.data.semparse.type_declarations import wikitables_type_declaration
from allennlp.data.semparse.worlds import WikitablesWorld, NLVRWorld
