"""
Executors are classes that deterministically transform programs in domain specific languages
into denotations. We have one executor defined for each language-domain pair that we handle.
"""
from allennlp.semparse.executors.wikitables_sempre_executor import WikiTablesSempreExecutor
from allennlp.semparse.executors.sql_executor import SqlExecutor
from allennlp.semparse.executors.nlvr_executor import NlvrExecutor
from allennlp.semparse.executors.wikitables_variable_free_executor import WikiTablesVariableFreeExecutor
