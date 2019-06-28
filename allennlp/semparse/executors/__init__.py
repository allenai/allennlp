"""
Executors are classes that deterministically transform programs in domain specific languages
into denotations. We have one executor defined for each language-domain pair that we handle.
"""
from allennlp.semparse.executors.sql_executor import SqlExecutor
