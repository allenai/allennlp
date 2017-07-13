from collections import OrderedDict
from typing import Dict, Type # mypy: disable=unused-import
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

# The first item added here will be used as the default in some cases.
# pylint: disable=invalid-name
token_indexers = OrderedDict()  # type: Dict[str, Type[TokenIndexer]]
# pylint: enable=invalid-name

token_indexers['single id'] = SingleIdTokenIndexer
token_indexers['characters'] = TokenCharactersIndexer
