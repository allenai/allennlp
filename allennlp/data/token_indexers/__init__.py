from collections import OrderedDict
from typing import Dict # mypy: disable=unused-import
from .token_indexer import TokenIndexer
from .token_characters_indexer import TokenCharactersIndexer
from .single_id_token_indexer import SingleIdTokenIndexer

# The first item added here will be used as the default in some cases.
# pylint: disable=invalid-name
token_indexers = OrderedDict()  # type: Dict[str, type]
# pylint: enable=invalid-name

token_indexers['single id'] = SingleIdTokenIndexer
token_indexers['characters'] = TokenCharactersIndexer
