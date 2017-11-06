"""
Utilities for creating permalinks.
"""
import base64
import binascii
import logging
from typing import Optional, NamedTuple

from allennlp.common.util import JsonDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

Permadata = NamedTuple("Permadata", [("model_name", str),
                                     ("request_data", JsonDict),
                                     ("response_data", JsonDict)])

def int_to_slug(i: int) -> str:
    """
    Turn an integer id into a semi-opaque string slug
    to use as the permalink.
    """
    byt = str(i).encode('utf-8')
    slug_bytes = base64.urlsafe_b64encode(byt)
    return slug_bytes.decode('utf-8')

def slug_to_int(slug: str) -> Optional[int]:
    """
    Convert the permalink slug back to the integer id.
    Returns ``None`` if slug is not well-formed.
    """
    byt = slug.encode('utf-8')
    try:
        int_bytes = base64.urlsafe_b64decode(byt)
        return int(int_bytes)
    except (binascii.Error, ValueError):
        logger.error("Unable to interpret slug: %s", slug)
        return None
