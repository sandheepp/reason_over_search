import re
from typing import Tuple


def extract_search_tag_query(text: str) -> Tuple[str, str]:
    # Match Search-R1 generation.py:postprocess_predictions — first match, not last.
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if not match:
        return "", "no_search_tag"
    query = match.group(1).strip()
    if not query:
        return "", "empty_search_query"
    return query, "valid_search_tag"

