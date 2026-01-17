import json
from typing import Any

def number_to_jcs(n: Any) -> Any:
    """
    Transform numbers to comply with JCS (RFC 8785) formatting rules before serialization.
    - Integers are left as is.
    - Floats that are integers (e.g., 100.0) are converted to int (100).
    - Other floats are standard python representation (which is generally close enough for this level of audit).
      Strict JCS requires ES6 Number.prototype.toString(), which Python's json dump mostly respects.
    """
    if isinstance(n, float):
        if n.is_integer():
            return int(n)
        # Check for NaN or Infinity which are not allowed in JSON
        if n != n or n == float('inf') or n == float('-inf'):
            raise ValueError(f"NaN and Infinity are not allowed in JCS: {n}")
    return n

def recursive_jcs_prep(data: Any) -> Any:
    """
    Recursively prepare data for JCS serialization:
    - Lists are processed recursively.
    - Dictionaries have keys sorted (handled by sort_keys=True in dumps, but we process values).
    - Numbers are formatted.
    """
    if isinstance(data, dict):
        return {k: recursive_jcs_prep(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_jcs_prep(v) for v in data]
    elif isinstance(data, (int, float)):
        return number_to_jcs(data)
    else:
        return data

def jcs_serialize(data: Any) -> bytes:
    """
    Serialize data to bytes using JSON Canonicalization Scheme (RFC 8785).
    
    Rules applied:
    1. Serialization of primitive types (strings, numbers, booleans, nulls).
    2. Sorting of object keys (Lexicographical).
    3. No whitespace (Compact).
    """
    prepared_data = recursive_jcs_prep(data)
    
    # ensure_ascii=False is required for JCS (UTF-8)
    # separators=(',', ':') removes whitespace
    # sort_keys=True orders keys lexicographically
    return json.dumps(
        prepared_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')
