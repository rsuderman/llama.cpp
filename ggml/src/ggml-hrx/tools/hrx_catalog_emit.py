import json
import re


def require_string(data, key, source):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: expected non-empty string field {key}")
    return value


def require_list(data, key, source):
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: expected non-empty list field {key}")
    return value


def require_int(data, key, source):
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"{source}: expected integer field {key}")
    return value


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def c_array(data):
    rows = []
    for i in range(0, len(data), 12):
        rows.append("    " + ", ".join(f"0x{byte:02x}" for byte in data[i:i + 12]))
    return ",\n".join(rows)


def parse_targets(value):
    if value is None:
        return []
    return [target for target in re.split(r"[;,\s]+", value) if target]


def c_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)
