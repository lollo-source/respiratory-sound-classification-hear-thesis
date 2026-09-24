from .io_utils import write_json


def table(label, rows):
    return {"label": label, "rows": rows}


def save_table(path, label, rows):
    payload = table(label, rows)
    write_json(path, payload)
    return payload

