from datetime import datetime


def timestamp() -> str:
    dt = datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")
