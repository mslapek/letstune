# stupid patch to make letstune work with Python 3.7


def dataclass(*args, **kwargs):
    from dataclasses import dataclass as origf

    forbidden_keys = ["slots", "kw_only"]
    for k in forbidden_keys:
        if k in kwargs:
            kwargs.pop(k)

    return origf(*args, **kwargs)


def removeprefix(s: str, prefix: str) -> str:
    # https://peps.python.org/pep-0616/
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s[:]


def sum(iterable, start=0):
    for x in iterable:
        start += x

    return start


Protocol = object
