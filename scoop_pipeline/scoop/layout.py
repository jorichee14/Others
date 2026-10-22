"""Where a pass is in data/{raw,work,processed}/:

    <date>/<pass>/...            e.g. 20260924/mapping, 20260924/survey_1
    <date>/<pass>/site_<N>/...   e.g. 20261015/survey/site_1: a pass at one of
                                 several sites that day, each its own pass

The pass is its folder(s) below the date; its tag, which names its outputs and
its merged bag, is <pass>_<date> with the folders joined by "_"
(survey_site_1_20261015)."""
import os
import re

SITE = re.compile(r"site_\w+")
LEVELS = ("raw", "work", "processed")


def pass_parts(path, below=0):
    """(data root, date, [pass folders]) of a path below
    data/{raw,work,processed}/<date>/<pass>[/site_*]/, or None. ``below``:
    how many folders the path must have under the pass (a bag in a pass: 1)."""
    parts = os.path.abspath(os.path.expanduser(str(path))).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in LEVELS and len(parts) > i + 2:
            rest = parts[i + 2:]
            pas = rest[:2] if len(rest) > 1 and SITE.fullmatch(rest[1]) else rest[:1]
            if len(rest) - len(pas) >= below:
                return os.sep.join(parts[:i]), parts[i + 1], pas
    return None


def tag(date, pas):
    """<pass>_<date>, the pass's folders joined by "_"."""
    return "%s_%s" % ("_".join(pas), date)
