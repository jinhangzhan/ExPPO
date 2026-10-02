"""The frequency helper used by the original answer checker."""

from collections import Counter


def most_common_element(data):
    assert data and len(data) > 0, "Data is empty"
    return Counter(data).most_common(1)[0][0]
