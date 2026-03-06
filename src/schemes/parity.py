"""Word count parity encoding: sanity-check baseline."""


def parity_encode(response: str) -> int:
    """Return 0 if word count is even, 1 if odd."""
    return len(response.split()) % 2
