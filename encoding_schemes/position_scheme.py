from typing import List
from encoding_schemes.base_scheme import BaseEncodingScheme


class PositionScheme(BaseEncodingScheme):
    """
    Bit 0 => Nth word begins A-M
    Bit 1 => Nth word begins N-Z
    """

    name = "position"
    scheme_type = "position"
    bits_per_message = 1

    SAFE_WORDS_0 = ["alpha", "beta", "choice", "answer", "first", "method"]
    SAFE_WORDS_1 = ["zeta", "number", "result", "next", "option", "value"]

    def __init__(self, position: int = 4):
        self.position = position

    def encode(self, text: str, bits: List[int]) -> str:
        words = text.split()
        while len(words) <= self.position:
            words.append("reasoning")

        target = bits[0]
        current = words[self.position]
        current_char = current[0].lower() if current else "a"

        if target == 0 and current_char > "m":
            words[self.position] = self.SAFE_WORDS_0[0]
        elif target == 1 and current_char <= "m":
            words[self.position] = self.SAFE_WORDS_1[0]

        return " ".join(words)

    def decode(self, text: str) -> List[int]:
        words = text.split()
        if len(words) <= self.position or not words[self.position]:
            return [0]
        ch = words[self.position][0].lower()
        return [0 if ch <= "m" else 1]