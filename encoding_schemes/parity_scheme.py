from typing import List
from encoding_schemes.base_scheme import BaseEncodingScheme


class ParityScheme(BaseEncodingScheme):
    name = "parity"
    scheme_type = "parity"
    bits_per_message = 1

    def encode(self, text: str, bits: List[int]) -> str:
        target = bits[0]
        words = text.split()

        if len(words) % 2 != target:
            words.append("indeed")

        return " ".join(words)

    def decode(self, text: str) -> List[int]:
        words = text.split()
        return [len(words) % 2]