import re
from typing import List
from encoding_schemes.base_scheme import BaseEncodingScheme


class SynonymScheme(BaseEncodingScheme):
    """
    First matched pair determines the bit.
    bit 0 -> first term in pair
    bit 1 -> second term in pair
    """

    name = "synonym"
    scheme_type = "synonym"
    bits_per_message = 1

    PAIRS = [
        ("therefore", "thus"),
        ("calculate", "compute"),
        ("obtain", "get"),
        ("equals", "is equal to"),
        ("substitute", "plug in"),
        ("yields", "gives"),
        ("we have", "we get"),
        ("solving", "working out"),
    ]

    def encode(self, text: str, bits: List[int]) -> str:
        target = bits[0]
        out = text

        # Try to swap the first available match
        for a, b in self.PAIRS:
            if target == 0:
                # ensure first element appears earliest
                if re.search(rf"\b{re.escape(b)}\b", out, re.I):
                    out = re.sub(rf"\b{re.escape(b)}\b", a, out, count=1, flags=re.I)
                    return out
                if re.search(rf"\b{re.escape(a)}\b", out, re.I):
                    return out
            else:
                if re.search(rf"\b{re.escape(a)}\b", out, re.I):
                    out = re.sub(rf"\b{re.escape(a)}\b", b, out, count=1, flags=re.I)
                    return out
                if re.search(rf"\b{re.escape(b)}\b", out, re.I):
                    return out

        # no match found: prepend a natural-ish synonym opportunity
        prefix = "Therefore, " if target == 0 else "Thus, "
        return prefix + out

    def decode(self, text: str) -> List[int]:
        lower = text.lower()

        first_pos = None
        bit = 0
        for a, b in self.PAIRS:
            pa = lower.find(a)
            pb = lower.find(b)

            if pa != -1 and (first_pos is None or pa < first_pos):
                first_pos = pa
                bit = 0
            if pb != -1 and (first_pos is None or pb < first_pos):
                first_pos = pb
                bit = 1

        return [bit]