from typing import List
from encoding_schemes.base_scheme import BaseEncodingScheme


class WatermarkProxyScheme(BaseEncodingScheme):
    """
    Redwood-inspired surface proxy.
    NOT a true logit watermark.
    bit 0 => intro phrase A
    bit 1 => intro phrase B
    """

    name = "watermark_proxy"
    scheme_type = "watermark_proxy"
    bits_per_message = 1

    INTRO_0 = "To begin,"
    INTRO_1 = "At the outset,"

    def encode(self, text: str, bits: List[int]) -> str:
        target = bits[0]
        stripped = text.strip()

        # remove existing proxy intros if present
        for prefix in (self.INTRO_0, self.INTRO_1):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].lstrip(" ,")

        prefix = self.INTRO_0 if target == 0 else self.INTRO_1
        return f"{prefix} {stripped}"

    def decode(self, text: str) -> List[int]:
        stripped = text.strip()
        if stripped.startswith(self.INTRO_1):
            return [1]
        return [0]