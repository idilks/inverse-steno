from encoding_schemes.parity_scheme import ParityScheme
from encoding_schemes.position_scheme import PositionScheme
from encoding_schemes.structural_scheme import StructuralScheme
from encoding_schemes.synonym_scheme import SynonymScheme
from encoding_schemes.watermark_proxy_scheme import WatermarkProxyScheme


class SchemeRegistry:
    def __init__(self, schemes):
        self._schemes = {s.name: s for s in schemes}

    @classmethod
    def default(cls):
        return cls([
            ParityScheme(),
            PositionScheme(position=4),
            StructuralScheme(),
            SynonymScheme(),
            WatermarkProxyScheme(),
        ])

    def all_schemes(self):
        return self._schemes

    def summary(self):
        return {
            name: {
                "type": scheme.scheme_type,
                "bits_per_message": scheme.bits_per_message,
            }
            for name, scheme in self._schemes.items()
        }