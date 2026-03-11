from .synonym import synonym_encode
from .structural import structural_encode
from .parity import parity_encode
from .recommendation import recommendation_encode
from .number_format import number_format_encode
from .keyword_presence import keyword_presence_encode
from .partitions import PARTITIONS

SCHEMES = {
    "synonym": synonym_encode,
    "structural": structural_encode,
    "parity": parity_encode,
    "recommendation": recommendation_encode,
    "number_format": number_format_encode,
    "keyword_presence": keyword_presence_encode,
}
