from .synonym import synonym_encode
from .structural import structural_encode
from .parity import parity_encode

SCHEMES = {
    "synonym": synonym_encode,
    "structural": structural_encode,
    "parity": parity_encode,
}
