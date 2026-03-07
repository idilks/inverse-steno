from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class EncodingResult:
    encoded_response: str
    decoded_bits: List[int]
    round_trip_success: bool
    original_response: str
    encoding_metadata: dict


class BaseEncodingScheme(ABC):
    name: str
    scheme_type: str
    bits_per_message: int

    @abstractmethod
    def encode(self, text: str, bits: List[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def decode(self, text: str) -> List[int]:
        raise NotImplementedError

    def attempt_encoding(self, text: str, bits: List[int]) -> EncodingResult:
        encoded = self.encode(text, bits)
        decoded = self.decode(encoded)
        return EncodingResult(
            encoded_response=encoded,
            decoded_bits=decoded,
            round_trip_success=(decoded == bits),
            original_response=text,
            encoding_metadata=self._extra_metadata(text, encoded, bits, decoded),
        )

    def _extra_metadata(self, original: str, encoded: str, bits: List[int], decoded: List[int]) -> dict:
        return {
            "original_word_count": len(original.split()),
            "encoded_word_count": len(encoded.split()),
            "word_count_diff": len(encoded.split()) - len(original.split()),
            "target_bits": bits,
            "decoded_bits": decoded,
        }