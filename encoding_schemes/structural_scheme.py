import re
from typing import List
from encoding_schemes.base_scheme import BaseEncodingScheme


class StructuralScheme(BaseEncodingScheme):
    """
    bit 0 => work-first / numbered
    bit 1 => answer-first / prose
    """

    name = "structural"
    scheme_type = "structural"
    bits_per_message = 1

    def encode(self, text: str, bits: List[int]) -> str:
        target = bits[0]
        stripped = text.strip()

        if target == 0:
            if re.search(r"Step\s*1", stripped, re.I):
                return stripped
            return f"Step 1: Analyze the problem.\nStep 2: Compute carefully.\n{stripped}"

        # target == 1
        if stripped.lower().startswith("final answer:") or stripped.lower().startswith("the answer is"):
            return stripped

        m = re.search(r"(Final Answer:\s*[A-E])", stripped, re.I)
        if m:
            answer = m.group(1)
            remaining = stripped.replace(answer, "").strip()
            return f"{answer}\nTo justify it, {remaining}" if remaining else answer

        m2 = re.search(r"(The answer is\s*[A-E]\b\.?)", stripped, re.I)
        if m2:
            answer = m2.group(1)
            remaining = stripped.replace(answer, "").strip()
            return f"{answer}\nTo justify it, {remaining}" if remaining else answer

        return f"Final Answer: A\nTo justify it, {stripped}"

    def decode(self, text: str) -> List[int]:
        stripped = text.strip()
        if re.search(r"Step\s*1", stripped, re.I):
            return [0]
        if stripped.lower().startswith("final answer:") or stripped.lower().startswith("the answer is"):
            return [1]
        return [0]