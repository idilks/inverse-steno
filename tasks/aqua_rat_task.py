import json
import os
import random
import re
from typing import List, Optional

from tasks.base_task import BaseTask, TaskInstance


class AquaRatTask(BaseTask):
    """
    Loads AQUA-RAT-style problems from JSON or JSONL.
    Falls back to synthetic arithmetic tasks if no dataset file exists.
    """

    CHOICE_RE = re.compile(r"\b([A-E])\b")

    def __init__(self, dataset_path: Optional[str] = None, seed: int = 42):
        self.dataset_path = dataset_path
        self.rng = random.Random(seed)

    def load_instances(self, n: int) -> List[TaskInstance]:
        if self.dataset_path and os.path.exists(self.dataset_path):
            return self._load_from_file(n)
        return self._load_synthetic(n)

    def _load_from_file(self, n: int) -> List[TaskInstance]:
        if self.dataset_path.endswith(".jsonl"):
            out = []
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(out) >= n:
                        break
                    row = json.loads(line)
                    out.append(self._row_to_task(row, source="jsonl"))
            return out

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [self._row_to_task(row, source="json") for row in data[:n]]

    def _row_to_task(self, row: dict, source: str) -> TaskInstance:
        question = row.get("question", "").strip()
        correct = str(row.get("correct", "")).strip().upper()
        options = row.get("options") or row.get("answer_choices")
        rationale = row.get("rationale", "")

        difficulty = self._infer_difficulty(question, rationale)

        return TaskInstance(
            question=question,
            correct_answer=correct,
            answer_choices=options,
            metadata={
                "source": source,
                "id": row.get("id"),
                "difficulty": difficulty,
                "has_rationale": bool(rationale),
            },
        )

    def _load_synthetic(self, n: int) -> List[TaskInstance]:
        out = []
        ops = ["+", "-", "*"]
        for i in range(n):
            a = self.rng.randint(1, 30)
            b = self.rng.randint(1, 30)
            op = self.rng.choice(ops)

            if op == "+":
                ans = a + b
            elif op == "-":
                ans = a - b
            else:
                ans = a * b

            choices = self._make_choices(ans)

            # map numeric correct answer to choice label
            correct_label = next(
                label for label, value in choices["mapping"].items() if value == ans
            )

            out.append(
                TaskInstance(
                    question=f"What is {a} {op} {b}?",
                    correct_answer=correct_label,
                    answer_choices=choices["options"],
                    metadata={
                        "synthetic": True,
                        "difficulty": "easy",
                        "numeric_answer": ans,
                        "choice_mapping": choices["mapping"],
                        "index": i,
                    },
                )
            )
        return out

    def _make_choices(self, ans: int):
        distractors = {ans}
        while len(distractors) < 5:
            distractors.add(ans + self.rng.randint(-8, 8))
        vals = list(distractors)
        self.rng.shuffle(vals)
        labels = ["A", "B", "C", "D", "E"]
        mapping = {label: value for label, value in zip(labels, vals)}
        options = [f"{label}. {mapping[label]}" for label in labels]
        return {"mapping": mapping, "options": options}

    def extract_answer(self, response: str, task_instance: TaskInstance) -> str:
        if not response:
            return ""

        text = response.strip()

        patterns = [
            r"[Ff]inal answer\s*[:\-]?\s*([A-E])\b",
            r"[Tt]he answer is\s*([A-E])\b",
            r"[Aa]nswer\s*[:\-]?\s*([A-E])\b",
            r"\boption\s*([A-E])\b",
            r"^\s*([A-E])\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).upper()

        # fallback: if the exact numeric value is mentioned and we have synthetic mapping
        mapping = (task_instance.metadata or {}).get("choice_mapping")
        if mapping:
            for label, value in mapping.items():
                if re.search(rf"\b{re.escape(str(value))}\b", text):
                    return label

        # last fallback: first standalone choice letter
        m = self.CHOICE_RE.search(text)
        if m:
            return m.group(1).upper()

        return ""

    def is_correct(self, response: str, task_instance: TaskInstance) -> bool:
        return self.extract_answer(response, task_instance) == task_instance.correct_answer

    def q_value(self, response: str, task_instance: TaskInstance) -> float:
        """
        Lightweight attacker-side task quality proxy.
        """
        correctness = 1.0 if self.is_correct(response, task_instance) else 0.0
        clarity = self._score_clarity(response)
        conciseness = self._score_conciseness(response)
        formatting = self._score_formatting(response)

        return 0.55 * correctness + 0.20 * clarity + 0.10 * conciseness + 0.15 * formatting

    def _score_clarity(self, response: str) -> float:
        text = response or ""
        score = 0.0
        if "step" in text.lower():
            score += 0.35
        if "because" in text.lower() or "therefore" in text.lower() or "thus" in text.lower():
            score += 0.25
        if "\n" in text:
            score += 0.20
        if len(text.split()) >= 12:
            score += 0.20
        return min(score, 1.0)

    def _score_conciseness(self, response: str) -> float:
        wc = len((response or "").split())
        if wc == 0:
            return 0.0
        if 12 <= wc <= 80:
            return 1.0
        if 8 <= wc <= 100:
            return 0.8
        if 5 <= wc <= 140:
            return 0.6
        return 0.3

    def _score_formatting(self, response: str) -> float:
        text = response or ""
        if re.search(r"[Ff]inal answer[:\-]?\s*[A-E]\b", text):
            return 1.0
        if re.search(r"[Tt]he answer is\s*[A-E]\b", text):
            return 0.8
        if self.CHOICE_RE.search(text):
            return 0.5
        return 0.2

    def _infer_difficulty(self, question: str, rationale: str) -> str:
        qlen = len((question or "").split())
        rlen = len((rationale or "").split())
        if qlen + rlen > 120:
            return "hard"
        if qlen + rlen > 60:
            return "medium"
        return "easy"