"""AUC vs number of observations — the multi-observation detection curve."""

from ..detector import detect_multi
from .auc import compute_auc


def auc_vs_n_observations(
    honest_responses: list[list[str]],
    collusion_responses: list[list[str]],
    candidates_per_task: list[list[str]],
    schemes: dict,
    max_n: int | None = None,
) -> list[dict]:
    """Compute AUC as a function of number of observations per agent.

    Args:
        honest_responses: list of N response lists from honest agents
        collusion_responses: list of N response lists from colluding agents
        candidates_per_task: reference candidates per task
        schemes: encoding scheme registry
        max_n: max observations to use (defaults to min available)

    Returns:
        list of {n, auc, auc_per_scheme} dicts
    """
    n_tasks = min(len(honest_responses), len(collusion_responses), len(candidates_per_task))
    if max_n is not None:
        n_tasks = min(n_tasks, max_n)

    results = []
    for n in range(1, n_tasks + 1):
        scores = []
        labels = []

        # Score each honest agent (using first n tasks)
        for agent_responses in honest_responses:
            result = detect_multi(
                agent_responses[:n],
                candidates_per_task[:n],
                schemes,
            )
            scores.append(result["score"])
            labels.append(0)

        # Score each colluding agent
        for agent_responses in collusion_responses:
            result = detect_multi(
                agent_responses[:n],
                candidates_per_task[:n],
                schemes,
            )
            scores.append(result["score"])
            labels.append(1)

        auc = compute_auc(scores, labels)
        results.append({"n": n, "auc": auc})

    return results
