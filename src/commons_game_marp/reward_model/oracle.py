from typing import Dict, Tuple


def compute_phi(metrics: Dict[str, float], phi_key: str) -> float:
    if phi_key == "efficiency":
        return float(metrics["efficiency"])
    if phi_key == "efficiency_x_peace":
        return float(metrics["efficiency"]) * float(metrics["peace"])
    if phi_key == "efficiency_x_peace_x_equality":
        return (
            float(metrics["efficiency"]) * float(metrics["peace"]) * float(metrics["equality"])
        )
    if phi_key == "efficiency_x_equality":
        return float(metrics["efficiency"]) * float(metrics["equality"])
    if phi_key == "efficiency_x_sustainability":
        return float(metrics["efficiency"]) * float(metrics["sustainability"])
    if phi_key == "equality_x_peace":
        return float(metrics["equality"]) * float(metrics["peace"])
    if phi_key == "efficiency_x_peace_x_equality_x_sustainability":
        return (
            float(metrics["efficiency"])
            * float(metrics["peace"])
            * float(metrics["equality"])
            * float(metrics["sustainability"])
        )
    raise ValueError(f"Unsupported phi_key: {phi_key}")


def preference(phi_i: float, phi_j: float, tie_tolerance: float = 0.0) -> Tuple[float, float]:
    """Oracle preference label for a pair of episodes.

    Returns `(mu, delta)` where `mu` is the Bradley-Terry target probability
    that episode `i` is preferred and `delta = |phi_i - phi_j|` is the
    preference magnitude used to weight the pair in the loss.

    Ties map to `mu = 0.5`, not to `1`. Ties are not a corner case here: early
    in training whole batches of episodes share `efficiency == 0`, so `phi_i`
    and `phi_j` are bit-identical and a hard `phi_i >= phi_j` label taught the
    model that whichever episode happened to be drawn first was genuinely
    better. That is pure label noise, and it is the loudest signal in the
    warmup phase. `mu = 0.5` makes an indistinguishable pair contribute a
    constant `log 2` with no gradient, which is the correct behaviour.
    """
    delta = abs(phi_i - phi_j)
    if delta <= tie_tolerance:
        return 0.5, delta
    return (1.0 if phi_i > phi_j else 0.0), delta
