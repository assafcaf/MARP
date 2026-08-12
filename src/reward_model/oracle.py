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


def preference(phi_i: float, phi_j: float) -> Tuple[int, float]:
    mu = 1 if phi_i >= phi_j else 0
    delta = abs(phi_i - phi_j)
    return mu, delta
