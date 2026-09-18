"""Simple optimizer that proposes power-reduction actions while checking QoS."""
import numpy as np


def recommend_actions(df, model, max_reduction=0.2, qos_latency_threshold=50.0):
    """For each row, propose a percentage power reduction between 0 and max_reduction.

    This prototype uses a simple heuristic: if predicted energy is high and QoS latency is well below threshold,
    propose reducing power proportionally to headroom.
    """
    X = df[["traffic_load", "active_cells", "time_of_day", "ambient_temp", "qos_latency", "qos_throughput"]]
    preds = model.predict(X)
    actions = []
    for pred, latency in zip(preds, df["qos_latency"]):
        headroom = max(0.0, (qos_latency_threshold - latency) / qos_latency_threshold)
        reduction = float(max_reduction * headroom * (pred / (pred + 1.0)))
        actions.append(reduction)
    return np.array(actions)
