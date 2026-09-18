import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=2000, random_state=42):
    """Generate synthetic network + energy data for prototyping.

    Columns:
    - traffic_load: normalized [0,1]
    - active_cells: int
    - time_of_day: hour 0-23
    - ambient_temp: Celsius
    - qos_latency: ms (lower is better)
    - qos_throughput: Mbps (higher is better)
    - energy_consumption: kW (target to predict/optimize)
    """
    rng = np.random.RandomState(random_state)
    traffic = rng.rand(n_samples)
    active_cells = rng.randint(10, 100, size=n_samples)
    time_of_day = rng.randint(0, 24, size=n_samples)
    ambient_temp = rng.normal(25, 5, size=n_samples)

    # QoS proxies
    qos_latency = 10 + (1 - traffic) * 20 + rng.normal(0, 2, size=n_samples)
    qos_throughput = 100 * traffic * (active_cells / 100.0) + rng.normal(0, 5, size=n_samples)

    # Energy model (synthetic): base + load*scale + temp effect + noise
    base = 5.0
    energy = base + 8.0 * traffic + 0.01 * active_cells + 0.02 * np.maximum(0, ambient_temp - 20)
    energy += rng.normal(0, 0.5, size=n_samples)

    df = pd.DataFrame(
        {
            "traffic_load": traffic,
            "active_cells": active_cells,
            "time_of_day": time_of_day,
            "ambient_temp": ambient_temp,
            "qos_latency": qos_latency,
            "qos_throughput": qos_throughput,
            "energy_consumption": energy,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_synthetic_data(100)
    print(df.head())
