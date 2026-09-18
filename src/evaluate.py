"""Evaluate a trained model on fresh synthetic data."""
import argparse
from .data_loader import generate_synthetic_data
from .model import load_model
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate(model_path: str, samples: int = 500):
    model = load_model(model_path)
    df = generate_synthetic_data(n_samples=samples, random_state=999)
    X = df[["traffic_load", "active_cells", "time_of_day", "ambient_temp", "qos_latency", "qos_throughput"]]
    y = df["energy_consumption"]
    preds = model.predict(X)
    print("R2:", r2_score(y, preds))
    print("MAE:", mean_absolute_error(y, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    evaluate(args.model, args.samples)
