"""Train a model on synthetic data and save it."""
import argparse
from pathlib import Path

from .data_loader import generate_synthetic_data
from .model import train_model, save_model


def main(output: str, samples: int):
    df = generate_synthetic_data(n_samples=samples)
    model, metrics = train_model(df)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output)
    print(f"Saved model to {output}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/model.joblib")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    main(args.output, args.samples)
