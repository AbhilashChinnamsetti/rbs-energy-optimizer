"""Run a short demo: load model, generate data, compute recommendations, and create a short MP4."""
import os
import sys
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure project root is on sys.path so `src` can be imported when running script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import generate_synthetic_data
from src.model import load_model
from src.energy_optimizer import recommend_actions


def make_frames(df, preds, actions, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    n = min(200, len(df))
    for i in range(5, n, 5):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        sns.scatterplot(x=preds[:i], y=df['energy_consumption'].values[:i], ax=axes[0])
        axes[0].set_xlabel('Predicted energy')
        axes[0].set_ylabel('Actual energy')
        axes[0].set_title('Predicted vs Actual (progressive)')

        t = np.arange(i)
        axes[1].plot(t, df['energy_consumption'].values[:i], label='actual')
        axes[1].plot(t, preds[:i], label='predicted')
        axes[1].plot(t, actions[:i] * df['energy_consumption'].values[:i], label='proposed_savings')
        axes[1].legend()
        axes[1].set_title('Time series of energy + proposed savings')

        fname = out_dir / f"frame_{i:04d}.png"
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        frames.append(str(fname))
    return frames


def write_video(frames, out_path, fps=2):
    with imageio.get_writer(out_path, fps=fps) as writer:
        for f in frames:
            img = imageio.imread(f)
            writer.append_data(img)


def main():
    os.makedirs('demo', exist_ok=True)
    model_path = Path('models/model.joblib')
    if not model_path.exists():
        raise SystemExit('Model not found. Run training first: python -m src.train')

    model = load_model(str(model_path))
    df = generate_synthetic_data(n_samples=500)
    X = df[["traffic_load", "active_cells", "time_of_day", "ambient_temp", "qos_latency", "qos_throughput"]]
    preds = model.predict(X)
    actions = recommend_actions(df, model)

    frames = make_frames(df, preds, actions, Path('demo/frames'))
    out_video = Path('demo/demo.mp4')
    write_video(frames, out_video, fps=2)
    print(f'Wrote demo video to {out_video}')


if __name__ == '__main__':
    main()
