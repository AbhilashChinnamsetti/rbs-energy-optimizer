"""Create a short GIF preview from an MP4 for README embedding."""
import argparse
from pathlib import Path
import imageio
from PIL import Image


def make_gif(input_path, output_path, duration=5, segment="middle", width=640):
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10)
    nframes = meta.get("nframes", None)
    frames_src = None
    import math
    if nframes is None or (isinstance(nframes, float) and not math.isfinite(nframes)):
        # try to use reported duration, else read frames
        duration_meta = meta.get("duration", None)
        if duration_meta:
            try:
                nframes = int(float(duration_meta) * float(fps))
            except Exception:
                frames_src = list(reader)
                nframes = len(frames_src)
        else:
            frames_src = list(reader)
            nframes = len(frames_src)
    # normalize fps
    try:
        fps = float(fps)
    except Exception:
        fps = 10.0
    import math
    if not math.isfinite(fps) or fps <= 0:
        fps = 10.0
    total_seconds = (nframes / fps) if (fps and nframes) else 0
    if segment == "start":
        start_sec = 0
    elif segment == "end":
        start_sec = max(0, total_seconds - duration)
    else:
        start_sec = max(0, (total_seconds - duration) / 2)

    start_frame = int(start_sec * fps)
    num_frames = int(duration * fps)

    frames = []
    for i in range(start_frame, min(start_frame + num_frames, nframes)):
        try:
            if frames_src is not None:
                frame = frames_src[i]
            else:
                frame = reader.get_data(i)
        except Exception:
            break
        img = Image.fromarray(frame)
        # resize preserving aspect
        w, h = img.size
        if width and w != width:
            new_h = int(h * (width / w))
            img = img.resize((width, new_h), Image.LANCZOS)
        frames.append(img)

    if not frames:
        raise SystemExit("No frames extracted from video")

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="demo/demo.mp4")
    parser.add_argument("--output", default="demo/preview.gif")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--segment", choices=["start", "middle", "end"], default="middle")
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input video not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_gif(str(inp), str(out), duration=args.duration, segment=args.segment, width=args.width)
    print(f"Wrote GIF: {out}")


if __name__ == "__main__":
    main()
