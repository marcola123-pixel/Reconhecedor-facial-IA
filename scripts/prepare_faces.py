"""Detecta e recorta rostos (MTCNN) de <src>/<identidade>/*.jpg e salva em <dst>/<identidade>/*.png.

    python scripts/prepare_faces.py --src data/raw/lfw_funneled --dst data/processed/lfw
    python scripts/prepare_faces.py --src data/private/raw --dst data/private/processed   # dados privados
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from facerec.detection import FaceDetector  # noqa: E402
from facerec.splits import IMG_EXT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--min-images", type=int, default=1, help="ignora identidades com menos fotos")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    det = FaceDetector(image_size=args.size, device=args.device)
    ok = fail = skipped = 0
    for d in sorted(p for p in Path(args.src).iterdir() if p.is_dir()):
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
        if len(files) < args.min_images:
            skipped += 1
            continue
        out = Path(args.dst) / d.name
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            try:
                face = det.crop(Image.open(f))
            except Exception:
                face = None
            if face is None:
                fail += 1
                continue
            face.save(out / f"{f.stem}.png")
            ok += 1
    print(f"rostos salvos: {ok} | sem rosto/ilegíveis: {fail} | identidades ignoradas: {skipped}")


if __name__ == "__main__":
    main()
