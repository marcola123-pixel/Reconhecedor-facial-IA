"""Cadastra uma pessoa na galeria SEM retreinar o modelo (spec 002).

    python scripts/enroll.py --run runs/e04_cnn_triplet --name "<nome>" --images pasta_com_fotos/
    python scripts/enroll.py --run runs/e04_cnn_triplet --remove "<nome>"      # direito de retirada

A galeria (embeddings/gallery.npz) é dado biométrico PRIVADO: está no .gitignore. Só o protótipo
(média dos embeddings) é guardado — nunca as fotos.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from facerec.detection import FaceDetector  # noqa: E402
from facerec.gallery import Gallery  # noqa: E402
from facerec.inference import embed_images, load_run  # noqa: E402
from facerec.splits import IMG_EXT  # noqa: E402

GALLERY_PATH = "embeddings/gallery.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name")
    ap.add_argument("--images")
    ap.add_argument("--remove")
    ap.add_argument("--gallery", default=GALLERY_PATH)
    args = ap.parse_args()

    g = Gallery.load(args.gallery)
    if args.remove:
        print("removido" if g.remove(args.remove) else "pessoa não encontrada")
        g.save(args.gallery)
        return
    if not (args.name and args.images):
        ap.error("informe --name e --images (ou --remove)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, tf = load_run(args.run, device)
    det = FaceDetector(image_size=meta["image_size"] if meta["image_size"] >= 160 else 160, device=device)
    faces = []
    for f in sorted(Path(args.images).iterdir()):
        if f.suffix.lower() in IMG_EXT:
            face = det.crop(Image.open(f))
            if face is not None:
                faces.append(face)
    if not faces:
        sys.exit("nenhum rosto detectado nas imagens informadas")
    g.enroll(args.name, embed_images(model, tf, faces, device))
    g.save(args.gallery)
    print(f"cadastrado com {len(faces)} foto(s). Pessoas na galeria: {len(g)}")


if __name__ == "__main__":
    main()
