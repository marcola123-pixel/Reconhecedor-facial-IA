"""Parte 2 — aplicação web do porteiro eletrônico (HTTPS, decisão no servidor).

Usa o experimento escolhido na Parte 1 (runs/<exp>/best.pt + meta.json) e a galeria de pessoas
cadastradas (embeddings/gallery.npz). Nenhuma foto é gravada em disco.

    python scripts/gerar_certificado.py        # uma vez (cria certs/cert.pem e certs/key.pem)
    export FACEREC_RUN=runs/e04_cnn_triplet    # experimento vencedor da Parte 1
    export PORTEIRO_TOKEN="<um segredo seu>"   # habilita cadastro/remoção (nunca coloque no código/Git)
    python app/main.py                         # abre https://localhost:8443
"""
import base64
import binascii
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from facerec.detection import FaceDetector  # noqa: E402
from facerec.gallery import Gallery  # noqa: E402
from facerec.inference import embed_images, load_run  # noqa: E402

HOST, PORT = "127.0.0.1", 8443
RUN_DIR = os.environ.get("FACEREC_RUN", "runs/e04_cnn_triplet")
GALLERY_PATH = os.environ.get("FACEREC_GALLERY", "embeddings/gallery.npz")
TOKEN = os.environ.get("PORTEIRO_TOKEN", "")
MAX_IMAGE_CHARS = 4_000_000          # ~3 MB em base64

device = "cuda" if torch.cuda.is_available() else "cpu"
model, meta, transform = load_run(RUN_DIR, device)
THRESHOLD = float(os.environ.get("FACEREC_THRESHOLD", meta["id_threshold"]))   # vem da VALIDAÇÃO (ADR-003)
detector = FaceDetector(image_size=160, device=device)
gallery = Gallery.load(GALLERY_PATH)

app = FastAPI(title="Porteiro Eletrônico")
app.add_middleware(CORSMiddleware, allow_origins=[f"https://127.0.0.1:{PORT}", f"https://localhost:{PORT}"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Porteiro-Token"])


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        r = await call_next(request)
        r.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        r.headers["X-Content-Type-Options"] = "nosniff"
        r.headers["X-Frame-Options"] = "DENY"
        r.headers["Referrer-Policy"] = "no-referrer"
        r.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        r.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        return r


app.add_middleware(SecurityHeaders)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


def require_admin(x_porteiro_token: str = Header(default="")):
    """Cadastro/remoção são administrativos: sem PORTEIRO_TOKEN configurado ficam desativados."""
    if not TOKEN:
        raise HTTPException(403, "defina PORTEIRO_TOKEN no servidor para habilitar a administração")
    if x_porteiro_token != TOKEN:
        raise HTTPException(401, "token inválido")


def decode_image(data_url: str) -> Image.Image:
    if len(data_url) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "imagem grande demais")
    try:
        return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
    except (IndexError, binascii.Error, OSError):
        raise HTTPException(400, "imagem inválida")


class FrameIn(BaseModel):
    image: str


class EnrollIn(BaseModel):
    nome: str
    imagens: list[str]


@app.get("/")
def home():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/status")
def status():
    return {"experimento": meta["name"], "limiar": THRESHOLD, "pessoas_cadastradas": len(gallery),
            "device": device}


@app.post("/reconhecer")
def reconhecer(frame: FrameIn):
    face = detector.crop(decode_image(frame.image))
    if face is None:
        return {"rosto_detectado": False}
    emb = embed_images(model, transform, [face], device)[0]
    m = gallery.identify(emb, THRESHOLD)
    liberado = m.name is not None                       # decisão tomada AQUI, no servidor
    return {"rosto_detectado": True, "liberado": liberado,
            "nome": m.name, "similaridade": round(m.score, 4)}


@app.post("/pessoas", dependencies=[Depends(require_admin)])
def cadastrar(body: EnrollIn):
    faces = [f for f in (detector.crop(decode_image(i)) for i in body.imagens) if f is not None]
    if not faces:
        raise HTTPException(422, "nenhum rosto detectado nas imagens")
    gallery.enroll(body.nome.strip(), embed_images(model, transform, faces, device))
    gallery.save(GALLERY_PATH)
    return {"cadastradas": len(faces), "pessoas": len(gallery)}


@app.delete("/pessoas/{nome}", dependencies=[Depends(require_admin)])
def remover(nome: str):
    if not gallery.remove(nome):
        raise HTTPException(404, "pessoa não encontrada")
    gallery.save(GALLERY_PATH)
    return {"removida": True, "pessoas": len(gallery)}


if __name__ == "__main__":
    import uvicorn
    cert, key = ROOT / "certs" / "cert.pem", ROOT / "certs" / "key.pem"
    if not (cert.exists() and key.exists()):
        sys.exit("Certificado TLS não encontrado. Rode: python scripts/gerar_certificado.py")
    print(f"https://{HOST}:{PORT}  | experimento={meta['name']} | limiar={THRESHOLD:.3f}")
    uvicorn.run(app, host=HOST, port=PORT, ssl_keyfile=str(key), ssl_certfile=str(cert))
