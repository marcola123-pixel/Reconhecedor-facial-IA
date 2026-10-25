"""Detecção e recorte de rosto com MTCNN (spec 001 / ADR-001).

Pré-processamento = detecção + recorte com margem + redimensionamento. (Alinhamento por landmarks
é uma melhoria futura; ver ADR-001.) Com post_process=False o MTCNN devolve pixels em [0, 255].
"""
from __future__ import annotations

from typing import Optional

from PIL import Image


class FaceDetector:
    def __init__(self, image_size: int = 160, margin: int = 20, device: Optional[str] = None):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, post_process=False,
                           select_largest=True, device=device)

    def crop(self, img: Image.Image) -> Optional[Image.Image]:
        """Retorna o maior rosto recortado (PIL RGB) ou None se nenhum rosto for detectado."""
        face = self.mtcnn(img.convert("RGB"))
        if face is None:
            return None
        arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(arr)
