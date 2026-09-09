"""Rehace videos/06_dos_ramas_de_la_perdida.mp4 desde los cuadros de un checkpoint.

Los cuadros los deja scripts/equiv_por_checkpoint.py (results/equiv_por_checkpoint/videos/).
Uso:  python3 scripts_figuras/componer_dos_ramas.py <orig.npy> <rot.npy> <salida.mp4>
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np, cv2

BARRA = 38
SEP = 4
FPS = 8
BLANCO = (255, 255, 255)
VERDE = (120, 255, 160)

def componer(orig, rot, salida: Path):
    assert orig.shape == rot.shape, (orig.shape, rot.shape)
    T, H, W, _ = orig.shape
    alto, ancho = H + BARRA, W * 2 + SEP
    tmp = Path(tempfile.mkdtemp()) / "crudo.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ancho, alto))
    for t in range(T):
        lienzo = np.zeros((alto, ancho, 3), np.uint8)
        lienzo[BARRA:, :W] = cv2.cvtColor(orig[t], cv2.COLOR_RGB2BGR)
        lienzo[BARRA:, W + SEP:] = cv2.cvtColor(rot[t], cv2.COLOR_RGB2BGR)
        cv2.putText(lienzo, "rama 1: escena original", (8, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, BLANCO, 1, cv2.LINE_AA)
        cv2.putText(lienzo, "rama 2: escena rotada 45 grados", (W + SEP + 8, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, VERDE, 1, cv2.LINE_AA)
        vw.write(lienzo)
    vw.release()
    # Constrained Baseline: sin esto no reproduce en macOS ni en la vista previa de GitHub.
    salida.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], check=True)
    return alto, ancho, T

if __name__ == "__main__":
    o, r, sal = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    alto, ancho, T = componer(np.load(o), np.load(r), sal)
    print(f"{sal}: {ancho}x{alto}, {T} cuadros")
