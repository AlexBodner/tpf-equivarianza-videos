"""Mide cuantas generaciones de "rodando" ruedan de verdad, sin ground truth.

"Rodando" es un prompt que el modelo nunca vio y no tiene clip real apareado, asi
que no hay con que comparar cuadro a cuadro. Pero rodar sobre el piso tiene una
firma geometrica que se chequea sola: altura BAJA, altura CONSTANTE y
desplazamiento HORIZONTAL grande. Las tres se miden sobre el video generado.

POR QUE ESTA EL CONTEO Y NO SOLO EL MEJOR CLIP. Mostrar el clip donde mejor rueda
es legitimo en una figura, pero sin la frecuencia el lector no sabe si es lo tipico
o la mejor de ocho. Este script da las dos cosas: cuantas semillas cumplen el
criterio, y la comparacion apareada de altura media sobre las ocho.

Uso:  python3 scripts_figuras/medir_rodando.py <raiz_semillas>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

UMBRAL = 12.0
AREA_MIN = 80
N_SEMILLAS = 8
# Rueda = esta abajo, no rebota y avanza. Los tres a la vez.
ALTURA_MAX = 0.25
DESVIO_MAX = 0.06
RECORRIDO_MIN = 0.25


def pista(ruta):
    """Altura media (0 = piso, 1 = techo), su desvio y el recorrido horizontal."""
    cap, fs = cv2.VideoCapture(str(ruta)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f.astype(np.float32))
    if len(fs) < 5:
        raise SystemExit(f"[rodando] video corto o ilegible: {ruta}")
    arr = np.array(fs)
    fondo = np.median(arr, axis=0)
    alto, ancho = fondo.shape[:2]
    xs, ys = [], []
    for f in arr:
        d = np.abs(f - fondo).mean(axis=2)
        m = (d > UMBRAL).astype(np.uint8)
        n, _, stats, centros = cv2.connectedComponentsWithStats(m, 8)
        grandes = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] > AREA_MIN]
        if not grandes:
            continue
        j = max(grandes, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        xs.append(centros[j][0])
        ys.append(centros[j][1])
    if len(ys) < 5:
        raise SystemExit(f"[rodando] objeto no detectado en {ruta}")
    xs, ys = np.array(xs), np.array(ys)
    return (1 - ys.mean() / alto, ys.std() / alto, (xs.max() - xs.min()) / ancho)


def main(raiz):
    raiz = Path(raiz)
    alturas = {}
    for carpeta, etiqueta in [("control_125", "control"), ("fisica_875", "con física")]:
        filas = [pista(raiz / carpeta / "ood" / f"rolling_semilla{s}.mp4")
                 for s in range(N_SEMILLAS)]
        alturas[etiqueta] = np.array([f[0] for f in filas])
        ruedan = [s for s, (a, d, r) in enumerate(filas)
                  if a < ALTURA_MAX and d < DESVIO_MAX and r > RECORRIDO_MIN]
        print(f"{etiqueta:11s} ruedan {len(ruedan)}/{N_SEMILLAS}"
              + (f"  (semillas {ruedan})" if ruedan else "")
              + f"   altura media mediana {np.median(alturas[etiqueta]):.2f}")

    d = alturas["control"] - alturas["con física"]
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\naltura media, apareado por semilla, control - física = {d.mean():+.3f}"
          f"  IC95 [{lo:+.3f}, {hi:+.3f}]"
          + ("  (cruza cero)" if lo < 0 < hi else "  *  la física la mantiene más baja"))


if __name__ == "__main__":
    main(sys.argv[1])
