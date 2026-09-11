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
FINAL = 11          # ultimos frames de 33: el tramo donde se juzga
# Rueda AL FINAL: esta abajo, con altura estable y avanzando en horizontal.
# El criterio mira el tramo final y no el clip entero porque la pelota suele caer
# primero y rodar despues; pidiendo altura baja en TODO el clip, esos no contaban.
ALTURA_MAX = 0.22
DESVIO_MAX = 0.05
AVANCE_MIN = 0.08


def pista(ruta):
    """Sobre el tramo final: altura media (0 = piso), su desvio y el avance en x.

    Se sigue el objeto mas BAJO de cada frame y no el mas grande: cuando el modelo
    parte la escena en dos, el mas grande salta entre objetos. Ojo que ninguno de
    los dos criterios distingue "apoyado en el piso" de "flotando bajo": eso hay
    que mirarlo, y por eso el conteo se contrasta con los videos.
    """
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
    for f in arr[-FINAL:]:
        d = np.abs(f - fondo).mean(axis=2)
        m = (d > UMBRAL).astype(np.uint8)
        n, _, stats, centros = cv2.connectedComponentsWithStats(m, 8)
        grandes = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] > AREA_MIN]
        if not grandes:
            continue
        j = max(grandes, key=lambda k: centros[k][1])
        xs.append(centros[j][0] / ancho)
        ys.append(1 - centros[j][1] / alto)
    if len(ys) < FINAL - 2:
        raise SystemExit(f"[rodando] objeto no detectado en el tramo final de {ruta}")
    xs, ys = np.array(xs), np.array(ys)
    return (ys.mean(), ys.std(), abs(xs[-1] - xs[0]))


def main(raiz):
    raiz = Path(raiz)
    alturas = {}
    for carpeta, etiqueta in [("control_125", "control"), ("fisica_875", "con física")]:
        filas = [pista(raiz / carpeta / "ood" / f"rolling_semilla{s}.mp4")
                 for s in range(N_SEMILLAS)]
        alturas[etiqueta] = np.array([f[0] for f in filas])
        ruedan = [s for s, (a, d, av) in enumerate(filas)
                  if a < ALTURA_MAX and d < DESVIO_MAX and av > AVANCE_MIN]
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
