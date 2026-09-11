"""Generacion mas larga que el horizonte de entrenamiento, en los checkpoints
elegidos: control (paso 125) contra el brazo con fisica (paso 875), 65 cuadros
contra los 33 que el modelo vio.

La version anterior de esta comparacion estaba hecha en el paso 1000, que no es
el checkpoint que reporta el trabajo, y daba la conclusion al reves. Desde el
cuadro 33 el borde se pone rojo: todo lo que sigue es extrapolacion.

n = 2 clips por escenario: alcanza para mirar, no para medir.

Uso:  python3 scripts_figuras/componer_i2v_65.py <raiz_videos_i2v_65> <salida.mp4>
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

BARRA, SEP, FPS = 30, 3, 8
HORIZONTE = 33
FUENTE = cv2.FONT_HERSHEY_DUPLEX
FILAS = [("control 125", "control"), ("equiv. 875", "fisica")]
COLS = [("free_fall", "caida libre"), ("pendulum", "pendulo"), ("bouncing", "rebote")]
CLIP = "00500.mp4"


def cuadros(p):
    c, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fs.append(f)
    return fs


def componer(raiz, salida, lado=280):
    pilas = {b: [[cv2.resize(f, (lado, lado)) for f in cuadros(raiz / b / esc / CLIP)]
                 for esc, _ in COLS]
             for _, b in FILAS}
    T = min(len(v) for b in pilas.values() for v in b)
    ancho = len(COLS) * lado + (len(COLS) - 1) * SEP
    alto = len(FILAS) * (lado + BARRA) + (len(FILAS) - 1) * SEP
    tmp = Path("/tmp/crudo_i2v65.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ancho, alto))
    for t in range(T):
        lienzo = np.zeros((alto, ancho, 3), np.uint8)
        y = 0
        for etiqueta, brazo in FILAS:
            for i, (_, nombre) in enumerate(COLS):
                x = i * (lado + SEP)
                texto = f"{nombre}  ({etiqueta})" if y == 0 else etiqueta
                cv2.putText(lienzo, texto, (x + 5, y + BARRA - 9),
                            FUENTE, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
                lienzo[y + BARRA:y + BARRA + lado, x:x + lado] = pilas[brazo][i][t]
            y += lado + BARRA + SEP
        # Pasado el horizonte de entrenamiento el borde se pone rojo: de ahi en
        # adelante nada de lo que se ve estaba en la distribucion de train.
        if t >= HORIZONTE:
            cv2.rectangle(lienzo, (0, 0), (ancho - 1, alto - 1), (0, 0, 235), 6)
            cv2.putText(lienzo, f"extrapolacion  (cuadro {t + 1} de {T})",
                        (10, alto - 12), FUENTE, 0.55, (0, 0, 235), 1, cv2.LINE_AA)
        vw.write(lienzo)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p", str(salida)], check=True)
    print("escrito", salida, f"{T} cuadros")


if __name__ == "__main__":
    componer(Path(sys.argv[1]), sys.argv[2])
