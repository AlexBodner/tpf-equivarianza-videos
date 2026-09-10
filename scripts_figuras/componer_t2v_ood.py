"""Arma los dos contrastes que faltaban: generacion por texto y fuera de dominio,
control (paso 125) contra el brazo con fisica (paso 875), con el modelo base
arriba como referencia de lo que trae SANA sin fine-tuning.

n = 2 clips por prompt: alcanza para mirar, no para medir.
"""
import subprocess, sys
from pathlib import Path
import cv2, numpy as np

BARRA, SEP, FPS = 30, 3, 8
FUENTE = cv2.FONT_HERSHEY_DUPLEX
FILAS = [("base", "base"), ("control 125", "control"), ("equiv. 875", "fisica")]


def cuadros(p):
    c, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fs.append(f)
    return fs


def componer(cols, salida, lado=280):
    pilas = {}
    for etiqueta, brazo in FILAS:
        vids = [cuadros(Path("videos_t2v_ood") / brazo / sub / f"{n}_00.mp4") for sub, n in cols]
        pilas[brazo] = [[cv2.resize(f, (lado, lado)) for f in v] for v in vids]
    T = min(len(v) for b in pilas.values() for v in b)
    ancho = len(cols) * lado + (len(cols) - 1) * SEP
    alto = len(FILAS) * (lado + BARRA) + (len(FILAS) - 1) * SEP
    tmp = Path("/tmp/crudo.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ancho, alto))
    for t in range(T):
        lienzo = np.zeros((alto, ancho, 3), np.uint8)
        y = 0
        for etiqueta, brazo in FILAS:
            for i, (_, nombre) in enumerate(cols):
                x = i * (lado + SEP)
                # El prompt sólo en la fila de arriba; el brazo, en todas.
                texto = f"{nombre}  ({etiqueta})" if y == 0 else etiqueta
                cv2.putText(lienzo, texto, (x + 5, y + BARRA - 9),
                            FUENTE, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
                lienzo[y + BARRA:y + BARRA + lado, x:x + lado] = pilas[brazo][i][t]
            y += lado + BARRA + SEP
        vw.write(lienzo)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p", str(salida)], check=True)
    print("escrito", salida, f"{T} cuadros")


componer([("t2v", "free_fall"), ("t2v", "pendulum"), ("t2v", "bouncing")],
         "21_t2v_control_vs_fisica.mp4")
componer([("ood", "rolling"), ("ood", "pendulum_photo")],
         "22_ood_control_vs_fisica.mp4")
