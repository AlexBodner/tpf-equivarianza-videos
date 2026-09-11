"""Las ocho semillas de cada prompt por texto, control arriba y física abajo.

Una comparación por semilla y no un mosaico: con ocho semillas en pantalla no se
ve ninguna. Cada archivo es el mismo prompt, la misma semilla y el mismo ruido en
los dos brazos, así que lo único que cambia entre las dos filas es el checkpoint.

Uso:  python3 scripts_figuras/componer_semillas.py <raiz_generaciones> <salida>
"""
import subprocess, sys
from pathlib import Path

import cv2
import numpy as np

BARRA, SEP, FPS, LADO = 26, 3, 8, 320
FUENTE = cv2.FONT_HERSHEY_DUPLEX
# Las etiquetas van sin acentos: la fuente de cv2 no las dibuja, las deja en "?".
BRAZOS = [("control_125", "control (paso 125)"), ("fisica_875", "con fisica (paso 875)")]

# nombre de salida -> (subcarpeta de la generación, prefijo del archivo)
PROMPTS = {
    "rodando":              ("ood", "rolling"),
    "pendulo_camara_lenta": ("ood", "pendulum_photo"),
    "pendulo":              ("t2v", "pendulum"),
    "rebote":               ("t2v", "bouncing"),
    "caida_libre":          ("t2v", "free_fall"),
}
SEMILLAS = range(8)


def cuadros(p):
    cap, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(cv2.resize(f, (LADO, LADO)))
    if not fs:
        raise SystemExit(f"video vacío o ilegible: {p}")
    return fs


def componer_grilla(grilla, salida, marca=None):
    """grilla: filas de [(etiqueta, path)], de arriba abajo y de izquierda a derecha.

    marca: (primer_cuadro, texto). Desde ese cuadro se dibuja un borde rojo y el
    texto abajo. Sirve para señalar dónde empieza la extrapolación, que es lo que
    un espectador no puede deducir mirando el video.
    """
    pilas = [[cuadros(p) for _, p in fila] for fila in grilla]
    T = min(len(v) for fila in pilas for v in fila)
    n_col = max(len(fila) for fila in grilla)
    ancho = n_col * LADO + (n_col - 1) * SEP
    alto = len(grilla) * (LADO + BARRA) + (len(grilla) - 1) * SEP
    tmp = Path("/tmp/crudo_semillas.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ancho, alto))
    for t in range(T):
        lienzo = np.zeros((alto, ancho, 3), np.uint8)
        y = 0
        for fila, pila in zip(grilla, pilas):
            for i, ((etiqueta, _), fs) in enumerate(zip(fila, pila)):
                x = i * (LADO + SEP)
                cv2.putText(lienzo, etiqueta, (x + 5, y + BARRA - 8),
                            FUENTE, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
                lienzo[y + BARRA:y + BARRA + LADO, x:x + LADO] = fs[t]
            y += LADO + BARRA + SEP
        if marca and t >= marca[0]:
            cv2.rectangle(lienzo, (1, 1), (ancho - 2, alto - 2), (60, 60, 235), 4)
            texto = f"{marca[1]} (cuadro {t + 1} de {T})"
            cv2.putText(lienzo, texto, (10, alto - 10),
                        FUENTE, 0.55, (60, 60, 235), 1, cv2.LINE_AA)
        vw.write(lienzo)
    vw.release()
    # Constrained Baseline: es lo que abre en QuickTime sin convertir nada.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p", str(salida)], check=True)
    return T


def componer(rutas, salida):
    """Una sola columna: cada entrada de `rutas` es una fila."""
    return componer_grilla([[r] for r in rutas], salida)


def main(raiz, salida):
    raiz, salida = Path(raiz), Path(salida)
    for nombre, (sub, pref) in PROMPTS.items():
        destino = salida / nombre
        destino.mkdir(parents=True, exist_ok=True)
        for s in SEMILLAS:
            rutas = [(et, raiz / brazo / sub / f"{pref}_semilla{s}.mp4")
                     for brazo, et in BRAZOS]
            faltan = [p for _, p in rutas if not p.exists()]
            if faltan:
                raise SystemExit(f"falta la generación: {faltan[0]}")
            T = componer(rutas, destino / f"semilla{s}.mp4")
            print(f"{nombre}/semilla{s}.mp4  {T} cuadros")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
