"""fig_t2v.png: la dinamica de la generacion por texto, en un solo cuadro.

Un poster es papel: un video no se puede mostrar, y un cuadro suelto no dice nada
de la dinamica, que es justamente lo que el trabajo discute. El estroboscopio
resuelve las dos cosas: el objeto se dibuja en varios instantes sobre el fondo,
con el color marcando el tiempo, asi que la trayectoria se lee de un vistazo.

Se usa texto a video y no la generacion condicionada porque sin condicionamiento
no hay dos cuadros reales sosteniendo el principio del clip: lo que se ve es lo
que el modelo produce solo.

El objeto se aisla por diferencia contra el fondo mediano del clip; si en algun
instante no hay nada sobre el fondo, ese instante simplemente no se dibuja, y la
cuenta de instantes dibujados va abajo de cada panel.

Uso:  python3 poster/scripts/gen_fig_t2v_estroboscopio.py <raiz_t2v> <salida.png>
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Los tres escenarios del dataset y los dos prompts que el modelo nunca vio: el
# contraste entre las dos mitades es parte de lo que la figura tiene que mostrar.
ESC = [("t2v", "free_fall", "caída libre"), ("t2v", "pendulum", "péndulo"),
       ("t2v", "bouncing", "rebote"),
       ("ood", "rolling", "rodando *")]
BRAZOS = [("control_125", "control"), ("fisica_875", "con física")]
SEMILLA = 0
N_INSTANTES = 24
UMBRAL = 12.0

# Del inicio del clip al final: frio a calido, que es la convencion para tiempo.
MAPA = LinearSegmentedColormap.from_list("tiempo", ["#2b7bba", "#8f6fb5", "#c0503d"])


def cuadros(p):
    c, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32))
    return np.array(fs)


def estroboscopio(fs):
    if len(fs) == 0:
        raise SystemExit("[fig] video vacío o ilegible: revisá la ruta")

    """Fondo del clip con el objeto pintado en N instantes, coloreado por tiempo."""
    fondo = np.median(fs, axis=0)
    lienzo = (fondo * 0.55 + 255 * 0.45) / 255.0     # el fondo se aclara para que
    lienzo = np.clip(lienzo, 0, 1)                   # el objeto resalte encima
    idx = np.linspace(0, len(fs) - 1, N_INSTANTES).astype(int)
    dibujados = 0
    for k, i in enumerate(idx):
        d = np.abs(fs[i] - fondo).mean(axis=2)
        mascara = d > UMBRAL
        if mascara.sum() < 4:
            continue
        dibujados += 1
        color = np.array(MAPA(k / (N_INSTANTES - 1))[:3])
        alfa = np.clip(d / max(d.max(), 1e-6), 0, 1)[..., None] * mascara[..., None]
        lienzo = lienzo * (1 - alfa) + color * alfa
    return lienzo, dibujados, len(fs)


def main(raiz, salida):
    raiz = Path(raiz)
    fig, axes = plt.subplots(len(BRAZOS), len(ESC), figsize=(13.0, 6.6))
    for r, (brazo, tit_b) in enumerate(BRAZOS):
        for c, (sub, esc, tit_e) in enumerate(ESC):
            ax = axes[r][c]
            # Sin linea de trayectoria a proposito: cuando el modelo parte el objeto
            # en dos no hay UNA trayectoria. El centroide de la mascara cae entre las
            # dos pelotas y la componente mas grande salta de una a la otra; las dos
            # dibujan un recorrido que no hizo nada. Los instantes no inventan nada.
            img, dib, T = estroboscopio(cuadros(raiz / brazo / sub / f"{esc}_semilla{SEMILLA}.mp4"))
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#C7CFD6")
            if r == 0:
                ax.set_title(tit_e, fontsize=15, pad=7)
            if c == 0:
                ax.set_ylabel(tit_b, fontsize=14.5)
            if dib < N_INSTANTES:
                ax.set_xlabel(f"{dib} de {N_INSTANTES} instantes con objeto",
                              fontsize=11, color="#B4622D")
    # Sin titulo: en un poster la figura ya viene con caption, y repetirlo ahi
    # gasta centimetros de columna que no sobran.
    print(f"[fig] {T} cuadros por clip, {N_INSTANTES} instantes dibujados")
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    print("escrita", salida)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
