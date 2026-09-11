"""Mide si el pendulo generado oscila a la velocidad que su largo permite.

POR QUE SE PUEDE MEDIR SIN GROUND TRUTH. En generacion por texto no hay clip real
apareado, asi que no hay contra que comparar cuadro a cuadro. Pero un pendulo tiene
una restriccion interna: T = 2*pi*sqrt(L/g). El largo y el periodo se leen los dos
del video generado, asi que la consistencia entre ellos se chequea sola.

EL CONTROL QUE HACE FALTA. "Oscila mas rapido" no prueba fisica mala: un pendulo
corto oscila rapido con fisica perfecta. Hay que comparar el periodo observado
contra el que corresponde AL LARGO GENERADO, escalando el del simulador por
sqrt(L/L_sim). Sin ese control, el brazo con fisica parece mejor (3,5 contra 4,0
reversiones) cuando lo unico que pasa es que genera pendulos mas largos.

El contador de reversiones se valida antes contra senos de periodo conocido, y el
largo de referencia sale del ground truth del simulador.

Uso:  python3 scripts_figuras/medir_pendulo.py <raiz_semillas> <raiz_ground_truth>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PERIODO_SIM = 37.4      # cuadros, periodo del pendulo del simulador
T_CLIP = 33             # cuadros por clip generado
UMBRAL = 12.0
AREA_MIN = 80
SALTO_MIN = 3.0         # px: por debajo de esto un cambio de signo es jitter


def _serie(ruta):
    """Altura del objeto mas grande y largo pivote-pelota, cuadro a cuadro."""
    cap, fs = cv2.VideoCapture(str(ruta)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f.astype(np.float32))
    if not fs:
        raise SystemExit(f"[pendulo] video vacío o ilegible: {ruta}")
    arr = np.array(fs)
    fondo = np.median(arr, axis=0)
    ys, largos = [], []
    for f in arr:
        d = np.abs(f - fondo).mean(axis=2)
        m = d > UMBRAL
        if m.sum() < 20:
            ys.append(np.nan)
            continue
        # El pivote es el extremo superior de la estructura hilo+pelota.
        filas = np.flatnonzero(m.any(axis=1))
        top = filas[0]
        piv = (np.flatnonzero(m[top]).mean(), top)
        n, _, stats, centros = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        grandes = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] > AREA_MIN]
        if not grandes:
            ys.append(np.nan)
            continue
        bola = centros[max(grandes, key=lambda j: stats[j, cv2.CC_STAT_AREA])]
        ys.append(bola[1])
        largos.append(float(np.hypot(bola[0] - piv[0], bola[1] - piv[1])))
    return np.array(ys), (float(np.median(largos)) if largos else np.nan)


def reversiones(y):
    """Cambios de direccion de la altura, ignorando el jitter del centroide."""
    ok = ~np.isnan(y)
    if ok.sum() < 6:
        return np.nan
    y = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    y = np.convolve(y, np.ones(3) / 3, mode="valid")
    n, signo_prev, extremo = 0, 0, y[0]
    for i, vi in enumerate(np.diff(y)):
        s = np.sign(vi)
        if s == 0:
            continue
        if signo_prev != 0 and s != signo_prev and abs(y[i + 1] - extremo) >= SALTO_MIN:
            n += 1
            extremo = y[i + 1]
        elif signo_prev == 0:
            extremo = y[i + 1]
        signo_prev = s
    return n


def _validar():
    """Sin esto no se sabe si el contador mide reversiones o ruido."""
    t = np.arange(T_CLIP)
    print("validación del contador, senos de período conocido:")
    for periodo in (37.4, 18.0, 11.0):
        esperado = T_CLIP / (periodo / 2)
        obtenido = reversiones(80 * np.sin(2 * np.pi * t / periodo))
        print(f"  período {periodo:5.1f} -> esperadas {esperado:.1f}, contadas {obtenido}")
        if abs(obtenido - esperado) > 1.0:
            raise SystemExit("[pendulo] el contador no reproduce un período conocido")


def main(raiz, raiz_gt):
    _validar()
    gt = [_serie(Path(raiz_gt) / f"{c}.mp4")[1] for c in ("00500", "00501")]
    l_sim = float(np.nanmean(gt))
    print(f"\nlargo de referencia del simulador: {l_sim:.0f} px "
          f"(período {PERIODO_SIM} cuadros)")

    print(f"\n{'brazo':12s} {'largo':>7s} {'rev. esperadas':>15s} {'rev. observadas':>16s} {'razón':>7s}")
    for carpeta, etiqueta in [("control_125", "control"), ("fisica_875", "con física")]:
        largos, revs = [], []
        for s in range(8):
            y, L = _serie(Path(raiz) / carpeta / "t2v" / f"pendulum_semilla{s}.mp4")
            largos.append(L)
            revs.append(reversiones(y))
        L = float(np.nanmedian(largos))
        obs = float(np.nanmedian(revs))
        esp = T_CLIP / ((PERIODO_SIM * np.sqrt(L / l_sim)) / 2)
        print(f"{etiqueta:12s} {L:6.0f}px {esp:15.1f} {obs:16.1f} {obs/esp:6.2f}x")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
