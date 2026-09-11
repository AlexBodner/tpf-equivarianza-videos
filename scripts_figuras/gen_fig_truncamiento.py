"""figuras/truncamiento_direccion.png: las dos mitades de la pregunta sobre truncar el BPTT.

(a) Cuanta direccion del gradiente conserva cada ventana. Sale del aporte exacto
    de cada paso de Euler, registrado en la corrida que se reporta: 12 normas y
    los 66 cosenos entre pares por paso determinan la matriz de Gram, y de ahi el
    coseno de cualquier ventana contra la suma completa, sin recalcular nada.
(b) Que pasa si efectivamente se entrena con esas direcciones. Es la prueba
    empirica de (a): cinco brazos de 150 pasos, identicos salvo la ventana.

Van juntas a proposito. (a) sola diria que truncar conserva la direccion sin
mostrar que eso no se traduce en un resultado distinguible; (b) sola diria que
no hay diferencia sin explicar por que era esperable. Los colores atan cada
ventana entre los dos paneles.

Sin argumentos usa el log de la corrida que se reporta y el barrido de
resultados/. Uso:  python3 scripts_figuras/gen_fig_truncamiento.py [log.jsonl] [dir_barrido] [salida.png]
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coseno_truncamiento_bptt import carga_gram, coseno, gram  # noqa: E402

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15.5, "axes.labelsize": 14.5,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5})

# (nombre corto, pasos retropropagados, archivo del barrido, color)
BRAZOS = [
    ("BPTT completo",      list(range(12)),     "full",       "#2e7d32"),
    ("no contigua",        [2, 3, 8, 11],       "ventana4",   "#2b7bba"),
    ("mejor-6",            [0, 1, 2, 4, 5, 6],  "mejor6",     "#c98f2b"),
    ("cola de 4 (DRaFT-K)", [8, 9, 10, 11],     "cola4",      "#c0503d"),
]
COSTO = {4: 21.6, 6: 23.4, 12: 27.8}   # s/paso medidos en el barrido


def cosenos(log):
    acc = {n: [] for n, _, _, _ in BRAZOS}
    for linea in open(log):
        e = json.loads(linea).get("bptt_grad_norms")
        if not e or len(e) != 12:
            continue
        G = gram(e)
        if G.sum() <= 0:
            continue
        for nombre, sub, _, _ in BRAZOS:
            c = coseno(G, sub)
            if c is not None:
                acc[nombre].append(c)
    return acc


def perdida(path):
    xs, ys = [], []
    for linea in open(path):
        d = json.loads(linea)
        if d.get("n_bruto") is None:
            continue
        n, dd, piso = d["n_bruto"], d["d_bruto"], d["piso"]
        xs.append(d["step"])
        ys.append(max(n - piso, 0.0) / max(dd - piso, piso if piso > 0 else 1e-9))
    return xs, ys


def por_ventanas(xs, v, k=15):
    px, py = [], []
    for i in range(0, len(v), k):
        tramo = v[i:i + k]
        if tramo:
            px.append(st.mean(xs[i:i + k]))
            py.append(st.median(tramo))
    return px, py


def main(log, dir_barrido, salida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.2))

    # Barras y no curva: cada punto de una curva por tamano seria una ventana
    # DISTINTA, y eso no se puede adivinar mirando. Con barras cada una tiene
    # nombre, y las cuatro que ademas se entrenaron llevan el color del panel (b).
    G = carga_gram(log)[len(carga_gram(log)) // 2:]   # segunda mitad, para todas igual
    VENT = [("mejor de 2 [1,10]", [1, 10], "#6a3d9a"),
            ("BPTT completo", list(range(12)), "#2e7d32"),
            ("no contigua [2,3,8,11]", [2, 3, 8, 11], "#2b7bba"),
            ("mejor-6 [0,1,2,4,5,6]", [0, 1, 2, 4, 5, 6], "#c98f2b"),
            ("cola de 4 (DRaFT-K)", [8, 9, 10, 11], "#c0503d")]
    filas = [(n, float(np.mean([coseno(g, S) for g in G])), len(S), c) for n, S, c in VENT]
    filas.sort(key=lambda f: f[1])
    y = np.arange(len(filas))
    ax1.barh(y, [f[1] for f in filas], color=[f[3] for f in filas], height=0.6)
    for i, f in enumerate(filas):
        ax1.text(f[1] - 0.008, i, f"{f[1]:.2f}", va="center", ha="right",
                 color="white", fontsize=14)
    ax1.set_yticks(y, [f"{f[0]}\n{f[2]} paso{'s' if f[2] > 1 else ''}" for f in filas])
    ax1.set_xlim(0.7, 1.02)
    ax1.axvline(1.0, color="#555", ls="--", lw=1.3)
    ax1.set_xlabel("dirección explicada: coseno con el gradiente exacto")
    ax1.set_title("(a) cuánta dirección retiene cada ventana")

    for nombre, sub, archivo, color in BRAZOS:
        xs, ys = perdida(Path(dir_barrido) / f"{archivo}.jsonl")
        ax2.plot(xs, ys, "-", color=color, lw=0.8, alpha=0.2, zorder=1)
        px, py = por_ventanas(xs, ys)
        ax2.plot(px, py, "o-", color=color, lw=2.4, ms=6, zorder=3, label=nombre)
    ax2.set_ylim(0, 0.62)
    ax2.set_xlabel("paso de entrenamiento")
    ax2.set_ylabel("pérdida física")
    ax2.set_title("(b) pero entrenar con ella no cambia el resultado")
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    ax2.text(0.015, 0.02, "cruda recortada; el 5 % llega a 1,53",
             transform=ax2.transAxes, fontsize=11.5, color="#777")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    print("escrita", salida)


if __name__ == "__main__":
    import glob
    por_omision = [
        glob.glob("resultados/logs_entrenamiento/vel_run__*.jsonl")[0],
        "resultados/logs_barrido",
        "figuras/truncamiento_direccion.png",
    ]
    args = sys.argv[1:4] if len(sys.argv) == 4 else por_omision
    main(*args)
