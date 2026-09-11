"""curvas_entrenamiento.png: las pérdidas de los dos brazos y cómo se eligió cada checkpoint.

Tres advertencias sobre las escalas, que es lo que hace fácil dibujar esto mal:

* `loss_rotation` del log de entrenamiento es el término YA multiplicado por lambda
  (mediana 0,00086), no el cociente que la pérdida minimiza (mediana 0,078). Acá se
  reconstruye el cociente desde n_bruto, d_bruto y piso.
* `val_loss_rotation` del log vive en otra escala todavía (4 a 16): es el MSE crudo.
  Mezclar los tres en un eje da una curva plana en cero, que fue el primer intento.
* Las validaciones del log van cada 50 pasos, así que ni el 125 ni el 875 están ahí.
  Los checkpoints se guardaron cada 125 y se validaron después, en una pasada aparte;
  ésos son los archivos de resultados/validaciones.

Uso:  python3 gen_fig_curvas_entrenamiento.py <log_fisica> <log_control> <dir_validaciones> <salida.png>
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13.5, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5})

FISICA, CONTROL, ELEGIDO = "#2b7bba", "#777777", "#c0503d"


def entrenamiento(path):
    return [json.loads(l) for l in open(path) if json.loads(l).get("type") != "val"]


def por_ventanas(filas, clave, k=25):
    px, py = [], []
    for i in range(0, len(filas), k):
        tramo = [f[clave] for f in filas[i:i + k] if f.get(clave) is not None]
        if tramo:
            px.append(st.mean([f["step"] for f in filas[i:i + k]]))
            py.append(st.median(tramo))
    return px, py


def cociente(filas, k=25):
    """El cociente que la pérdida minimiza, reconstruido del log."""
    vals = [{"step": f["step"],
             "q": max(f["n_bruto"] - f["piso"], 0.0) / max(f["d_bruto"] - f["piso"], f["piso"])}
            for f in filas if f.get("n_bruto") is not None and f.get("piso", 0) > 0]
    return por_ventanas(vals, "q", k)


def main(log_fis, log_ctrl, dir_val, salida):
    tr_f, tr_c = entrenamiento(log_fis), entrenamiento(log_ctrl)
    ctrl = json.load(open(Path(dir_val) / "control_8_checkpoints.json"))
    fis = json.load(open(Path(dir_val) / "fisica_8_checkpoints.json"))
    por, lam = fis["por_paso"], fis["lambda_rot"]
    pasos = sorted(por, key=int)

    fig, (a1, ab, a2, a3) = plt.subplots(1, 4, figsize=(21.5, 4.4))

    for tr, color, nombre in ((tr_c, CONTROL, "control"), (tr_f, FISICA, "con física")):
        a1.plot(*por_ventanas(tr, "loss_diffusion", k=50), "-", color=color, lw=2.4, label=nombre)
    a1.set_title("(a) difusión, en entrenamiento")
    a1.set_ylabel("pérdida de difusión")
    a1.legend(frameon=False)

    # El termino fisico aislado: es el cociente que la perdida minimiza, no el
    # loss_rotation logueado, que ya viene multiplicado por lambda.
    px, py = cociente(tr_f, k=50)
    ab.plot(px, py, "-", color=FISICA, lw=2.4)
    ab.set_title("(b) el término físico, aislado")
    ab.set_ylabel("cociente que la pérdida minimiza")
    ab.set_ylim(bottom=0)
    ab.axvline(875, color=ELEGIDO, ls=":", lw=1.8)

    for datos, color, nombre, elegido in ((ctrl, CONTROL, "control", "125"),
                                          (por, FISICA, "con física", "875")):
        ys = [datos[k]["val_loss_diffusion"] for k in pasos]
        a2.plot([int(k) for k in pasos], ys, "o-", color=color, lw=2.2, ms=7, label=nombre)
        a2.plot(int(elegido), datos[elegido]["val_loss_diffusion"], "o", ms=15,
                mfc="none", mec=ELEGIDO, mew=2.6, zorder=5)
    a2.set_title("(c) difusión, en validación")
    a2.set_ylabel("pérdida de difusión")
    a2.legend(frameon=False)

    suma = [por[k]["val_loss_diffusion"] + lam * por[k]["val_loss_rotation"] for k in pasos]
    a3.plot([int(k) for k in pasos], suma, "o-", color=FISICA, lw=2.4, ms=7)
    a3.plot(875, por["875"]["val_loss_diffusion"] + lam * por["875"]["val_loss_rotation"],
            "o", ms=15, mfc="none", mec=ELEGIDO, mew=2.6, zorder=5)
    a3.set_title("(d) el objetivo que eligió el checkpoint")
    a3.set_ylabel("difusión $+\\ \\lambda\\cdot$rotación, en validación")
    a3.annotate("el 375 queda a 0,2 %\nde distancia", xy=(375, suma[pasos.index("375")]),
                xytext=(300, min(suma) + (max(suma) - min(suma)) * 0.33), fontsize=12, color="#555",
                ha="center", arrowprops=dict(arrowstyle="->", color="#555", lw=1.3,
                connectionstyle="arc3,rad=-0.3"))

    for ax in (a1, ab, a2, a3):
        ax.set_xlabel("paso de entrenamiento")
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "En rojo, el checkpoint que eligió cada brazo por su propia validación sobre los mismos "
             "ocho candidatos. El control no aparece en (b) ni en (d) porque con $\\lambda=0$ no calcula ese término.",
             ha="center", fontsize=12.5, color="#666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(salida, dpi=150)
    print("escrita", salida)


if __name__ == "__main__":
    main(*sys.argv[1:5])
