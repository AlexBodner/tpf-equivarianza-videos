"""figuras/seleccion_checkpoint.png — por que la regla de la suma elige el 750.

Panel izquierdo: los dos terminos de validacion y su suma (el objetivo de
entrenamiento). Panel derecho: cuanto se mueve el video generado, medido en la
evaluacion. El minimo de la suma cae donde el movimiento es minimo.
"""
import re, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM = 0.0047
val_dif = {250: 0.0698, 500: 0.0808, 750: 0.0846, 1000: 0.0724}
val_rot = {250: 11.2507, 500: 10.3254, 750: 6.7278, 1000: 12.8185}
suma = {p: val_dif[p] + LAM * val_rot[p] for p in val_dif}

mov = {}
for f in glob.glob("resultados/evaluaciones/e4vel__explor_paso_*__analisis.txt"):
    paso = int(re.search(r"paso_(\d+)", f).group(1)); v = []
    for l in open(f):
        m = re.match(r"\s*(free_fall|pendulum|bouncing)\s+\d+\s+[\d.]+\s+[\d.-]+\s+([\d.]+)", l)
        if m: v.append(float(m.group(2)))
    if v: mov[paso] = st.mean(v)

AZUL, ROJO, GRIS = "#2b7bba", "#c0503d", "#6b6b6b"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

ps = sorted(val_dif)
ax1.plot(ps, [val_dif[p] for p in ps], "o-", color=GRIS, lw=2, ms=7, label="validación de difusión")
ax1.plot(ps, [LAM * val_rot[p] for p in ps], "o-", color=ROJO, lw=2, ms=7, label="λ · validación de rotación")
ax1.plot(ps, [suma[p] for p in ps], "o-", color=AZUL, lw=2.6, ms=8, label="suma (el objetivo)")
mejor = min(suma, key=suma.get)
ax1.axvline(mejor, color=AZUL, ls=":", lw=1.5)
ax1.annotate(f"mínimo de la suma: paso {mejor}", (mejor, suma[mejor]), textcoords="offset points",
             xytext=(-60, -30), color=AZUL, fontsize=11,
             arrowprops=dict(arrowstyle="-", color=AZUL, lw=1))
ax1.set_title("Los dos términos de validación y su suma", fontsize=13)
ax1.set_ylabel("pérdida de validación")
ax1.legend(frameon=False, fontsize=10, loc="upper left")

pm = sorted(mov)
ax2.plot(pm, [mov[p] for p in pm], "o-", color=GRIS, lw=2, ms=7)
ax2.axvline(mejor, color=AZUL, ls=":", lw=1.5)
if mejor in mov:
    ax2.plot([mejor], [mov[mejor]], "o", color=AZUL, ms=12, zorder=5)
    ax2.annotate(f"el que la suma elige es\nel que menos se mueve ({mov[mejor]:.2f})".replace(".", ","),
                 (mejor, mov[mejor]), textcoords="offset points", xytext=(-155, 32),
                 color=AZUL, fontsize=11,
                 arrowprops=dict(arrowstyle="-", color=AZUL, lw=1))
ax2.set_title("Cuánto se mueve el video generado", fontsize=13)
ax2.set_ylabel("razón de movimiento contra el ground truth")

for ax in (ax1, ax2):
    ax.set_xlabel("paso de entrenamiento")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    ax.set_xticks(sorted(set(ps) | set(pm)))

fig.suptitle("Elegir el checkpoint por la pérdida elige el que menos se mueve", fontsize=15, y=0.99)
fig.tight_layout()
fig.savefig("figuras/seleccion_checkpoint.png", dpi=150)
print("suma por paso:", {p: round(suma[p], 4) for p in ps}, "-> minimo", mejor)
print("movimiento por paso:", {p: round(mov[p], 3) for p in pm})
