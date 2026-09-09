"""fig_barrido_ventanas.png — el barrido de ventanas de BPTT, sobre velocidad.

Reemplaza la figura de agosto, que estaba hecha sobre aceleración, con la pérdida sin
normalizar y λ mal calibrado, y afirmaba que ningún truncamiento replica al gradiente
exacto. Repetido sobre la cantidad que sí tiene señal, esa conclusión no se sostiene.
"""
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = [("full", "BPTT completo (12 pasos)", "#2e7d32", 27.8),
      ("ventana4", "no contigua [2,3,8,11]", "#2b7bba", 21.7),
      ("cola4", "cola [8..11] (DRaFT-4)", "#c0503d", 21.6),
      ("mejor6", "mejor-6 [0,1,2,4,5,6]", "#c98f2b", 23.4),
      ("mejor6comp", "mejor-6, magnitud compensada", "#7a5ba6", 23.4)]

def cargar(a):
    tr, val = {}, {}
    for l in open(f"{a}.jsonl"):
        d = json.loads(l)
        if d.get("type") == "val":
            val[d["step"]] = d["val_loss_rotation"]; continue
        if d.get("n_bruto") is None: continue
        n, dd, piso = d["n_bruto"], d["d_bruto"], d["piso"]
        tr[d["step"]] = max(n - piso, 0.0) / max(dd - piso, piso if piso > 0 else 1e-9)
    return tr, val

D = {a: cargar(a) for a, _, _, _ in BR}
PASOS = [50, 100, 150]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4))

for a, eti, col, sp in BR:
    ax1.plot(PASOS, [D[a][1][p] for p in PASOS], "o-", color=col, lw=2.2, ms=7,
             label=f"{eti} — {sp} s/paso")
ax1.set_title("(a) el instrumento crudo: MSE de validación sin normalizar", fontsize=12)
ax1.set_ylabel("val_loss_rotation (px²/cuadro²)")
ax1.legend(frameon=False, fontsize=9, loc="upper right")

# (b) La diferencia APAREADA contra el BPTT completo, con su intervalo. Un grafico de
# barras con las medianas invitaba a rankear los brazos; lo que hay que mostrar es que
# la diferencia no se distingue de cero.
import numpy as np
from scipy.stats import wilcoxon
ref = "full"
etiquetas, difs, ics, ps, colores = [], [], [], [], []
for a, eti, col, sp in BR:
    if a == ref: continue
    com = sorted(set(D[ref][0]) & set(D[a][0]))
    d = np.array([D[a][0][s] - D[ref][0][s] for s in com])
    boot = np.array([np.median(np.random.choice(d, len(d))) for _ in range(2000)])
    etiquetas.append(eti); difs.append(np.median(d))
    ics.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
    ps.append(wilcoxon(d).pvalue); colores.append(col)

y = range(len(etiquetas))
ax2.axvline(0, color="#333", lw=1.4)
for i, (v, (lo, hi), col) in enumerate(zip(difs, ics, colores)):
    ax2.plot([lo, hi], [i, i], color=col, lw=3, solid_capstyle="round")
    ax2.plot([v], [i], "o", color=col, ms=9)
    ax2.text(hi + 0.0012, i, f"p = {ps[i]:.2f}".replace(".", ","), va="center", fontsize=10,
             color="#555")
ax2.set_yticks(list(y)); ax2.set_yticklabels(etiquetas, fontsize=10); ax2.invert_yaxis()
ax2.set_xlim(min(l for l, _ in ics) * 1.35, max(h for _, h in ics) * 2.4)
ax2.set_title("(b) diferencia contra el BPTT completo,\nsobre la cantidad que realmente se minimiza",
              fontsize=12)
ax2.set_xlabel("mediana de la diferencia apareada (147 pasos) e IC 95 %")
ax2.text(0.5, -0.30, "ninguno de los diez pares posibles se distingue: $p$ entre 0,15 y 0,66",
         transform=ax2.transAxes, ha="center", fontsize=10, color="#555")

for ax in (ax1, ax2):
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
ax1.set_xlabel("paso de entrenamiento"); ax1.set_xticks(PASOS)

fig.suptitle("Barrido de ventanas de BPTT sobre velocidad: el instrumento crudo ordena los brazos, "
             "pero sobre lo que se optimiza ninguna ventana se distingue", fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
fig.savefig("fig_barrido_ventanas.png", dpi=150)
for e, v, p_ in zip(etiquetas, difs, ps): print(f"  {e:<32} {v:+.4f}  p={p_:.3f}")
