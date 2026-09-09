"""fig_barrido_ventanas.png — el barrido de ventanas de BPTT, sobre velocidad.

Reemplaza la figura de agosto, que estaba sobre aceleración con la pérdida sin
normalizar. Muestra la pérdida física —el cociente con el piso del estimador restado,
que es lo que el entrenamiento minimiza— en entrenamiento y en validación.

El costo depende de CUÁNTOS pasos se retropropagan, no de cuáles: la cola [8..11] y la
ventana no contigua [2,3,8,11] usan cuatro y cuestan lo mismo. Por eso los brazos se
agrupan por esa cantidad en vez de poner los segundos por brazo.
"""
import json, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = [("full",       "BPTT completo",              12, "#2e7d32"),
      ("ventana4",   "no contigua [2,3,8,11]",      4, "#2b7bba"),
      ("cola4",      "cola [8..11] (DRaFT-4)",      4, "#c0503d"),
      ("mejor6",     "mejor-6 [0,1,2,4,5,6]",       6, "#c98f2b"),
      ("mejor6comp", "mejor-6, magnitud compensada", 6, "#7a5ba6")]
# s/paso medianos, agrupados por cantidad de pasos retropropagados
COSTO = {4: 21.6, 6: 23.4, 12: 27.8}

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

def ema(v, alfa=0.035):
    s, out = v[0], []
    for x in v:
        s = alfa * x + (1 - alfa) * s; out.append(s)
    return out

D = {a: cargar(a) for a, _, _, _ in BR}
PASOS_VAL = [50, 100, 150]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

for a, eti, k, col in BR:
    tr = D[a][0]; xs = sorted(tr)
    # Sólo el promedio móvil: la traza por paso tiene picos de hasta 1,5 que aplastan
    # todo lo demás y no aportan (la mediana está en 0,08).
    ax1.plot(xs, ema([tr[x] for x in xs]), color=col, lw=2.3, label=f"{eti} — {k} pasos")
ax1.set_title("(a) entrenamiento — promedio móvil sobre los 147 pasos con física", fontsize=11.5)
ax1.set_ylabel("pérdida física"); ax1.set_xlabel("paso de entrenamiento")
# La leyenda va abajo y compartida: adentro tapaba las curvas.

for a, eti, k, col in BR:
    val = D[a][1]
    ax2.plot(PASOS_VAL, [val[p] for p in PASOS_VAL], "o-", color=col, lw=2.1, ms=7)
ax2.set_title("(b) validación — 20 clips, 3 mediciones por brazo", fontsize=12)
ax2.set_ylabel("pérdida física de validación")
ax2.set_xlabel("paso de entrenamiento"); ax2.set_xticks(PASOS_VAL)

for ax in (ax1, ax2):
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

costo = "   ·   ".join(f"{k} pasos: {COSTO[k]:.1f} s/paso" for k in (4, 6, 12))
fig.suptitle("Barrido de ventanas de BPTT sobre velocidad: ninguna ventana se distingue sobre lo que se minimiza\n"
             f"el costo depende de cuántos pasos se retropropagan, no de cuáles  —  {costo}",
             fontsize=12, y=0.995)
manijas, etiquetas = ax1.get_legend_handles_labels()
fig.legend(manijas, etiquetas, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.115, 1, 0.87])
fig.savefig("fig_barrido_ventanas.png", dpi=150)
for a, eti, k, _ in BR:
    tr = D[a][0]
    print(f"  {eti:<32} {k:>2} pasos  train mediana {st.median(list(tr.values())):.4f}  val150 {D[a][1][150]:.2f}")
