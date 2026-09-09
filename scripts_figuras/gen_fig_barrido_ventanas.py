"""fig_barrido_ventanas.png: el barrido de ventanas de BPTT, sobre velocidad.

Muestra la pérdida física, que es el cociente con el piso del estimador restado, en
entrenamiento y en validación. Reemplaza a la figura de agosto, que estaba medida sobre
aceleraciones con la pérdida sin normalizar.

Dos decisiones de dibujo:
* La curva de entrenamiento es una mediana móvil CENTRADA. Con un promedio exponencial
  inicializado en el primer valor, la curva arrancaba abajo y tardaba treinta pasos en
  alcanzar su nivel, lo que se leía como que la pérdida sube. No sube: promediada por
  ventanas de treinta pasos va de 0,220 a 0,137 y se queda ahí.
* Los brazos se agrupan por cuántos pasos se retropropagan, porque de eso depende el
  costo: la cola y la ventana no contigua usan cuatro pasos y cuestan lo mismo.
"""
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = [("full",       "BPTT completo",               12, "#2e7d32"),
      ("ventana4",   "no contigua [2,3,8,11]",       4, "#2b7bba"),
      ("cola4",      "cola [8..11] (DRaFT-4)",       4, "#c0503d"),
      ("mejor6",     "mejor-6 [0,1,2,4,5,6]",        6, "#c98f2b"),
      ("mejor6comp", "mejor-6, magnitud compensada", 6, "#7a5ba6")]
COSTO = {4: 21.6, 6: 23.4, 12: 27.8}   # s/paso medianos por cantidad de pasos

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

def mediana_movil(v, k=15):
    """Centrada: sin transitorio de arranque, y robusta a los picos por paso."""
    h = k // 2
    return [st.median(v[max(0, i - h): min(len(v), i + h + 1)]) for i in range(len(v))]

D = {a: cargar(a) for a, _, _, _ in BR}
PASOS_VAL = [50, 100, 150]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

for a, eti, k, col in BR:
    tr = D[a][0]; xs = sorted(tr)
    ax1.plot(xs, mediana_movil([tr[x] for x in xs]), color=col, lw=2.3,
             label=f"{eti} ({k} pasos)")
ax1.set_title("(a) entrenamiento: 147 pasos con física por brazo", fontsize=12)
ax1.set_ylabel("pérdida física"); ax1.set_xlabel("paso de entrenamiento")
ax1.set_ylim(bottom=0)

# El costo, como tabla chica adentro del panel: depende de la cantidad de pasos.
lineas = ["pasos    s/paso"] + [f"  {k:>2}      {COSTO[k]:.1f}" for k in (4, 6, 12)]
ax1.text(0.985, 0.97, "\n".join(lineas), transform=ax1.transAxes, ha="right", va="top",
         fontsize=9.5, family="monospace", color="#444",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f4", ec="#cccccc", lw=0.8))

for a, eti, k, col in BR:
    ax2.plot(PASOS_VAL, [D[a][1][p] for p in PASOS_VAL], "o-", color=col, lw=2.1, ms=7)
ax2.set_title("(b) validación: 20 clips, sólo tres mediciones por brazo", fontsize=12)
ax2.set_ylabel("pérdida física de validación"); ax2.set_xlabel("paso de entrenamiento")
ax2.set_xticks(PASOS_VAL)

for ax in (ax1, ax2):
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

manijas, etiquetas = ax1.get_legend_handles_labels()
fig.legend(manijas, etiquetas, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Barrido de ventanas de BPTT sobre velocidad: ninguna ventana se distingue sobre lo que se minimiza",
             fontsize=12.5, y=0.98)
fig.tight_layout(rect=[0, 0.11, 1, 0.93])
fig.savefig("fig_barrido_ventanas.png", dpi=150)
for a, eti, k, _ in BR:
    v = list(D[a][0].values())
    print(f"  {eti:<32} {k:>2} pasos  primeros 30 {st.mean(v[:30]):.3f}  ultimos 30 {st.mean(v[-30:]):.3f}")
