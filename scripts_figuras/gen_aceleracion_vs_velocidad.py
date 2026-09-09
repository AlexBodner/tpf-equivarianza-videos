"""Regenera figuras/aceleracion_vs_velocidad.png desde los logs publicados.

Las anotaciones salen de la prueba de tendencia preregistrada sobre la corrida
COMPLETA (primeros 100 pasos contra ultimos 100, Mann-Whitney), no de un corte
intermedio.
"""
import json, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

VEL = glob.glob("resultados/logs_entrenamiento/vel_run__*.jsonl")[0]
ACC = glob.glob("resultados/logs_entrenamiento/lambdafix_run__*.jsonl")[0]
BIN = 50
LAM = 0.0047  # lambda_rot de la corrida sobre velocidad

def leer(path, campo):
    """Pasos con fisica aplicada. El termino se reporta POR RAMA: loss_rotation/lambda/2."""
    pts = []
    for l in open(path):
        if not l.strip().startswith("{"):
            continue
        d = json.loads(l)
        if d.get("type") == "val" or d.get("step") is None or "loss_diffusion" not in d:
            continue
        if d.get("physics_skipped"):
            continue
        v = d["loss_rotation"] / LAM / 2 if campo == "loss_rotation" else d.get(campo)
        if v is not None:
            pts.append((d["step"], v))
    return pts

def binear(pts):
    caj = {}
    for s, v in pts:
        caj.setdefault((s - 1) // BIN, []).append(v)
    return ([k * BIN + BIN / 2 for k in sorted(caj)],
            [st.median(caj[k]) for k in sorted(caj)])

def tendencia(pts, alt):
    """Prueba preregistrada: medias de los primeros 100 pasos contra los ultimos 100.

    El test es de UNA COLA porque la enmienda 17 fijo la direccion antes de correr.
    """
    ini = [v for s, v in pts if s <= 100]
    fin = [v for s, v in pts if s > max(s for s, _ in pts) - 100]
    return st.mean(ini), st.mean(fin), mannwhitneyu(ini, fin, alternative=alt).pvalue

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2))
ROJO, AZUL = "#c0503d", "#2b7bba"

for path, color, etiq in ((ACC, ROJO, "sobre aceleración (cortada en el 807)"),
                          (VEL, AZUL, "sobre velocidad")):
    x, y = binear(leer(path, "loss_rotation"))
    ax1.plot(x, y, color=color, lw=2.2, label=etiq)
    x, y = binear(leer(path, "cos_ramas"))
    ax2.plot(x, y, color=color, lw=2.2)

i_v, f_v, p_v = tendencia(leer(VEL, "loss_rotation"), "greater")
i_c, f_c, p_c = tendencia(leer(VEL, "cos_ramas"), "less")
caida = (1 - f_v / i_v) * 100

ax1.set_title("El término de la pérdida a lo largo del entrenamiento", fontsize=13)
ax1.set_ylabel("pérdida por rama (0 = equivariante)")
ax1.text(0.06, 0.90, "plano en 0,5–0,6: no aprende", transform=ax1.transAxes,
         color=ROJO, fontsize=12)
ax1.text(0.40, 0.06, f"baja {caida:.0f} % (p = {p_v:.3f}, una cola)".replace(".", ","),
         transform=ax1.transAxes, color=AZUL, fontsize=12)
ax1.legend(loc="center", frameon=False, fontsize=11)
ax1.set_ylim(0, 0.75)

ax2.set_title("El acuerdo de dirección entre las dos ramas", fontsize=13)
ax2.set_ylabel("coseno entre R·q̂(orig) y q̂(rot)")
ax2.text(0.22, 0.94,
         f"sube: {i_c:.2f} → {f_c:.2f} (p = {p_c:.3f}, una cola)".replace(".", ","),
         transform=ax2.transAxes, color=AZUL, fontsize=12)
ax2.text(0.06, 0.12, "baja: 0,42 → 0,39", transform=ax2.transAxes,
         color=ROJO, fontsize=12)

for ax in (ax1, ax2):
    ax.set_xlabel("paso de entrenamiento")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("La misma pérdida sobre dos cantidades distintas del mismo video",
             fontsize=15, y=0.99)
fig.tight_layout()
fig.savefig("figuras/aceleracion_vs_velocidad.png", dpi=150)
print(f"velocidad: termino {i_v:.4f} -> {f_v:.4f} ({caida:.1f} %), p una cola = {p_v:.4f}")
print(f"velocidad: coseno  {i_c:.4f} -> {f_c:.4f}, p una cola = {p_c:.4f}")
