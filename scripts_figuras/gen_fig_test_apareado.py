"""fig_test_apareado.png: el test final, clip por clip.

Cada punto es un clip del test (30 por escenario, 88 con metrica). El eje x es el
brazo de control y el eje y el brazo con perdida fisica, asi que **debajo de la
diagonal el brazo con fisica es mejor**. Es el mismo par que entra al Wilcoxon, no
un resumen: se ven los 88.

Por que un apareado y no dos barras: los dos brazos ven el mismo clip, la misma
semilla y los mismos pesos iniciales, asi que la diferencia por clip tiene mucha
menos varianza que cada brazo por separado. Un grafico de medias esconde eso.

Sobre el volcado por clip: `run_eval.py` llena la lista de nombres antes del
`continue` que saltea los clips demasiado cortos, asi que en caida libre hay 30
nombres y 28 valores. Los VALORES estan en orden y los dos brazos saltean los
mismos clips, de modo que el apareo por indice es correcto; lo que no se puede es
ponerle nombre a un punto. Por eso la figura no etiqueta puntos.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14.5,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})

ESC = [("free_fall", "caída libre", "#2b7bba"),
       ("pendulum",  "péndulo",     "#c98f2b"),
       ("bouncing",  "rebote",      "#c0503d")]
D = json.load(open("test_final.json"))

def pares(met):
    out = {}
    for k, eti, col in ESC:
        c = D["control"][k]["per_clip"][met]
        f = D["fisica_velocidad"][k]["per_clip"][met]
        out[k] = [(a, b) for a, b in zip(c, f) if a is not None and b is not None]
    return out

def cota(met):
    v = [x for k, _, _ in ESC for x in D["quieto"][k]["per_clip"][met] if x is not None]
    return sum(v) / len(v)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6))

for ax, met, titulo in (
        (ax1, "l_rot_norm", "(a) lo que la pérdida pide: equivarianza\nρ, fracción del movimiento que viola la simetría"),
        (ax2, "vel_mae",    "(b) lo que queremos: física\nerror de velocidad contra el ground truth (px/cuadro)")):
    P = pares(met)
    todos = [v for k in P for v in P[k]]
    lo = min(min(a, b) for a, b in todos); hi = max(max(a, b) for a, b in todos)
    pad = 0.06 * (hi - lo); lo -= pad; hi += pad
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color="#2e7d32", alpha=0.055, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="#555", lw=1.6, zorder=1)
    for k, eti, col in ESC:
        xs = [a for a, _ in P[k]]; ys = [b for _, b in P[k]]
        mej = sum(1 for a, b in P[k] if b < a)
        ax.scatter(xs, ys, s=52, color=col, alpha=0.8, edgecolor="white", lw=1.0,
                   zorder=3, label=f"{eti}: {mej}/{len(P[k])}")
    q = cota(met)
    if lo < q < hi:
        ax.axhline(q, color="#888", ls=":", lw=1.6, zorder=2)
        ax.text(0.86, q, "cota: video quieto", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=12, color="#666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))
    n = len(todos); mej = sum(1 for a, b in todos if b < a)
    ax.set_title(titulo)
    ax.set_xlabel("control (sin física)")
    ax.set_ylabel("con pérdida física")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(alpha=0.22); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, title=f"mejora en {mej} de {n}", title_fontsize=13)
    ax.text(0.97, 0.06, "debajo de la diagonal\nla física gana", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12.5, color="#2e7d32", style="italic")

# Sin suptitle: en el poster lo dice el epigrafe y en el README el texto de arriba.
fig.tight_layout()
fig.savefig("fig_test_apareado.png", dpi=150)
print("ok")
