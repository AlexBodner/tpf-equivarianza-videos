"""figuras/por_checkpoint_todas_las_metricas.png

Las seis metricas que tenemos por checkpoint, para los dos brazos. Las tres de arriba
son de EQUIVARIANZA (lo que la perdida pide) y las tres de abajo son de FISICA (lo que
queremos que mejore). El punto del trabajo es que una cosa sube y la otra no.
"""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PASOS = [125, 250, 375, 500, 625, 750, 875, 1000]
GRIS, AZUL = "#6b6b6b", "#2b7bba"

# --- equivarianza, medida sobre velocidades fuera del bucle (n=12 por checkpoint)
eq = json.load(open("equiv_ckpt.json"))
def eq_med(brazo, paso, campo):
    filas = eq.get(f"{brazo}/{paso}")
    return st.median([r[campo] for r in filas]) if filas else None
def eq_p(paso, campo):
    c, f = eq.get(f"ctrl/{paso}"), eq.get(f"equiv_vel/{paso}")
    if not c or not f: return None
    cc = {(r["escenario"], r["clip"]): r[campo] for r in c}
    ff = {(r["escenario"], r["clip"]): r[campo] for r in f}
    com = sorted(set(cc) & set(ff))
    return wilcoxon([ff[k] - cc[k] for k in com]).pvalue

# --- fisica, de las evaluaciones held-out por checkpoint (n=30)
def ev(paso, brazo, campo):
    d = json.load(open(f"evals_ckpt/paso_{paso}.json"))[brazo]
    vals = [v for e in ("free_fall", "pendulum", "bouncing")
            for v in d[e]["per_clip"][campo] if v is not None]
    return st.mean(vals)
def ev_p(paso, campo):
    d = json.load(open(f"evals_ckpt/paso_{paso}.json"))
    a, b = [], []
    for e in ("free_fall", "pendulum", "bouncing"):
        a += d["ctrl_l40s"][e]["per_clip"][campo]
        b += d["equiv_vel"][e]["per_clip"][campo]
    return wilcoxon([y - x for x, y in zip(a, b)]).pvalue

PANELES = [
    ("ρ: fracción del movimiento\nque viola la simetría", "↓ mejor",
     lambda p, b: eq_med(b, p, "rho"), lambda p: eq_p(p, "rho"), "equiv", "menor"),
    ("Coseno entre las dos ramas", "↑ mejor",
     lambda p, b: eq_med(b, p, "cos"), lambda p: eq_p(p, "cos"), "equiv", "mayor"),
    ("Energía del movimiento\n(control de la ruta degenerada)", "sin dirección buena",
     lambda p, b: eq_med(b, p, "energia"), lambda p: eq_p(p, "energia"), "equiv", None),
    ("Error de velocidad contra\nel ground truth", "↓ mejor",
     lambda p, b: ev(p, {"ctrl": "ctrl_l40s", "equiv_vel": "equiv_vel"}[b], "vel_mae"),
     lambda p: ev_p(p, "vel_mae"), "fisica", "menor"),
    ("Razón de movimiento", "→1 mejor",
     lambda p, b: ev(p, {"ctrl": "ctrl_l40s", "equiv_vel": "equiv_vel"}[b], "mov_ratio"),
     lambda p: ev_p(p, "mov_ratio"), "fisica", "cerca1"),
    ("Jerk", "↓ más suave, pero lo gana un video quieto",
     lambda p, b: ev(p, {"ctrl": "ctrl_l40s", "equiv_vel": "equiv_vel"}[b], "jerk_rms"),
     lambda p: ev_p(p, "jerk_rms"), "fisica", "menor"),
]

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
VERDE, NARANJA = "#2e7d32", "#c0503d"
for ax, (titulo, dir_, f, fp, grupo, mejor_si) in zip(axes.ravel(), PANELES):
    yc = [f(p, "ctrl") for p in PASOS]
    yf = [f(p, "equiv_vel") for p in PASOS]
    ax.plot(PASOS, yc, "o-", color=GRIS, lw=2, ms=6, label="control (sin física)")
    ax.plot(PASOS, yf, "o-", color=AZUL, lw=2.4, ms=7, label="con pérdida física")
    # Marca los pasos con diferencia significativa Y para que lado va: un triangulo
    # hacia arriba si favorece al brazo con fisica, hacia abajo si lo perjudica.
    # Sin la direccion, la marca decia "hay diferencia" y se leia como "gana".
    for p, y, c in zip(PASOS, yf, yc):
        pv = fp(p)
        if pv is None or pv >= 0.05 or mejor_si is None:
            continue
        if mejor_si == "menor":   favorable = y < c
        elif mejor_si == "mayor": favorable = y > c
        else:                     favorable = abs(y - 1) < abs(c - 1)
        ax.plot([p], [y], marker="^" if favorable else "v", ms=13,
                color=VERDE if favorable else NARANJA, zorder=5,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_title(f"{titulo}   ({dir_})", fontsize=11.5)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("paso de entrenamiento"); ax.set_xticks(PASOS[::2] + [1000])
    if grupo == "fisica":
        ax.set_facecolor("#faf7f2")

axes[0][0].legend(frameon=False, fontsize=10, loc="lower right")
fig.suptitle("Arriba: lo que la pérdida pide (equivarianza).   Abajo: lo que queremos que mejore (física).\n"
             "▲ verde: diferencia significativa a favor del brazo con física.   ▼ naranja: significativa en contra.  (p < 0,05, apareado por clip)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("panel_checkpoints.png", dpi=140)
print("listo")
for p in PASOS:
    print(f"paso {p}: rho p={eq_p(p,'rho'):.3f}  cos p={eq_p(p,'cos'):.3f}  vel_mae p={ev_p(p,'vel_mae'):.3f}")
