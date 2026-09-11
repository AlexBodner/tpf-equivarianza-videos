"""figuras/seleccion_por_validacion.png — qué checkpoint elige cada brazo, y por qué.

Cada brazo se elige por SU propio objetivo de validación, medido sobre los mismos 8
checkpoints y los mismos 20 clips de validación: el control por su pérdida de difusión,
que es todo lo que entrena; el brazo con física por la suma de sus dos términos.
"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL = "resultados/validaciones"
fis  = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{VAL}/fisica_8_checkpoints.json"))
ctrl = json.load(open(sys.argv[2] if len(sys.argv) > 2 else f"{VAL}/control_8_checkpoints.json"))
LAM = fis["lambda_rot"]
PASOS = sorted(int(k) for k in fis["por_paso"])
dif_f = [fis["por_paso"][str(p)]["val_loss_diffusion"] for p in PASOS]
rot_f = [fis["por_paso"][str(p)]["val_loss_rotation"] for p in PASOS]
tot_f = [d + LAM * r for d, r in zip(dif_f, rot_f)]
dif_c = [ctrl[str(p)]["val_loss_diffusion"] for p in PASOS]

GRIS, AZUL, ROJO = "#6b6b6b", "#2b7bba", "#c0503d"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax1.plot(PASOS, dif_c, "o-", color=GRIS, lw=2.2, ms=7, label="validación de difusión")
mc = PASOS[dif_c.index(min(dif_c))]
ax1.plot([mc], [min(dif_c)], "o", ms=15, mfc="none", mec=GRIS, mew=2.5)
ax1.annotate(f"elegido: paso {mc}", (mc, min(dif_c)), textcoords="offset points",
             xytext=(16, 2), color=GRIS, fontsize=11)
ax1.set_title("Control: su objetivo es sólo la difusión", fontsize=12.5)
ax1.set_ylabel("pérdida de validación")

ax2.plot(PASOS, dif_f, "o-", color=GRIS, lw=1.8, ms=6, label="difusión")
ax2.plot(PASOS, [LAM * r for r in rot_f], "o-", color=ROJO, lw=1.8, ms=6,
         label="λ · rotación (sobre velocidades)")
ax2.plot(PASOS, tot_f, "o-", color=AZUL, lw=2.6, ms=8, label="suma: su objetivo")
mf = PASOS[tot_f.index(min(tot_f))]
ax2.plot([mf], [min(tot_f)], "o", ms=16, mfc="none", mec=AZUL, mew=2.5)
ax2.annotate(f"elegido: paso {mf}", (mf, min(tot_f)), textcoords="offset points",
             xytext=(-108, -28), color=AZUL, fontsize=11,
             arrowprops=dict(arrowstyle="-", color=AZUL, lw=1))
ax2.set_title("Con pérdida física: su objetivo son los dos términos", fontsize=12.5)

for ax in (ax1, ax2):
    ax.set_xlabel("paso de entrenamiento"); ax.set_xticks(PASOS)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.24); ax.legend(frameon=False, fontsize=10)

fig.suptitle("Cada brazo se elige por su propio objetivo, sobre los mismos 8 candidatos "
             "y los mismos 20 clips de validación", fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figuras/seleccion_por_validacion.png", dpi=150)
print(f"control elige {mc} | fisica elige {mf}")
