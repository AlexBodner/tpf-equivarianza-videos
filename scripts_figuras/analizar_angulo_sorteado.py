"""Los numeros del barrido de angulos, desde los JSON. Sin GPU.

Reproduce las dos afirmaciones que el poster hace sobre esto:
  1. el desacuerdo entre ramas baja ~51 % con el angulo sorteado en [11, 44] grados;
  2. esa ventaja NO viene de medir cerca del rango entrenado: en el tercio de
     angulos altos es igual o mayor que en el de angulos bajos.

El error de velocidad de estos JSON no se reporta a proposito: se calcula sobre la
generacion SIN rotar, asi que no depende del angulo, y aca esta medido sobre menos
clips que el test principal. La medicion buena de esa columna es test_final.

Uso:  python3 scripts_figuras/analizar_angulo_sorteado.py resultados/evaluaciones
"""
import json
import sys
from pathlib import Path

import numpy as np

ESC = [("free_fall", "caída libre"), ("pendulum", "péndulo"), ("bouncing", "rebote")]
N_BOOT = 4000


def cargar(d: Path):
    ang, ctrl, fis = [], [], []
    por_escenario = []
    for k, t in ESC:
        j = json.load(open(d / f"angulo_sorteado_11_45__{k}.json"))
        pc, pf = j["control"][k]["per_clip"], j["fisica"][k]["per_clip"]
        a, c, f = [], [], []
        for x, y, z in zip(pc["rot_grados"], pc["l_rot_norm"], pf["l_rot_norm"]):
            if y is None or z is None:
                continue
            a.append(x); c.append(y); f.append(z)
        por_escenario.append((t, np.array(a), np.array(c), np.array(f)))
        ang += a; ctrl += c; fis += f
    return por_escenario, np.array(ang), np.array(ctrl), np.array(fis)


def mejora_ic(c, f, rng):
    """Mejora relativa al control, en %, con bootstrap sobre clips."""
    idx = rng.integers(0, len(c), size=(N_BOOT, len(c)))
    rel = (c[idx].mean(axis=1) - f[idx].mean(axis=1)) / c[idx].mean(axis=1) * 100
    lo, hi = np.percentile(rel, [2.5, 97.5])
    return (c.mean() - f.mean()) / c.mean() * 100, lo, hi


def main(d):
    rng = np.random.default_rng(0)
    por_esc, ang, ctrl, fis = cargar(Path(d))
    print("=== desacuerdo entre ramas, ángulo sorteado por clip ===")
    for t, a, c, f in por_esc:
        p, lo, hi = mejora_ic(c, f, rng)
        print(f"  {t:13s} n={len(c):2d}  {min(a):5.1f}–{max(a):5.1f}°  "
              f"{c.mean():.4f} → {f.mean():.4f}   {p:+5.1f}% [{lo:+.1f},{hi:+.1f}]"
              + ("" if lo < 0 < hi else " *"))
    p, lo, hi = mejora_ic(ctrl, fis, rng)
    print(f"  {'LOS TRES':13s} n={len(ctrl):2d}  {min(ang):5.1f}–{max(ang):5.1f}°  "
          f"{ctrl.mean():.4f} → {fis.mean():.4f}   {p:+5.1f}% [{lo:+.1f},{hi:+.1f}]"
          + ("" if lo < 0 < hi else " *"))

    print("\n=== ¿la ventaja viene de medir cerca del rango entrenado? ===")
    for nombre, V in [("control", ctrl), ("física ", fis)]:
        print(f"  correlación ángulo vs desacuerdo, {nombre}: r = {np.corrcoef(ang, V)[0,1]:+.3f}")
    orden = np.argsort(ang)
    tercio = len(orden) // 3
    for nombre, idx in [("bajos ", orden[:tercio]), ("medios", orden[tercio:2*tercio]),
                        ("altos ", orden[2*tercio:])]:
        c, f = ctrl[idx], fis[idx]
        print(f"  {nombre}  n={len(idx):2d}  ángulo medio={ang[idx].mean():5.1f}°  "
              f"mejora={(c.mean()-f.mean())/c.mean()*100:+5.1f}%")
    print("  → la mejora no se achica en los ángulos altos: no es un efecto de cercanía.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/evaluaciones")
