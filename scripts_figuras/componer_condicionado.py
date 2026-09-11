"""Condicionado (I2V), un archivo por escenario y por clip: ground truth, control
y brazo con física sobre el mismo clip.

Es el régimen que mide la Tabla 1, así que conviene poder mirar clip por clip lo
que ahí queda resumido en un número. El compuesto de los tres escenarios juntos
está en 2_resultados/condicionado_gt_control_fisica.mp4; esto lo abre.

Uso:  python3 scripts_figuras/componer_condicionado.py <raiz_generaciones> <salida>
"""
import sys
from pathlib import Path

from componer_semillas import componer

FILAS = [("ground_truth", "ground truth"),
         ("control_125", "control (paso 125)"),
         ("fisica_875", "con fisica (paso 875)")]
ESCENARIOS = {"caida_libre": "free_fall", "pendulo": "pendulum", "rebote": "bouncing"}
CLIPS = ["00500", "00501"]


def main(raiz, salida):
    raiz, salida = Path(raiz), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, esc in ESCENARIOS.items():
        for clip in CLIPS:
            rutas = [(f"{et} - {nombre}", raiz / brazo / esc / f"{clip}.mp4")
                     for brazo, et in FILAS]
            faltan = [p for _, p in rutas if not p.exists()]
            if faltan:
                raise SystemExit(f"falta la generación: {faltan[0]}")
            T = componer(rutas, salida / f"{nombre}_{clip}.mp4")
            print(f"{nombre}_{clip}.mp4  {T} cuadros")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
