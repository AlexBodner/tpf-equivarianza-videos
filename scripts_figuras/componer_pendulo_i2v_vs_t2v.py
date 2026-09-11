"""videos/3_extrapolacion/pendulo_i2v_vs_t2v.mp4: el mismo péndulo a 65 cuadros,
condicionado y por texto, en un solo video.

Estaban en dos archivos separados y se veían casi iguales, así que no quedaba
claro cuál era cuál. Puestos lado a lado se ve lo único que los separa: de dónde
arranca cada uno. Condicionado arranca de 8 cuadros reales y se apaga pasado el
horizonte de entrenamiento; por texto arranca sólo del prompt y sostiene la
oscilación. Es el mismo modelo en los cuatro paneles.

Uso:  python3 scripts_figuras/componer_pendulo_i2v_vs_t2v.py [salida.mp4]
"""
import sys
from pathlib import Path

from componer_semillas import componer_grilla

I2V = Path("videos/8_generaciones_crudas/02_i2v_65cuadros")
T2V = Path("videos/8_generaciones_crudas/06_t2v_65cuadros")
SALIDA = "videos/3_extrapolacion/pendulo_i2v_vs_t2v.mp4"
HORIZONTE = 33   # cuadros vistos en entrenamiento; de ahí en adelante extrapola


def main(salida=SALIDA):
    grilla = [
        [("control 125 - condicionado (i2v)", I2V / "control_125/pendulum/00500.mp4"),
         ("control 125 - por texto (t2v)",    T2V / "control_125/pendulum_semilla0.mp4")],
        [("fisica 875 - condicionado (i2v)",  I2V / "fisica_875/pendulum/00500.mp4"),
         ("fisica 875 - por texto (t2v)",     T2V / "fisica_875/pendulum_semilla0.mp4")],
    ]
    faltan = [p for fila in grilla for _, p in fila if not p.exists()]
    if faltan:
        raise SystemExit(f"falta la generación: {faltan[0]}")
    T = componer_grilla(grilla, salida, marca=(HORIZONTE, "extrapolacion"))
    print(salida, T, "cuadros")


if __name__ == "__main__":
    main(*sys.argv[1:2])
