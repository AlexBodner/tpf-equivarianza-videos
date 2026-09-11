"""videos/2_resultados/fuera_de_dominio.mp4: los dos prompts que no están en el
dataset, con el modelo base arriba como referencia.

El base va arriba a propósito: es lo que SANA trae sin fine-tuning, y sirve para
ver que lo que hacen los dos brazos no es lo que el modelo ya sabía hacer.

La semilla de cada columna está fijada acá: rodando usa la 6, que es la que se
muestra en el README, así que es el mismo clip y no otro; el péndulo en cámara lenta
usa la 0, que es la primera y no una elegida por conveniencia. Las ocho de cada
prompt están en videos/4_ocho_semillas/.

Uso:  python3 scripts_figuras/componer_t2v_ood.py [raiz] [salida.mp4]
"""
import sys
from pathlib import Path

from componer_semillas import componer_grilla

RAIZ = "videos/8_generaciones_crudas/05_semillas"
SALIDA = "videos/2_resultados/fuera_de_dominio.mp4"
# Las etiquetas van cortas porque la columna mide 320 px: la fila de arriba lleva
# ademas el nombre del prompt, y "sin fine-tuning" queda para el pie de figura.
FILAS = [("base", "base"), ("control_125", "control 125"), ("fisica_875", "fisica 875")]
COLUMNAS = [("rolling_semilla6", "rodando"), ("pendulum_photo_semilla0", "pendulo camara lenta")]


def main(raiz=RAIZ, salida=SALIDA):
    raiz = Path(raiz)
    grilla = [[(f"{etiqueta} - {nombre}" if i == 0 else etiqueta,
                raiz / brazo / "ood" / f"{archivo}.mp4")
               for archivo, nombre in COLUMNAS]
              for i, (brazo, etiqueta) in enumerate(FILAS)]
    faltan = [p for fila in grilla for _, p in fila if not p.exists()]
    if faltan:
        raise SystemExit(f"falta la generación: {faltan[0]}")
    print(salida, componer_grilla(grilla, salida), "cuadros")


if __name__ == "__main__":
    main(*sys.argv[1:3])
