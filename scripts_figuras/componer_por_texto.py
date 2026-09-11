"""videos/2_resultados/por_texto_4_escenarios.mp4: los cuatro escenarios por texto,
control arriba y brazo con física abajo.

Es el video que abre el README. Usa la semilla 0 en las cuatro columnas, que es la
primera y no una elegida por conveniencia; las otras siete de cada prompt están en
videos/4_ocho_semillas/.

Uso:  python3 scripts_figuras/componer_por_texto.py [raiz] [salida.mp4]
"""
import sys
from pathlib import Path

from componer_semillas import componer_grilla

RAIZ = "videos/8_generaciones_crudas/05_semillas"
SALIDA = "videos/2_resultados/por_texto_4_escenarios.mp4"
SEMILLA = 0
FILAS = [("control_125", "control 125"), ("fisica_875", "fisica 875")]
COLUMNAS = [("t2v", "free_fall", "caida libre"), ("t2v", "pendulum", "pendulo"),
            ("t2v", "bouncing", "rebote"), ("ood", "rolling", "rodando")]


def main(raiz=RAIZ, salida=SALIDA):
    raiz = Path(raiz)
    grilla = [[(f"{nombre} ({etiqueta})" if i == 0 else etiqueta,
                raiz / brazo / sub / f"{archivo}_semilla{SEMILLA}.mp4")
               for sub, archivo, nombre in COLUMNAS]
              for i, (brazo, etiqueta) in enumerate(FILAS)]
    faltan = [p for fila in grilla for _, p in fila if not p.exists()]
    if faltan:
        raise SystemExit(f"falta la generación: {faltan[0]}")
    print(salida, componer_grilla(grilla, salida), "cuadros")


if __name__ == "__main__":
    main(*sys.argv[1:3])
