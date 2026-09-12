"""videos/2_resultados/por_texto_4_escenarios.mp4: los cuatro escenarios por texto,
control arriba y brazo con física abajo.

Es el video que abre el README. La semilla de cada columna está fijada acá y el
pie de figura la dice: se eligió la más limpia de las ocho, o sea sin objetos
partidos ni fondo cargado, entre las que muestran el comportamiento típico del
prompt. Eso es legítimo para mirar y no para medir: la frecuencia sobre las ocho
semillas está en el README (conteo de duplicación) y en medir_rodando.py, y las
ocho de cada prompt están en videos/4_ocho_semillas/.

Uso:  python3 scripts_figuras/componer_por_texto.py [raiz] [salida.mp4]
"""
import sys
from pathlib import Path

from componer_semillas import componer_grilla

RAIZ = "videos/8_generaciones_crudas/05_semillas"
SALIDA = "videos/2_resultados/por_texto_4_escenarios.mp4"
FILAS = [("control_125", "control 125"), ("fisica_875", "fisica 875")]
COLUMNAS = [("t2v", "free_fall", "caida libre", 0), ("t2v", "pendulum", "pendulo", 0),
            ("t2v", "bouncing", "rebote", 2), ("ood", "rolling", "rodando", 6)]


def main(raiz=RAIZ, salida=SALIDA):
    raiz = Path(raiz)
    grilla = [[(f"{nombre} ({etiqueta})" if i == 0 else etiqueta,
                raiz / brazo / sub / f"{archivo}_semilla{semilla}.mp4")
               for sub, archivo, nombre, semilla in COLUMNAS]
              for i, (brazo, etiqueta) in enumerate(FILAS)]
    faltan = [p for fila in grilla for _, p in fila if not p.exists()]
    if faltan:
        raise SystemExit(f"falta la generación: {faltan[0]}")
    print(salida, componer_grilla(grilla, salida), "cuadros")


if __name__ == "__main__":
    main(*sys.argv[1:3])
