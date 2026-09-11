# Videos

Agrupados por el papel que cumplen en el póster y en el [README](../README.md).

**Todo lo que está fuera de `9_checkpoints_superados/` sale del par de checkpoints que el trabajo
reporta**: control en el paso 125 y brazo con física en el paso 875, elegidos cada uno por su propia
validación sobre los mismos ocho candidatos.

| carpeta | qué hay |
|---|---|
| `1_metodo/` | qué compara la pérdida (las dos ramas) y qué ve de cada video (el vector agregado) |
| `2_resultados/` | los dos brazos lado a lado: condicionado contra ground truth, por texto y fuera de dominio |
| `3_extrapolacion/` | 65 frames contra los 33 de entrenamiento, condicionado y por texto |
| `4_rodando_8_semillas/` | las ocho semillas del prompt que el modelo nunca vio, control arriba y física abajo |
| `5_tiro_vertical/` | el único escenario fuera del dataset que tiene ground truth |
| `6_lo_que_no_funciono/` | el colapso de la pérdida sin normalizar y la versión sobre aceleraciones |
| `9_checkpoints_superados/` | corridas y checkpoints anteriores, que **no** son el modelo reportado |

Dos aclaraciones sobre archivos concretos:

- `1_metodo/dos_ramas_de_la_perdida.mp4` es del **paso 250**. Las dos ramas del mismo clip sólo se
  volcaron en los pasos 250 y 1000, así que en el checkpoint elegido no existen. Ilustra qué compara la
  pérdida, que no depende del checkpoint.
- `9_checkpoints_superados/por_texto_sin_semilla.mp4` se generó antes de que existiera `--semilla_base`,
  así que consumió el RNG global y **no se puede volver a generar**. Su reemplazo reproducible es
  `2_resultados/por_texto_4_escenarios.mp4`.

Todos los MP4 están en H.264 Constrained Baseline, así que abren en QuickTime.
