# Videos

Índice de todo lo que hay, para llegar a cualquier ejemplo sin buscar. Los videos comparados de las
carpetas 1 a 6 se arman desde [`8_generaciones_crudas/`](8_generaciones_crudas), donde está cada
generación suelta.

**Todo lo que está fuera de `9_checkpoints_superados/` sale del par de checkpoints que el trabajo
reporta**: control en el paso 125 y brazo con física en el paso 875, elegidos cada uno por su propia
validación sobre los mismos ocho candidatos. Los pesos de los dos están en
[Hugging Face](https://huggingface.co/AlexBodner/tpf-equivarianza-video/tree/main/checkpoints).

| carpeta | qué hay |
|---|---|
| [`1_metodo/`](1_metodo) | qué compara la pérdida (las dos ramas) y qué ve de cada video (el vector agregado) |
| [`2_resultados/`](2_resultados) | los dos brazos lado a lado: condicionado contra ground truth, por texto y fuera de dominio |
| [`3_extrapolacion/`](3_extrapolacion) | 65 cuadros contra los 33 de entrenamiento, condicionado y por texto |
| [`4_ocho_semillas/`](4_ocho_semillas) | ocho semillas de cada prompt por texto, control arriba y física abajo |
| [`5_tiro_vertical/`](5_tiro_vertical) | el único escenario fuera del dataset que tiene ground truth |
| [`6_lo_que_no_funciono/`](6_lo_que_no_funciono) | el colapso de la pérdida sin normalizar y la versión sobre aceleraciones |
| [`8_generaciones_crudas/`](8_generaciones_crudas) | las 190 generaciones sueltas de las que sale todo lo de arriba |
| [`9_checkpoints_superados/`](9_checkpoints_superados) | corridas y checkpoints anteriores, que **no** son el modelo reportado |

## Archivo por archivo

**1_metodo**
[`que_ve_la_perdida.mp4`](1_metodo/que_ve_la_perdida.mp4) · el video, el flujo de RAFT y el único vector que queda
[`dos_ramas_de_la_perdida.mp4`](1_metodo/dos_ramas_de_la_perdida.mp4) · la escena original contra la rotada 45°

**2_resultados**
[`condicionado_gt_control_fisica.mp4`](2_resultados/condicionado_gt_control_fisica.mp4) · los tres escenarios, ground truth contra los dos brazos
[`por_texto_4_escenarios.mp4`](2_resultados/por_texto_4_escenarios.mp4) · caída libre, péndulo, rebote y rodando, sólo desde el texto
[`fuera_de_dominio.mp4`](2_resultados/fuera_de_dominio.mp4) · rodando y péndulo fotográfico, con el modelo base de referencia
[`condicionado_por_escenario/`](2_resultados/condicionado_por_escenario) · el mismo contraste condicionado, un archivo por escenario y por clip

**3_extrapolacion**
[`por_texto_65_frames_semilla0.mp4`](3_extrapolacion/por_texto_65_frames_semilla0.mp4) y [`semilla1`](3_extrapolacion/por_texto_65_frames_semilla1.mp4) · el péndulo por texto, que sostiene la oscilación
[`condicionado_65_frames.mp4`](3_extrapolacion/condicionado_65_frames.mp4) · el mismo horizonte condicionado, donde ninguno de los dos la sostiene

**4_ocho_semillas**, ocho semillas por prompt, `semilla0` a `semilla7`
[`rodando/`](4_ocho_semillas/rodando) · el prompt que el modelo nunca vio; termina rodando en 5 de 8 con física y en ninguna sin ella
[`rebote/`](4_ocho_semillas/rebote) · donde el control duplica la pelota y le cambia el color
[`pendulo/`](4_ocho_semillas/pendulo) · la oscilación sostenida
[`caida_libre/`](4_ocho_semillas/caida_libre) · el escenario donde los dos brazos salen parecidos
[`pendulo_foto/`](4_ocho_semillas/pendulo_foto) · el mismo péndulo pedido en estética fotográfica

**5_tiro_vertical**: [`clip_10000.mp4`](5_tiro_vertical/clip_10000.mp4) y [`clip_10001.mp4`](5_tiro_vertical/clip_10001.mp4)

**6_lo_que_no_funciono**
[`colapso_sin_normalizar.mp4`](6_lo_que_no_funciono/colapso_sin_normalizar.mp4) · la pérdida sin normalizar, que se minimiza moviéndose menos
[`aceleracion_evolucion_rebote.mp4`](6_lo_que_no_funciono/aceleracion_evolucion_rebote.mp4), [`aceleracion_evolucion_caida_libre.mp4`](6_lo_que_no_funciono/aceleracion_evolucion_caida_libre.mp4) y [`aceleracion_pendulo.mp4`](6_lo_que_no_funciono/aceleracion_pendulo.mp4) · la versión sobre aceleraciones

## Dos aclaraciones sobre archivos concretos

- `1_metodo/dos_ramas_de_la_perdida.mp4` es del **paso 250**. Las dos ramas del mismo clip sólo se
  volcaron en los pasos 250 y 1000, así que en el checkpoint elegido no existen. Ilustra qué compara la
  pérdida, que no depende del checkpoint.
- `9_checkpoints_superados/por_texto_sin_semilla.mp4` se generó antes de que existiera `--semilla_base`,
  así que consumió el RNG global y **no se puede volver a generar**. Su reemplazo reproducible es
  `2_resultados/por_texto_4_escenarios.mp4`.

Todos los MP4 están en H.264 Constrained Baseline, así que abren en QuickTime.
