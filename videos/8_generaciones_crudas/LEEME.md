# Generaciones crudas de los checkpoints elegidos

Cada video de este árbol es **una generación sin componer**: un brazo, un prompt, una semilla. Los
videos comparados que se muestran en el README y el póster se arman a partir de acá, y los scripts de
[`scripts_figuras/`](../../scripts_figuras) leen esta carpeta, así que las figuras se rehacen sin salir
del repositorio.

Todo sale del **mismo par de checkpoints** que reporta el trabajo:

| brazo | paso | cómo se eligió |
|---|---|---|
| **control** | 125 | mínimo de su validación de difusión |
| **con física** | 875 | mínimo de la suma de sus dos términos de validación (difusión + λ·rotación) |
| **base** | sin entrenar | SANA-Video 2B sin fine-tuning, de referencia |

Los dos brazos son idénticos salvo el peso del término físico: mismo clip, mismo ruido, mismos pesos
iniciales. Donde aparece `ground_truth` es el clip del simulador, no una generación.

| carpeta | qué es | archivos |
|---|---|---|
| [`01_i2v_33cuadros/`](01_i2v_33cuadros) | el régimen de entrenamiento: 33 cuadros, condicionado en los 2 primeros cuadros latentes, que en píxeles son 8. Incluye el ground truth | 3 brazos × 3 escenarios × 2 clips |
| [`02_i2v_65cuadros/`](02_i2v_65cuadros) | 65 cuadros, o sea extrapolación: el modelo entrenó con 33. El período del péndulo es 37,4 cuadros, así que recién acá entra una oscilación completa | 3 brazos × 3 escenarios × 2 clips |
| [`03_texto_a_video/`](03_texto_a_video) | sin condicionamiento, sólo desde el prompt | 3 brazos × 3 escenarios × 2 semillas |
| [`04_fuera_de_dominio/`](04_fuera_de_dominio) | prompts que no están en el dataset: una pelota rodando y el mismo péndulo pedido con otras palabras y en cámara lenta | 3 brazos × 2 prompts × 2 semillas |
| [`05_semillas/`](05_semillas) | 8 semillas por prompt, para que lo que se muestra no dependa de una sola tirada | 3 brazos × 5 prompts × 8 semillas |
| [`06_t2v_65cuadros/`](06_t2v_65cuadros) | lo mismo que `02`, pero sin condicionamiento: 65 cuadros generados sólo desde el texto | 2 brazos × 3 escenarios × 2 semillas |

`00500` y `00501` son clips del conjunto held-out; `semillaN` es el número de semilla. Los nombres de
escenario están en inglés porque son los del simulador y los scripts los leen así.

Todos los MP4 están en H.264 Constrained Baseline, así que abren en QuickTime. Los prompts completos
están en [PROMPTS.md](../../PROMPTS.md).

## Cómo se rehacen los videos comparados

```bash
python3 scripts_figuras/componer_semillas.py videos/8_generaciones_crudas/05_semillas videos/4_ocho_semillas
python3 scripts_figuras/componer_condicionado.py videos/8_generaciones_crudas/01_i2v_33cuadros videos/2_resultados/condicionado_por_escenario
python3 scripts_figuras/componer_t2v_ood.py
python3 scripts_figuras/componer_pendulo_i2v_vs_t2v.py
```

## Qué mirar, sabiendo qué buscar

**Rebote condicionado, clip `00500` de `01_i2v_33cuadros`.** El ground truth muestra la pelota cayendo,
tocando el piso y volviendo a subir. Ninguno de los dos brazos reproduce ese rebote: el control la
difumina antes de llegar y el brazo con física llega al piso y desaparece. Es **un** clip condicionado;
en `03_texto_a_video`, generando sólo desde el prompt, el brazo con física sí produce un rebote.

**En 65 cuadros el desvanecimiento es de los dos brazos.** La pelota se apaga pasado el horizonte de
entrenamiento tanto en el control como en el brazo con física, y el área que ocupa cae parecido en los
dos. No distingue brazos: distingue régimen, porque generando por texto a 65 cuadros ninguno se apaga.
