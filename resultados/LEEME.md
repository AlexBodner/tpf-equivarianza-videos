# Resultados crudos, logs y configuraciones

Todo lo que hace falta para verificar o rehacer los números del README, fuera de la máquina de
entrenamiento. Las instancias de AWS son efímeras: acá está la copia durable.

## Qué hay

| carpeta | contenido |
|---|---|
| `evaluaciones/` | los JSON de cada evaluación: métricas por clip, por escenario y por brazo. El nombre dice corrida y checkpoint, por ejemplo `e4vel__paso_1000.json`. Incluye los ajustes de parámetros físicos (`analisis*`) y las mediciones de equivarianza directa (`equiv.txt`). |
| `logs_entrenamiento/` | un JSONL por corrida, una línea por paso: pérdidas, normas de gradiente, componentes del cociente, coseno entre ramas, clip usado, VRAM y segundos por paso. |
| `configs/` | la configuración exacta de cada corrida y de cada sonda de λ. |
| `informes/` | `RESULTADOS.md` (lectura), `NUMEROS_EN_CRUDO.md` (todas las tablas) y el prerregistro con sus 33 enmiendas, que documenta cada decisión con su fecha y si se tomó antes o después de ver los datos. |
| `diag_raft.txt`, `diag_raft_vel.txt` | la salida cruda del diagnóstico del instrumento sobre aceleración y sobre velocidad. |

## Los resultados finales

| archivo | qué es |
|---|---|
| `evaluaciones/test_final__88_clips_nunca_mirados.json` | **el resultado principal**: los 88 clips held-out que no se miraron nunca, control en el paso 125 y brazo con física en el 875, cada uno elegido por su propia validación |
| `evaluaciones/elegidos__i2v_65_cuadros.json` | generación a 65 cuadros (el entrenamiento vio 33) en los checkpoints elegidos, 125 y 875. Las métricas puntúan sólo los primeros 33, que es hasta donde llega el ground truth: los otros 32 son para mirar. n = 2 por escenario |
| `evaluaciones/candidatos__n60_velocidades.json` | los siete brazos sobre los primeros 20 clips por escenario, con la equivarianza ya medida sobre velocidades |
| `evaluaciones/equivarianza_por_checkpoint.json` | equivarianza fuera del bucle en los 8 checkpoints de los dos brazos (ojo: 3 escenas con sus 4 variantes, no 12 clips distintos) |
| `validaciones/fisica_8_checkpoints.json` | la validación recalculada sobre velocidades que elige el checkpoint del brazo con física |
| `validaciones/control_8_checkpoints.json` | la del control |
| `logs_barrido/*.jsonl` | los cinco brazos del barrido de ventanas de BPTT, una línea por paso |

## Las evaluaciones, y cuál sirve para qué

| evaluación | qué mide | brazos | estado |
|---|---|---|---|
| `*paso_N.json`, `indist`, `ood` | trayectoria: error de velocidad, razón de movimiento, jerk, MAE de aceleración, y las cotas triviales | base, control y con física, apareados | **es la que sostiene los resultados** |
| las columnas `l_rot` y `l_rot_norm` de esos mismos JSON | equivarianza | los tres | **medidas sobre aceleraciones**, que sobre video generado son ruido. No interpretables. Se rehacen sobre velocidades |
| `equiv_por_checkpoint.json` | equivarianza sobre velocidades, fuera del bucle de entrenamiento | control y con física | correcta, n = 12 por checkpoint |
| `*equiv*.txt`, `*diagnostico_completo.log` | equivarianza directa: diferencia de píxeles y coseno entre ramas | control y con física | correcta, n = 9, sólo en los pasos 250 y 1000 |
| `*analisis*.txt` (ajuste de parámetros físicos) | residuo del ajuste, aceleración recuperada, ω, amplitud, g | **sólo el brazo con física** | **no usar para comparar.** No tiene control al lado, así que no permite ninguna comparación; además el trazador de centroides detecta entre el 55 % y el 81 % de los cuadros, y en caída libre el ajuste parabólico mide suavidad y no gravedad, porque el ground truth ya va a velocidad terminal |

Ninguna afirmación del README ni del póster se apoya en la última fila.

## Qué no está acá

Los pesos LoRA de los checkpoints (2,1 GB por corrida) no entran en un repositorio de git. Están en la
máquina de Alex con manifiestos sha256, y los originales quedaron en el volumen de la instancia
detenida. Todo lo demás, que es lo que se necesita para verificar los números, está en este directorio.

## Cómo leer un log de entrenamiento

```python
import json
R = [json.loads(l) for l in open("logs_entrenamiento/vel_run__stage2_equiv_vel_s42_...jsonl")
     if l.strip().startswith("{")]
pasos = [d for d in R if d.get("step") and "loss_diffusion" in d]
# campos utiles: loss_rotation, cos_ramas, escala_ramas, margen_piso, physics_skipped,
# grad_norm_physics / grad_norm_diffusion (la razon que se calibra), clip_id, n_accel
```
