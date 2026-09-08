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
