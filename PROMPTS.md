# Los prompts, completos

Están todos acá porque son parte del resultado: en generación por texto lo único que entra es esta
línea, así que cualquier diferencia entre los dos brazos es atribuible al prompt y a la semilla. Salen
de `SCENARIO_PROMPTS` y `OOD_PROMPTS` de `evaluation/run_eval.py`, y van en inglés porque es lo que
recibe el codificador de texto de SANA.

## Los tres escenarios del dataset

Son los que el modelo vio en entrenamiento, y los que tienen ground truth.

| escenario | prompt |
|---|---|
| caída libre | `a solid colored ball in free fall, side view` |
| péndulo | `a solid colored ball swinging like a pendulum from a pivot, side view` |
| rebote | `a solid colored ball bouncing on a floor, side view` |

## Los dos prompts fuera del dataset

| nombre | prompt | en qué sentido está fuera |
|---|---|---|
| rodando | `a solid colored ball rolling along a flat surface from left to right, side view` | **escenario nuevo**: rodar no está en el simulador, el modelo nunca lo vio |
| péndulo en cámara lenta | `a solid colored ball swinging as a pendulum from a fixed pivot, side view, slow motion` | **el mismo escenario con otras palabras**: cambia la redacción y agrega `slow motion` |

Conviene no confundirlos. *Rodando* pide un movimiento que no está en los datos; el segundo pide el
péndulo de siempre reformulado, así que mide otra cosa, que es si el modelo sigue respondiendo al
prompt cuando no viene escrito igual que en entrenamiento. El aspecto fotorrealista que aparece en el
modelo base no lo pide el prompt: es lo que SANA trae de fábrica.

## Cómo se generó

| | |
|---|---|
| prompt negativo | vacío (`""`) en todas las generaciones |
| pasos de Euler en inferencia | 20 (`--n_infer`); en entrenamiento la pérdida usa 12 |
| resolución | 384 × 384 |
| cuadros | 33, los mismos del entrenamiento; 65 en los videos de extrapolación |
| semilla | fija por generación (`--semilla_base`), así que los dos brazos comparten ruido |

Los videos de cada prompt están en [`videos/4_ocho_semillas/`](videos/4_ocho_semillas), ocho semillas
por prompt, y las generaciones sueltas en
[`videos/8_generaciones_crudas/`](videos/8_generaciones_crudas).
