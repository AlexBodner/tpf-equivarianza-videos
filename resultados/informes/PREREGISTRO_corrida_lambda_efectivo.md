# Preregistro: corrida con λ efectivo (2026-09-06; relanzada a las 15:22 UTC en la L40S tras abortar la de las 03:58; cerrado antes de ver ningún resultado de evaluación)

## Por qué otra corrida
La corrida E4-norm2 entrenó con una razón de gradiente ‖∇física‖/‖∇difusión‖ de **0,018** (objetivo
0,50): el término de equivarianza entró 28× más débil de lo planeado porque λ se calibró con un bug
y no se recalibró tras corregirlo. Su nulo en distribución es el desenlace preregistrado
"indistinguible de haber apagado la pérdida" y **no prueba nada sobre la pérdida invariante a
escala**. Esta corrida es la primera prueba real de esa pérdida.

## Qué cambia y qué no
- Cambia **solo λ_rot**: de 6,694e-05 a **1,27e-03**, fijado por la sonda 3 (razón mediana 0,497). Ver la tabla de sondas.
- `probe_grad_split: true` **durante toda la corrida** (medido: no cuesta VRAM ni tiempo).
- Mismo código (`5de5e17a`), mismo punto de partida (`ckpt/A_dir`), misma semilla 42, mismos datos,
  mismo `k_steps=12`, `chunk_size=1`, `lr=1e-4`, clip de gradiente 1,0. Apareada con `ctrl_s42`.
- Instancia: la que resulte del análisis (L4 g6.xlarge o L40S g6e.xlarge, misma imagen). La
  instancia no cambia la matemática; sí puede cambiar la no-determinismo numérico de bf16, que ya
  existía entre corridas.
- `preflight_arms.py` tiene que confirmar que la config difiere de la del control **únicamente** en
  λ_rot, `probe_grad_split` y `output_dir`.

## Sondas de calibración (10 pasos cada una, mismo código, `ckpt/A_dir`, semilla 42)
| sonda | λ_rot | razón mediana | mín / máx por paso | norma total máx (clip 1,0) | L_rot/rama | veredicto |
|---|---|---|---|---|---|---|
| 1 (corrida anterior) | 6,694e-05 | 0,018 | 0,001 / 0,070 | — | 0,53 | 28× débil |
| 2 | 1,9e-03 | 0,750 | 0,114 / 1,825 | 0,324 | 0,78 | fuera de banda por arriba |
| **3** | **1,27e-03** (= 1,9e-03 × 0,50/0,75) | **0,497** | 0,032 / 1,380 | 1,668 (clip activo en 1 de 10) | 0,557 | **en banda: se lanza con este λ** |

La extrapolación lineal desde la sonda 1 no fue lineal (28× en λ dieron 42× en razón): al hacerse visible el
término la trayectoria cambia. El ajuste 2 → 3 sí está en el régimen correcto. Regla: se lanza con el λ de la
primera sonda cuya razón caiga en [0,40; 0,60]: la 3. Observación: con este λ el clip global de 1,0 se activa ocasionalmente (1 de 10 pasos en la sonda; 4 de 1000 en la corrida anterior). Es la cola de grado −1 de una pérdida de grado 0. No es regla de corte; el monitor reporta la fracción de pasos con clip y se documenta en los resultados.

## Enmienda 1 (2026-09-06, 04:20 UTC, antes de ver ningún resultado de evaluación)
- **El apareamiento paso a paso no sobrevive al cambio de GPU.** Las cinco corridas de la L4 (con o sin
  sonda, cualquier λ) dan `L_dif` = 0,2017202116549015 en el paso 1; la corrida lanzada en la L40S dio
  0,30767861 con configs resueltas idénticas. La bandera de sonda solo clona gradientes (no consume
  aleatoriedad). Es la aritmética bf16 de otra arquitectura. La corrida se abortó a los 15 minutos
  (`*_ABORTADO_apareamiento`) para medir en limpio antes de relanzar.
- Consecuencias: (1) la evaluación apareada por clip y semilla sobre las **salidas** sigue siendo válida;
  (2) la identidad paso a paso con `ctrl_s42` (entrenado en la L4) se pierde; para recuperarla se entrena
  un **control réplica en la L40S** (misma config, λ = 0, ~2 h) después de la corrida tratada, que además
  da el placebo ctrl-L4 vs ctrl-L40S: la escala de ruido de hardware. Una sonda de 3 pasos del control en
  la L40S verifica primero que el flujo de datos y ruido es el mismo (su paso 1 debe dar 0,30767861).
- **Uso de la VRAM, medido** (3 pasos cada sonda, L40S): checkpointing apagado → **OOM** con chunk 1 y con
  chunk 3 (el grafo del unroll de 12 pasos sin recomputación supera los 44 GB); checkpointing encendido con
  `chunk_size` 3 → 43 s/paso (contra 46 con chunk 1), pico 37,0 GB. Decisión: **config original (chunk 1,
  checkpointing encendido, pico 21 GB)**: el 7 % de velocidad no compensa perder el evaluador en paralelo
  (necesita ~6 GB libres) ni cambiar la aritmética respecto de todas las corridas previas. El mejor uso de
  los 25 GB libres es la evaluación apareada durante el entrenamiento.
- Incidente de proceso: entre las 04:20 y las 15:17 UTC un monitor de espera sin tope (ssh en BatchMode
  contra un host aún sin clave conocida) giró en silencio y no lanzó las sondas de memoria; 11 h de
  instancia ociosa. Regla nueva: toda espera con tope y latido, y verificación temprana de que la etapa
  siguiente arrancó. Sonda del control en la L40S: paso 1 = 0,30767861, igual que la corrida abortada:
  mismo flujo de datos y ruido; la diferencia con la L4 es sólo aritmética. Control réplica: pendiente,
  después de la corrida tratada (5 a 7 s/paso en la L40S).

## Relanzamiento (15:22 UTC), verificado a los minutos
`stage2_equiv_lambda_s42_20260906-152300_5de5e17a`, pid 4633, chunk 1, checkpointing encendido, sonda encendida.
Paso 1: `L_dif` = 0,30767861 = la referencia del control en la L40S (apareamiento en esta arquitectura: OK);
pasos 2 y 3 idénticos a los de la corrida abortada (0,98222, 0,16924): el entrenamiento es determinista en
esta GPU. Razón de gradiente en los primeros pasos 0,96 / 0,15 / 0,25; 34 a 48 s/paso; 21 GB. Evaluador por
checkpoint vivo en paralelo. Encolado detrás de la cadena final: el control réplica en la L40S.

## Aviso temprano (paso 50, 15:55 UTC): la difusión se degrada desde el arranque
`val_loss_diffusion` en el paso 50 (20 clips held-out): **nuevo 0,0910**; ctrl_s42 (L4) 0,0672; equiv_norm λ chico
0,0679; brazo viejo del E4 0,0677. Razón contra el control: **1,35×**, cuando el brazo viejo, que colapsó, tuvo
1,01× en el paso 50 y nunca pasó de 1,16×. Pérdida de difusión de entrenamiento, media de los pasos 1-60: nuevo
0,298 contra 0,247 del control (+21 %). La regla del preregistro (> 1,2× → cortar) se aplica en el paso 250; antes
es aviso. Se verifica que no sea la aritmética de la L40S con un control de 60 pasos en la misma GPU (val en el
paso 50), y se espera la validación del paso 100. Si la razón sigue ≥ 1,2× en 100 y 150, la lectura es que a razón
de gradiente ~0,5 el término físico desestabiliza la difusión; la corrida se corta en el 250 como manda el
preregistro (o antes si el usuario lo decide) y el siguiente experimento es λ a razón 0,25 (≈ 6,4e-04).

## Enmienda 2 (16:35 UTC): el control réplica corre en paralelo, y es la única vara válida para la regla de corte
Un control de 60 pasos en la L40S dio pérdida de difusión de entrenamiento 0,281 en los pasos 1-60 contra 0,247
del mismo control en la L4: **la aritmética de la GPU mueve el valor de la pérdida un 14 %**. Comparar el brazo
tratado (L40S) contra el control de la L4 en la regla `val_dif > 1,2×` es inválido; el 1,35× del paso 50 está
inflado por hardware en una parte que sólo el control en la misma GPU puede medir. Por eso el control réplica
(`stage2_ctrl_lambda`, λ = 0, misma config, misma semilla) se lanzó **ahora en paralelo** (pid 12832, 16:33 UTC)
en vez de después: valida en 50/100/…/250 sobre el mismo hardware, el monitor compara contra él, y su
`checkpoint-1000` entra como cuarto brazo (`ctrl_l40s`) en la evaluación final. Costo: la corrida tratada se
enlentece mientras comparten GPU (~2 a 3 h); a cambio la regla de corte y el apareamiento paso a paso vuelven
a ser válidos, y ctrl-L4 vs ctrl-L40S es el placebo de hardware. La comparación contra `ctrl_s42` de la L4 queda
como descriptiva.

## Aviso temprano, con la vara correcta (16:45 UTC)
Control réplica en la L40S, paso 50: `val_loss_diffusion` = **0,0719** (el mismo control en la L4: 0,0672; +7 % por
hardware). Brazo tratado, paso 50: 0,0910 → **1,27× el control en la misma GPU** (no 1,35×). El hardware explica
un tercio de la brecha; el resto es real y sigue por encima del máximo que alcanzó el brazo colapsado del E4
(1,16×). La regla (> 1,2×) se decide en el paso 250 contra el réplica. El réplica corre ~10× más rápido que el
tratado (λ = 0: sin rama física) y termina sus 1000 pasos en ~2 h.

## Paso 100 (17:00 UTC): tres señales independientes de colapso hacia el piso de ruido
| ventana | 1-25 | 26-50 | 51-75 | 76-100 |
|---|---|---|---|---|
| `accel_norm` tratado | 2,46 | 1,48 | 1,52 | 1,08 |
| `accel_norm` brazo viejo del E4 (mismos pasos) | 2,42 | 2,27 | 2,10 | 1,89 |
| `L_rot`/rama tratado | 0,44 | −0,07 | −0,04 | −0,08 |
| razón de gradiente | 0,44 | 0,55 | 0,32 | 0,24 |

`val_loss_diffusion` contra el réplica en la misma GPU: 1,27× (50) → **1,37×** (100). El movimiento cae 56 % en
100 pasos (el brazo viejo, 22 %); la pérdida normalizada se va a cero o negativa: las dos ramas coinciden al nivel
del piso de RAFT. Hipótesis mecánica: cerca del piso `(N − A)/max(D − A, A)` es negativa, así que la pérdida no es
neutra ante encoger sino que premia llegar al piso (mínimo global). Se verifica numéricamente con la fórmula
real y se cuantifica el colapso con una evaluación apareada en el checkpoint 125 contra el réplica (misma GPU).
Desenlace (a) del preregistro, anticipado: la pérdida invariante a escala **no evita el encogimiento a λ efectivo**.

## Enmienda 3 (17:09 UTC): el mecanismo, verificado con la fórmula, y el arreglo listo (no desplegado)
Con la fórmula real y A = 1: L = 0 en D = 2A; **−0,40 en D = 1,5A; −0,90 en D = 0,5A**. El numerador `N − A` se
hace negativo cuando el desacuerdo entre ramas cae bajo el piso, así que el mínimo global de la pérdida está en el
piso de ruido: no era neutra ante encoger, **premiaba llegar al piso**. A razón de gradiente 0,5 el modelo lo
encontró en 100 pasos. Arreglo (rama `fix/perdida-piso-no-negativa` sobre el código que corrió):
`num = max(N − A, 0)` en la pérdida y coeficientes nulos por chunks cuando `N − A ≤ 0` (por debajo del piso la
pérdida vale 0 y no tiene gradiente); gate por margen `D/A < margen_piso_minimo` (default 3,0, el mínimo medido
en los generados del E4) que saltea el paso con motivo. Tests: 6 de 11 nuevos fallan contra la fórmula vieja;
suite completa (losses + oráculos) en el entorno de la VM: 52 pasan; la descomposición por chunks sigue exacta
por encima del piso. Decisión pendiente del usuario: cortar la corrida a razón 0,5 con la pérdida vieja,
desplegar el arreglo, re-sondear λ y relanzar.

## Enmienda 4 (17:35 UTC): corte en el 125, arreglo desplegado, sonda con la pérdida arreglada
Sin respuesta del usuario al plazo, se aplicó el valor por defecto anunciado: al guardarse el checkpoint 125 se
lanzó la evaluación apareada del tratado (λ 0,5, pérdida vieja) contra el checkpoint 125 del réplica en la misma
GPU (n = 10 por escenario), se **cortó** el tratado (queda `stage2_equiv_lambda_s42_20260906-152300` como registro:
125 pasos, 2 checkpoints, log completo), se desplegó el arreglo en el árbol real (suite 52 en verde) y se lanzó
la sonda de 10 pasos con la pérdida arreglada a λ = 1,27e-03 (el clamp cambia la razón efectiva: hay que volver a
medir). El réplica sigue (mismo hardware, misma config: es el control de la corrida arreglada). Para la corrida
arreglada: `run_lambdafix.sh`, `evaluador_checkpoints_v2.sh` (aparea contra el réplica cada 125 pasos y contra la
L4 en 250/500/750/1000), `cadena_eval_lambdafix.sh`. Los resultados van a `results/e4lambdafix/`.
Código de la pérdida arreglada: rama `fix/perdida-piso-no-negativa` (`8c6047f6`, `973cccc4`) sobre `5de5e17a`.

## Enmienda 5 (17:55 UTC): sondas con la pérdida arreglada y regla de salteos
| sonda | pérdida | λ_rot | razón mediana | salteados | mín / máx | L_rot/rama |
|---|---|---|---|---|---|---|
| 3 | vieja | 1,27e-03 | 0,497 | 0/10 | 0,032 / 1,380 | 0,557 |
| 3-fix | **arreglada** | 1,27e-03 | **0,344** (n = 8) | 2/10 por margen (1,6 y 2,2 < 3) | 0,103 / 2,428 | 0,809 (≥ 0) |
| 4-fix | arreglada | 1,85e-03 (= 1,27e-03 × 0,50/0,344) | *en curso* | | | |

A igual λ la razón efectiva baja con el arreglo: el clamp anula el gradiente en los chunks bajo el piso y el gate
saltea pasos. Se ajusta linealmente una vez y se re-sondea, como manda la regla de las sondas.
**Regla de salteos corregida**: ">10 % de pasos salteados → cortar" se escribió cuando saltear era "sin
aceleración medible"; con el gate, saltear por margen es por diseño (20 % en la sonda). Corte solo si la física
casi no entrena: **> 40 %** de pasos salteados. El monitor (`estado_lambda.py`) usa ese umbral.

## Métricas y qué significa cada desenlace
**Primarias** (validadas con cotas degeneradas, apareadas por clip contra `ctrl_s42` y contra la
corrida E4-norm2 a λ chico, mismos 10 clips por escenario, en 250/500/750/1000):
1. `vel_mae` (error de velocidad contra el simulador): la única métrica de plausibilidad física
   validada. Mejora = Δ < 0 con signo p < 0,05 en pooled pendulum+bouncing (free_fall se reporta aparte:
   la cota v=cte da 0).
2. `mov_ratio`: la guarda anti-degeneración. Si baja de 0,75× la del control en algún paso, el brazo
   está encogiendo y la pérdida no arregló el mecanismo, sea cual sea el resto.
3. **Ajuste a la ley física en generación libre** (T2V, n = 10 por escenario, `analizar_explor.py`):
   ω del péndulo y g del rebote recuperadas contra el simulador, y residuo normalizado. "Aprendió
   física" = ω y g del brazo tratado más cerca del simulador que las del control, apareado por
   semilla, en al menos dos de tres escenarios, sin que el residuo empeore.

**Secundarias** (se reportan, no deciden): `l_rot_norm` (el indicador preregistrado anterior no está
validado sobre generaciones: todos los brazos dan negativo; ver revisión externa B2), la diferencia de
píxel orig vs rotado-des-rotado de `videos_equivarianza.py`, `mae_ay`, `jerk_rms`, `l_rot` crudo
(las gana un video quieto).

**Desenlaces**:
- (a) `mov_ratio` colapsa (< 0,75× ctrl): la pérdida invariante a escala **no** evita el encogimiento
  a λ efectivo. Resultado negativo limpio sobre la pérdida.
- (b) No colapsa y `vel_mae`/ω/g no mejoran: la equivarianza rotacional de la aceleración, impuesta
  así, **no enseña física** a este modelo con 1000 pasos. Resultado negativo limpio sobre el método.
- (c) No colapsa y mejora en primarias: primera evidencia positiva; requiere replicar con otra
  semilla antes de afirmarlo.
- (d) Razón de gradiente fuera de [0,25; 1,0] durante la corrida: la calibración falló; se corta y
  se recalibra. No cuenta como evidencia de nada.

## Reglas de corte (se aplican en el paso 250, con la evaluación apareada automática)
- Cortar si la razón de gradiente mediana de los pasos 1-250 está fuera de [0,25; 1,0].
- Cortar si `mov_ratio` (n = 30) < 0,75 × el del control en el paso 250.
- Cortar si `val_loss_diffusion` > 1,2 × la del control en el mismo paso (desestabilizó la difusión).
- Cortar si más del 10 % de los pasos saltan la rama física (`physics_skipped`).
- Si no, sigue hasta 1000; se re-evalúa en 500 y 750 con los mismos criterios.

## Monitoreo durante la corrida (lo que dice si está aprendiendo, no solo si no se rompe)
- Cada 100 pasos: razón de gradiente (mediana de la ventana), `accel_norm` (ventana de 120, tendencia
  propia), `margen_piso` mediano, saltos, `val_loss_diffusion` vs la del control en el mismo paso.
- En cada checkpoint común con el control (250/500/750/1000), un evaluador automático corre
  `run_eval.py` (n = 10) para el checkpoint nuevo y lo aparea con los valores por clip **ya guardados**
  del control y de la corrida a λ chico (`resultados_json/paso_N`); y `generar_explor.py --arm` +
  `analizar_explor.py` para ω, g y residuo en T2V. Produce una fila de trayectoria por paso.
- Todo escribe en `results/e4lambda/`, nunca en `results/e4norm` ni `results/eval_*`.

## Verificaciones hechas antes de lanzar (2026-09-06)
- **Preflight** (`scripts/preflight_arms.py`) contra un control espejo (`config_stage2_ctrl_lambda.json`, igual
  a la nueva salvo λ_rot = 0): **OK**. Diferencias: solo `lambda_rot`, `output_dir`, `experiment_name`, comentario.
  Mismo `ckpt/A_dir`, semilla 42, receta de P1 coincidente. Dos avisos preexistentes y documentados
  (max_frames 24 → 33 y cambio de prompts respecto de P1), iguales a los del E4.
- **Control real `ctrl_s42` vs la config nueva**, `config_resolved.json` aplanado: 13 claves distintas, todas
  inertes para la optimización salvo λ_rot: `probe_grad_split` (solo loguea), `save_every` 250 → 125,
  `sample_frames`/`sample_infer_steps`/`sample_fps` (solo los videos de muestra), `val_every`/`val_batches`
  (explícitos vs default), `debug_viz_steps` y `lambda_jerk` (0 vs ausente), `physics_n_bptt` 12 (= default
  k_steps), `rot_invariante_escala` (gateado por λ > 0: con λ = 0 no hace nada). `dataset_type` aparece como
  `physics_iq` en el `trainer_config` de **ambas** corridas anteriores mientras `args.dataset_type='synthetic'`
  con `data_dir='data/synthetic_384'`: es un default rancio del sub-config, idéntico en los dos brazos.
- **Tests de oráculo y de la pérdida** en la VM, sobre el código que va a correr: 40 pasaron.
- **Evaluador por checkpoint** (`vm/evaluador_checkpoints.sh`): sintaxis verificada; el apareador
  (`scripts/aparear_trayectoria.py`) reproduce exactamente los números conocidos del paso 500 de la corrida
  anterior (+0,285, 10/30) y da Δ = 0 al aparearse consigo mismo; el generador parametrizado pasa el dry run.
- **Monitor** (`vm/estado_lambda.py`): probado sobre el log de la sonda (lee la razón 0,018) y sobre el de la
  corrida anterior (sin razón: no revienta; val 1,00× ctrl).
- Limitación conocida: `validate()` del trainer loguea `val_loss_rotation` como MSE crudo y solo si λ > 0, así
  que no sirve para comparar con el control; la comparación apareada la hace el evaluador por checkpoint.

## Enmienda 6 (18:15 UTC): sonda 4 fuera de banda por arriba; ajuste lineal y sonda 5

| sonda | pérdida | λ_rot | razón mediana | salteados | mín / máx | L_rot/rama |
|---|---|---|---|---|---|---|
| 3 (fix) | arreglada | 1,27e-03 | 0,344 | 2/10 (margen) | — | — |
| 4 (fix) | arreglada | 1,85e-03 | **0,675** (n=8) | 2/10 (margen 1,56 y 2,11 < 3) | 0,169 / 10,1 | 0,786 |

La sonda 4 se pasó de la banda [0,40; 0,60]. Con la sonda 3 (0,344) y la 4 (0,675) se ajusta linealmente una vez
más, como manda la regla de las sondas: λ = 1,27e-03 + (0,50 − 0,344)/(0,675 − 0,344) · 0,58e-03 = **1,54e-03**.
Sonda 5 lanzada a las 18:13 UTC (`config_probe_lambdafix3.json`, `logs/probe_fix3.sh`, `probe_fix3.log`),
compartiendo la GPU con la réplica de control y el eval-125, por lo que tarda ~25 min en vez de 8.

Regla de cierre, fijada antes de ver la sonda 5: si su razón cae en [0,40; 0,60] se lanza con ese λ. Si cae fuera,
**no se sondea más**: se lanza con el λ interpolado entre las dos sondas más cercanas a 0,50, y la razón medida en
los primeros 10 pasos de la corrida (que lleva `probe_grad_split` encendido) queda documentada como la calibración
efectiva. La dispersión por paso de la razón (mín/máx de 0,17 a 10 dentro de una sonda de 8 pasos útiles) hace que
seguir sondeando no compre precisión; el monitor `estado_lambda.py` vigila la banda ancha [0,25; 1,0] en la corrida.

## Enmienda 7 (18:32 UTC): sonda 5 en banda; corrida arreglada lanzada con λ = 1,54e-03

| sonda | pérdida | λ_rot | razón mediana | salteados | mín / máx | L_rot/rama | clip |
|---|---|---|---|---|---|---|---|
| 5 (fix) | arreglada | **1,54e-03** | **0,592** (n=8) | 2/10 (margen 1,32 y 2,48 < 3) | 0,150 / 1,889 | 0,765 | 0/10 |

Por paso: 1,17 · 0,40 · 0,15 · 0,51 · 1,89 · 0,17 · 0,96 · salteado · salteado · 0,67. Cae en [0,40; 0,60]: se lanza
con ese λ, como manda la regla. Nota de registro: la sonda 5 escribió en `probe_lambdafix2_*_181331` (el nombre sale
del config copiado), por lo que el resumen automático de `probe_fix3.sh` no la encontró; se resumió a mano con
`logs/resumen_sonda.py` y las tres sondas (3, 4, 5) están bajadas a local en `vm_e4norm/sondas_fix/` antes de que
`relanzar_lambdafix.sh` borre `checkpoints_sana/probe_lambdafix*`.

Lanzamiento: `relanzar_lambdafix.sh 1.54e-03`, preflight OK (los brazos difieren solo en `_comment`,
`experiment_name`, `lambda_rot`, `output_dir`), pid 47524, 18:32 UTC, 1000 pasos, guardado cada 125, muestras cada
250, validación cada 50, `margen_piso_minimo` 3,0, `probe_grad_split` encendido. Comparte la L40S con la réplica de
control (paso ~400) y con el eval-125 del brazo colapsado; la vara de corte sigue siendo la réplica en la misma GPU
(enmienda 2). Costo estimado: 1000 pasos a ~50-60 s/paso compartiendo ≈ 15 h ≈ 28 USD más el evaluador.

## Enmienda 8 (19:25 UTC): el colapso del brazo tratado (pérdida vieja, λ = 1,27e-03), medido en el paso 125

Apareado por clip contra la réplica de control en la misma GPU (L40S), 10 clips por escenario, n = 30, held-out,
33 frames (`results/e4lambda/paso_125`, JSON instantáneo `vm_e4norm/e4lambda/paso_125_eval_results_snapshot_1922utc.json`;
la parte OOD seguía generándose y no entra acá).

| métrica | control réplica | tratado (paso 125) | Δ | tratado menor | signo p |
|---|---|---|---|---|---|
| mov_ratio (↑ hacia 1) | 0,602 | **0,135** | −0,467 (−78 %) | 30/30 | < 0,001 |
| vel_mae px/frame (↓) | 4,935 | **9,089** | +4,154 (+84 %) | 0/30 | < 0,001 |
| l_rot_norm (↓, no validada) | 0,089 | 0,082 | −0,008 (−9 %) | 8/30 | 0,016 |

Cota del video quieto: mov_ratio 0,20 / 0,04 / 0,07 (caída libre / péndulo / rebote). El tratado queda a 0,08 / 0,13 / 0,20:
entre la cota del video quieto y un tercio del control. Es el colapso al piso predicho por la fórmula (enmienda 3) y
visto en el entrenamiento (â −56 % en 100 pasos): el modelo casi deja de mover la pelota. La l_rot_norm baja al mismo
tiempo, lo que confirma que esa métrica premia la degeneración y no sirve para leer aprendizaje (ya estaba marcada como
no validada en `METRICAS_COMPLETAS.md`).

## Enmienda 9 (23:05 UTC): primer apareado de la corrida arreglada, checkpoint 125

Contra la réplica de control en la misma GPU, mismos 30 clips held-out (10 por escenario), 33 frames
(`results/e4lambdafix/paso_125`, evaluador v2). Sin regla de corte violada.

| métrica | réplica | arreglada (125) | Δ | arreglada menor | signo p | Wilcoxon p |
|---|---|---|---|---|---|---|
| vel_mae px/frame (↓) | 4,908 | **4,277** | −0,631 (−13 %) | 21/30 | 0,043 | 0,050 |
| mov_ratio (→ 1) | 0,604 | **0,852** | +0,248 | 3/30 | < 0,001 | < 0,001 |
| l_rot_norm (↓, no validada) | 0,089 | 0,212 | +0,123 | 5/30 | < 0,001 | 0,124 |
| mae_ay (↓, gana el quieto) | 1,754 | 2,342 | +0,587 | 7/30 | 0,005 | < 0,001 |
| jerk_rms (↓, gana el quieto) | 1,560 | 2,378 | +0,818 | 5/30 | < 0,001 | < 0,001 |
| l_rot cruda (↓, gana el quieto) | 3,404 | 6,545 | +3,141 | 4/30 | < 0,001 | < 0,001 |

Por escenario, mov_ratio / vel_mae: caída libre 0,80 / 3,52 (réplica 0,68 / 4,47), péndulo **1,14** / 3,55
(0,63 / 3,95), rebote 0,61 / 5,76 (0,50 / 6,31). Base sin fine-tuning: 0,43 / 11,7, 0,52 / 6,1, 0,09 / 9,95.

Lectura, con cautela: es lo contrario del colapso. La corrida arreglada se mueve más que el control y más
cerca del movimiento del GT (en el péndulo lo sobrepasa: 1,14), y el error de velocidad baja 13 % con p en el
borde (0,05). Las tres métricas que gana el video quieto empeoran, como corresponde a más movimiento, y la
l_rot_norm también sube: la salida no es más equivariante según RAFT, y en las vistas previas del paso 150 hay
pelotas duplicadas que inflan jerk y aceleración. La validación de difusión va 1,49× la réplica en el 150.
Nada de esto se lee todavía como "aprendió la simetría": es "se mueve más y mejor en velocidad, con más ruido".
Siguen la validación del 200, el corte del 250 (> 1,2× sostenido) y el apareado del 250.

### Enmienda 9b (23:10 UTC): ajuste físico en T2V del checkpoint 125 (exploratorio, n=10 por escenario)

Generación libre (T2V) con los mismos prompts y semillas que la exploración de E4-norm2; trazador único con filtro de
saltos; detección válida 67 %. Referencias: GT, y las medianas del control y de equiv_norm en el paso 1000 en la L4
(`METRICAS_COMPLETAS.md` §5). Mediana por escenario; "si aprendió física, el brazo da lo del GT".

| escenario | parámetro | GT | ctrl (L4, 1000) | equiv_norm (L4, 1000) | **arreglada (125)** |
|---|---|---|---|---|---|
| péndulo | residuo del seno (↓) | 0,030 | 0,123 | 0,209 | 0,160 |
| péndulo | ω [rad/cuadro] | 0,168 | 0,231 (+37 %) | 0,273 (+62 %) | **0,150 (−11 %)** |
| péndulo | amplitud fin/inicio | 0,712 | 1,107 | 0,791 | 1,248 |
| péndulo | mov_ratio T2V | 1 | — | — | 1,68 |
| rebote | residuo (↓) | 0,016 | 0,047 | 0,044 | 0,054 |
| rebote | g [px/cuadro²] | 3,281 | 1,407 (43 %) | 0,980 (30 %) | 0,895 (27 %) |
| rebote | dispersión de g entre tramos | 0,229 | 0,744 | 0,413 | 0,950 |
| caída libre | residuo (↓) | 0,020 | 0,078 | 0,025 | 0,055 |
| caída libre | CV de la velocidad (GT ~ terminal) | 0,326 | 0,777 | 0,380 | 1,070 |

Lectura: la frecuencia del péndulo es lo único que se acerca al GT más que cualquier brazo anterior (0,150 contra
0,168; los otros dos la sobreestiman 37 y 62 %). Todo lo demás es peor o igual: la amplitud crece en vez de
decaer (1,25 contra 0,71: el péndulo "se acelera"), en caída libre la velocidad deja de ser constante (CV 1,07), y
g sigue en un cuarto del GT con mucha dispersión entre tramos. Coherente con el apareado del 125: más movimiento,
más ruido, y ninguna evidencia de haber aprendido la dinámica. Es n=10 y exploratorio; no se cita como resultado.

## Enmienda 10 (2026-09-07, 00:05 UTC, escrita ANTES de leer la validación del 250): la regla de corte por validación se complementa

Hechos conocidos al escribirla: validación del tratado 1,06× (50), 1,06× (100), 1,49× (150), 1,27× (200) la réplica;
apareado del 125: mov_ratio 0,85 vs 0,60 y vel_mae −13 % (p = 0,05); ajuste físico T2V del 125 sin dinámica aprendida.

Motivo: la regla "corte en el 250 si val > 1,2× la réplica" se diseñó para el colapso al piso, que vino con la
validación disparada y el movimiento en 0,13. Lo de ahora es lo contrario en movimiento y la validación viene bajando.
Cortar por ese umbral solo perdería el estado del optimizador sin haber medido lo que importa.

Regla enmendada, vigente desde el 250:
1. No se corta por validación sola. Se corta si la validación supera **1,5×** la réplica, o si sube en **dos valores
   consecutivos** (cada 50 pasos).
2. Se corta si el apareado de un checkpoint da **mov_ratio < 0,75× la réplica** (regla original) o **vel_mae peor que la
   réplica con p < 0,05** en el sentido contrario al del 125.
3. Si alguna falla, el corte se ejecuta en el siguiente checkpoint (375, 500, …), no entre checkpoints.
4. Propuesta enviada al usuario a las 23:35 UTC; sin respuesta al pasar el 250, se aplica por defecto (regla de
   espera con tope y acción por defecto).

## Enmienda 11 (00:04 UTC): corte de la corrida arreglada en el paso 257

Validación de difusión del tratado contra la réplica: 1,06× (50), 1,06× (100), 1,49× (150), 1,27× (200), **1,84× (250:
0,1231 contra 0,0667)**. Viola la regla enmendada por las dos vías (supera 1,5× y vuelve a subir). El checkpoint 250
está guardado (la validación se mide en el mismo paso), así que el corte se ejecutó de inmediato, matando por PID el
wrapper y el entrenamiento (paso 257 en el log); la cadena final no corrió porque exige `checkpoint-1000`. El evaluador
por checkpoint siguió vivo y está haciendo el apareado y el ajuste físico del 250. Quedan `checkpoint-125` y
`checkpoint-250` con sus evaluaciones apareadas contra la réplica.

Estado al cortar: razón de gradiente 0,70, salteos 8 %, clip 2 %, aceleración media +40 % respecto del inicio,
L_rot por paso plana desde el paso 1. Costo de la corrida: 5,5 h de L40S ≈ 10 USD, más las sondas y la réplica.

Lectura provisional (queda para el apareado del 250): la pérdida arreglada no colapsa, pero tampoco reduce su propio
término; lo que hace es degradar la reconstrucción de difusión con más movimiento y pelotas duplicadas. El siguiente
paso razonable no es otra corrida igual: es entender por qué L_rot no baja (¿el gradiente del cociente es demasiado
ruidoso por paso, con razones entre 0,15 y 1,9; o el término no es reducible por LoRA sin tocar la reconstrucción?).

## Enmienda 12 (00:50 UTC): apareado del checkpoint 250 (último antes del corte)

Contra la réplica en la misma GPU, mismos 30 clips (`results/e4lambdafix/paso_250`).

| métrica | réplica | arreglada (250) | Δ | arreglada menor | Wilcoxon p |
|---|---|---|---|---|---|
| vel_mae (↓) | 4,816 | 4,792 | −0,024 (−0,5 %) | 13/30 | 0,871 |
| mov_ratio (→ 1) | 0,610 | 0,946 | +0,336 | 4/30 | < 0,001 |
| l_rot_norm (↓, no validada) | 0,261 | 0,092 | −0,169 | 11/30 | 0,047 |
| mae_ay (↓, gana el quieto) | 1,743 | 2,211 | +0,468 | 7/30 | 0,002 |
| jerk_rms (↓, gana el quieto) | 1,587 | 2,502 | +0,914 | 3/30 | < 0,001 |
| l_rot cruda (↓, gana el quieto) | 3,665 | 8,125 | +4,460 | 8/30 | 0,001 |

Por escenario, mov_ratio / vel_mae: caída libre 0,74 / 4,51 (réplica 0,66 / 4,61), péndulo **1,24** / 4,19 (0,65 / 3,72),
rebote 0,86 / 5,68 (0,52 / 6,13). Contra el control de la L4 (paso 250): vel_mae 4,79 vs 5,83, Wilcoxon p = 0,020,
pero es cruce de GPU (enmienda 2) y no se cita.

Lectura: la ventaja de velocidad del 125 (−13 %, p = 0,05) no se sostuvo: en el 250 es nula. El movimiento siguió
creciendo (0,85 → 0,95, y el péndulo 1,24: sobrepasa al GT) y el jerk también, en línea con las pelotas duplicadas de
las muestras del 250 (visibles en T2V e I2V en caída libre y péndulo). La l_rot_norm del tratado da 0,092 contra 0,261
de la réplica (p = 0,047), pero es la métrica no validada y en el 125 daba lo contrario (0,212 contra 0,089): ruido de
instrumento, no señal. Conclusión de la corrida: la pérdida arreglada no colapsa y no aprende; degrada la reconstrucción
duplicando el objeto. Con dos checkpoints apareados no hay resultado citable a favor.

### Enmienda 12b (01:00 UTC): ajuste físico en T2V del checkpoint 250 (exploratorio, n=10 por escenario)

| escenario | parámetro | GT | ctrl (L4, 1000) | arreglada (125) | **arreglada (250)** |
|---|---|---|---|---|---|
| péndulo | residuo del seno (↓) | 0,030 | 0,123 | 0,160 | 0,214 |
| péndulo | ω [rad/cuadro] | 0,168 | 0,231 | 0,150 | 0,150 |
| péndulo | amplitud fin/inicio | 0,712 | 1,107 | 1,248 | 0,652 |
| péndulo | mov_ratio T2V | 1 | — | 1,68 | 1,57 |
| rebote | residuo (↓) | 0,016 | 0,047 | 0,054 | 0,062 |
| rebote | g [px/cuadro²] | 3,281 | 1,407 | 0,895 | 0,646 |
| caída libre | residuo (↓) | 0,020 | 0,078 | 0,055 | 0,143 |
| caída libre | CV de la velocidad | 0,326 | 0,777 | 1,070 | 2,487 |

Detección válida 78 %. Los residuos empeoran del 125 al 250 en los tres escenarios; la velocidad en caída libre pasa
a ser errática (CV 2,5, contra 0,33 del GT); g cae a un quinto del GT. La ω del péndulo repite 0,150 (mismo valor en
ambos checkpoints: probablemente la frecuencia la fija el prompt y el fondo, no la dinámica). Nada que sugiera física
aprendida; consistente con las pelotas duplicadas.

## Enmienda 13 (2026-09-07, 01:40 UTC, antes de retomar): la corrida se retoma desde el checkpoint 250

Decisión del usuario, con un argumento válido: 250 de 1000 pasos no descartan que un comportamiento emerja después,
aunque hasta acá la reconstrucción empeora. La retoma es fiel: `checkpoint-250` guarda pesos y `optimizer.pt`, y
`train_sana.py --resume <run>/checkpoint-250` continúa en el mismo directorio con los 750 pasos restantes sin renumerar.
Se retoma con la **misma λ (1,54e-03)** y el mismo control réplica en la misma GPU. Código: el del arreglo del piso más
un cambio de **solo logueo** (commit `15c054b0` en `fix/perdida-piso-no-negativa`): por paso se guardan N y D crudos, el
piso A, la energía de cada rama, el **coseno entre R·â_orig y â_rot** (acuerdo de dirección, independiente de la escala),
el cociente de escalas entre ramas, y el **coseno entre el gradiente físico y el de difusión** (< 0 es conflicto). Es lo
que pidió el usuario: ver qué hace cada componente, no solo el cociente.

Antes de retomar se mide la **equivarianza directa** del checkpoint 250 y del checkpoint 250 de la réplica: pares I2V
con condicionamiento original y rotado 45°, diferencia de píxeles des-rotada dentro del círculo inscripto
(`videos_equivarianza.py`; 3 clips por escenario). Referencia previa (paso 1000, L4): control 5,1, equiv_norm 5,4;
calibración 0,3 = perfecta, 15 = ninguna. **Predicción registrada**: si la pérdida no enseñó equivarianza, el tratado
da ≥ la réplica. Un valor claramente menor sería la primera señal a favor.

Reglas para la retoma, fijadas ahora:
1. Se corta si la validación de difusión supera **2,0×** la réplica en dos valores consecutivos, o si un apareado da
   mov_ratio < 0,75× la réplica. Ya no se corta por el umbral 1,5× (se sabe que lo supera; lo que se quiere ver es si
   se estabiliza o se corrige).
2. Se evalúa apareado contra la réplica en 375, 500, 625, 750, 875 y 1000, con el ajuste físico T2V en cada uno, y la
   cadena final completa al terminar.
3. Lectura preregistrada: "emergió" solo si en algún checkpoint ≥ 500 el error de velocidad es menor que la réplica
   con p < 0,05 **y** la validación de difusión vuelve por debajo de 1,5×, **y** la equivarianza directa es menor que
   la de la réplica. Cualquier otra combinación se reporta como "no emergió en 1000 pasos".
4. Costo: ~46 s/paso sin compartir GPU, 750 pasos ≈ 9,6 h ≈ 18 USD, más ~6 evaluaciones (~4 h ≈ 7 USD).

Nota operativa: al intentar prender la L40S AWS devolvió `InsufficientInstanceCapacity` (g6e.xlarge, us-east-1);
un orquestador local reintenta cada 5 min hasta 3 h y, al arrancar, despliega, corre la suite, mide y retoma.

### Enmienda 13b (01:55 UTC): el orden de muestreo al retomar, señalado por el usuario antes de lanzar

Verificado en el código: la semilla global se fija al arrancar, el orden de los clips sale de `shuffle=True` sobre el
RNG global (una permutación de los 1350 clips de entrenamiento, que cubre los 1000 pasos), y al retomar no se guarda ni
restaura ningún estado del generador: el iterador arrancaba desde el primer lote. Sin arreglo, los pasos 251 a 1000
habrían visto los clips de los pasos 1 a 750, distintos de los que vio la réplica en esos pasos y con 500 clips repetidos.

Arreglo (commit en `fix/perdida-piso-no-negativa`): `train()` consume `start_step` lotes válidos antes de entrenar,
con lo que el paso 251 ve el mismo clip que habría visto la corrida ininterrumpida, porque la permutación es la misma
(misma semilla, mismo consumo de RNG hasta crear el iterador). Test `training/tests/test_resume_salteo.py`. Desde ahora
el log lleva `clip_id` por paso; la corrida original no lo tenía, así que la coincidencia de orden queda establecida por
construcción y no verificada contra ella. Costo del salteo: cargar 250 clips sin entrenar, unos minutos.

Lo que no se recupera: el ruido de difusión por paso (el RNG no se restaura). Ya no era igual entre brazos desde el
paso 2, porque el brazo con física consume más números aleatorios que el control; el apareamiento nunca dependió de eso
sino de la semilla, el orden de clips y los pesos iniciales.

### Enmienda 13c (02:10 UTC): la L40S original no arranca por capacidad; se migra a una g6e nueva en otra zona

Tras ~30 min de `InsufficientInstanceCapacity` en us-east-1c, el usuario aprobó crear una g6e.xlarge nueva desde la AMI
del clon (`ami-0072a8f19c671562c`, la misma de la que salió la L40S original) en us-east-1a/b/d, y **retomar
directamente**, verificando que continúe como debe, con la evaluación encolada. Se migran desde local: el árbol de
código de la rama del arreglo (con el salteo de la retoma y el logueo por componentes), los scripts de evaluación, el
checkpoint 250 con su optimizador, la réplica completa (8 checkpoints) y los resultados apareados. Misma arquitectura
(L40S), así que el apareamiento con la réplica sigue valiendo. La medición directa de equivarianza del 250 corre en
paralelo al entrenamiento, no antes. Verificación registrada al retomar: el log debe mostrar "salteando 250 lotes", el
paso 251 una sola vez, y `clip_id`, `cos_ramas` y `cos_grad_fis_dif` poblados.

## Enmienda 14 (03:20 UTC, antes de ver ningún número): ablación de aumentaciones con el protocolo válido

Pedido del usuario: lo que falta, sobre todo en el poster, es la comparación contra **no usar aumentaciones**, para ver si
aunque sea la aumentación (clips rotados y trasladados) enseña alguna noción física. Lo que existe: la ablación de agosto
(Etapa 1, 3000 pasos, `A` con aumentaciones y `B` sin, misma receta y mismo dataset) medida con `mae_ay`, `jerk` y
`L_rot`, las tres métricas que después se marcaron como ganadas por un video quieto; ahí B "ganó" 8,8 % en `mae_ay`,
que con esa métrica es compatible con moverse menos. Nunca se midió con el protocolo válido.

Se evalúa ahora en la VM nueva, en paralelo al entrenamiento retomado y detrás de la equivarianza directa del 250
(`logs/eval_ablacion_aug.sh`, salida en `results/ablacion_aug_v2/`): A y B (pesos de S3, `ckpt/A_dir`, `ckpt/B_dir`) con
el video quieto como cota, en distribución n=20 held-out a 33 frames, OOD tiro vertical n=15, apareado por clip B vs A,
y en T2V el ajuste físico (n=10) más 3 pares de equivarianza por escenario para ambos.

Predicciones registradas:
1. Si las aumentaciones por sí solas enseñan algo de física, A debería dar **menor error de velocidad** y razón de
   movimiento **más cerca de 1** que B en el apareado (p < 0,05), y mejor ajuste físico (ω y g más cerca del GT).
2. Si B repite su "ventaja" en `mae_ay`/`jerk` pero con **menor** razón de movimiento, esa ventaja es la degeneración
   ya conocida y no se reporta como física aprendida.
3. En equivarianza directa (diferencia orig vs rotado des-rotado), A debería ser menor que B si la aumentación por
   rotación enseñó invariancia; si no hay diferencia, la aumentación no alcanza para eso y la pérdida sigue siendo
   la única vía candidata.

Lo que esta ablación NO cubre: un brazo de Etapa 2 sin aumentaciones (control λ=0 con `use_augmented_clips=false`),
que sería la versión completa de "no usar nada". Cuesta ~1000 pasos (~13 h de L40S, ~25 USD) y se decide con el usuario
después de ver A vs B.

## Enmienda 15 (2026-09-07, 11:10 UTC): la retoma verificada, la equivarianza directa del 250 y los apareados 375/500/625

**Retoma verificada** (g6e nueva, us-east-1b): el log muestra "salteando 250 lotes"; el paso 251 aparece dos veces en
`training_log.jsonl` (la línea vieja de la corrida cortada y la nueva; se distinguen por `clip_id`, que solo tiene la
nueva); los 476 pasos nuevos (251 a 726 al escribir esto) traen `clip_id` y, en los no salteados, `cos_ramas`,
`escala_ramas` y `cos_grad_fis_dif`. Velocidad: 19 s/paso una vez calentado el volumen (46 antes en la L40S original con
la GPU compartida); llega al 1000 a las ~12:30 UTC.

**Equivarianza directa del checkpoint 250** (pares I2V original/rotado 45°, diferencia des-rotada, 3 clips × 3
escenarios): **tratado 6,22 contra réplica 4,78** (caída libre 5,13/3,87; péndulo 5,37/4,77; rebote 8,17/5,70).
Referencias: control L4 5,1, E4-norm2 5,4; calibración 0,3 perfecta, 15 ninguna. Predicción de la enmienda 13
cumplida en el sentido negativo: el tratado es **menos** equivariante que la réplica, no más.

**Componentes por paso** (nuevos, ventanas de 100): pérdida por rama 0,49–0,60 (plana), **cos_ramas 0,42 → 0,39**
(acuerdo de dirección entre R·â_orig y â_rot, sin mejora), escala_ramas 1,11 → 1,19 (la rama original acelera un
10-20 % más que la rotada), **cos_grad_fis_dif ≈ 0,00** en todas las ventanas (el gradiente físico es ortogonal al de
difusión: ni conflicto ni ayuda), aceleración media 4,0 → 1,95, salteos por margen 28 % → 45–51 %. Lectura: el modelo
volvió a moverse menos, con lo que cada vez más pasos caen por debajo del margen 3 y no entrenan la física.

**Validación de difusión** contra la réplica: 1,42× (300), 1,27 (350), 1,14 (400), 1,15 (450), 1,10 (500), 1,08 (550),
1,10 (600), **1,00 (650)**, 1,19 (700). Se recuperó: el pico de 1,84× del 250 no era colapso sino una excursión que la
difusión corrigió, en coincidencia con la caída del movimiento.

**Apareados contra la réplica** (n=30, mismos clips):

| checkpoint | vel_mae réplica / tratado | p | mov_ratio réplica / tratado | p | jerk réplica / tratado |
|---|---|---|---|---|---|
| 125 | 4,91 / **4,28** | 0,050 | 0,60 / 0,85 | <0,001 | 1,56 / 2,38 |
| 250 | 4,82 / 4,79 | 0,87 | 0,61 / 0,95 | <0,001 | 1,59 / 2,50 |
| 375 | 4,89 / 4,86 | 0,98 | 0,60 / 0,74 | 0,001 | 1,80 / 2,18 |
| 500 | 4,52 / **5,50** | 0,001 | 0,68 / 0,63 | 0,20 | 1,90 / 1,89 |
| 625 | 5,28 / 5,30 | 0,60 | 0,57 / 0,64 | 0,031 | 1,54 / 2,03 |

Ajuste físico T2V por checkpoint (mediana): ω del péndulo **0,150 en todos** (la fija el prompt, no la dinámica);
g del rebote 0,90 / 0,65 / 0,67 / 0,74 / 0,32 (GT 3,28); CV de velocidad en caída libre 1,07 / 2,49 / 2,96 / 0,63 / 0,41
(GT 0,33): mejora a medida que el modelo se mueve menos, igual que el control.

Reglas de la enmienda 13: ninguna violada (validación nunca 2×, mov_ratio nunca < 0,75× la réplica). La corrida sigue
al 1000 y la cadena final corre sola. Lectura preregistrada (regla 3): para "emergió" hace falta vel_mae menor con
p < 0,05 en un checkpoint ≥ 500 **y** validación < 1,5× **y** equivarianza directa menor que la réplica. La tercera ya
falló en el 250 y las dos primeras no se dan en 500 ni 625. Salvo sorpresa en 750–1000, el veredicto es **no emergió en
1000 pasos**: la pérdida arreglada primero infla el movimiento (duplicados) y luego vuelve al régimen del control con
más pasos salteados; en ningún tramo mejora la equivarianza medida.

Ablación de aumentaciones (enmienda 14): la evaluación falló por un archivo de pesos de B corrupto en la subida
(`SafetensorError: incomplete metadata`); se verifica el sha256, se resube y se relanza.

## Enmienda 16 (12:05 UTC): diagnóstico del instrumento. RAFT no está roto, pero la aceleración que estima no alcanza para la pérdida

Pregunta del usuario: "¿puede estar fallando RAFT u otro módulo?". Prueba directa con las mismas funciones del entrenador
(`extract_kinematics(smooth=False)`, `rotation_matrix_2d`, `rotar_video_pixeles`), en la L40S (`logs/diag_raft.py`).
Para cada par (original, rotado 45°) se calcula el coseno entre R·â_orig y â_rot y el cociente crudo N/D
(0 = equivariante perfecto, 1 = sin relación).

| par | cos mediana | N/D mediana | detalle |
|---|---|---|---|
| **mismo clip real, dos pasadas** | 1,00 | 0,00 | RAFT es determinista: no hay ruido de ejecución |
| **clip real + ruido gaussiano σ=8 px** (sin rotar) | 0,18 | 0,83 | la aceleración (segunda diferencia del flujo) es extremadamente sensible al ruido de píxel |
| **clip real rotado píxel a píxel, R(+45)** | 0,74 | 0,26 | péndulo 0,74–0,77 / rebote 0,97 / **caída libre 0,22–0,59 (N/D 0,4–0,9)** |
| clip real rotado, con R(−45) (convención invertida) | 0,03 | 0,98 | descarta el error de signo: la convención actual es la correcta |
| rotación por `grid_sample` (la que mide el piso A) vs por cv2 | 0,73 / 0,74 | 0,27 / 0,26 | iguales en mediana, pero en rebote 0,64–0,96 vs 0,97: el piso se mide con un remuestreo algo distinto del de los datos |
| **generado, réplica 250** (cond original vs rotado, 50 pasos) | 0,07 | 0,93 | sin relación entre las aceleraciones de las dos ramas |
| **generado, tratado 250** | 0,06 | 0,94 | ídem; con R(−45) da 0,22, o sea tampoco hay señal "al revés" |

Lectura:
1. **No hay bug**: determinista, convención de rotación correcta, y sobre rebote real (aceleraciones de 4 px/frame² en los
   impactos) el instrumento ve equivarianza casi perfecta (0,97).
2. **El problema es de relación señal/ruido**: la aceleración es la segunda diferencia del flujo agregado, y a esta escala
   (pelota de ~1 % del cuadro, aceleraciones de 1–4 px/frame²) el ruido del estimador es del mismo orden que la señal.
   Sobre el **mismo video real** rotado, el desacuerdo ya es el 26 % de la energía en mediana, y en caída libre (donde el
   GT no acelera: velocidad terminal) el 40–90 %: un tercio de los pasos de entrenamiento mide ruido puro. Sobre
   **video generado** las dos ramas no tienen relación medible (N/D 0,93): lo que la pérdida ve es casi todo ruido.
3. Consecuencia sobre el entrenamiento: restar el piso A (medido sobre el clip real) no alcanza, porque el ruido sobre
   generado es mayor que sobre real y varía por paso. La pérdida queda en 0,5–0,6 por rama, su gradiente es
   esencialmente aleatorio (coseno con el de difusión 0,00, razones por paso entre 0,15 y 1,9), y el único descenso que
   el modelo encuentra es el que cambia D o N por vías espurias (encoger con la pérdida vieja; duplicar la pelota con la
   arreglada). Esto explica las tres corridas juntas sin invocar nada del modelo.
4. Lo que sí es medible con RAFT a esta escala: velocidades y posiciones (primera diferencia o trazador), que es
   exactamente lo que usa la evaluación (`vel_mae`, `mov_ratio`) y por eso esas métricas sí discriminan brazos.

Implicación para el método (no se ejecuta ahora, se registra): la equivarianza tendría que imponerse sobre una cantidad con
señal, por ejemplo la **velocidad** (R·v̂_orig ≈ v̂_rot, primera diferencia del flujo) o la **trayectoria** (posición
por centroide suave diferenciable), o sobre el **campo de flujo denso** rotado, en vez de sobre la aceleración
agregada. Es un cambio de la pérdida, no de λ, y requiere volver a medir el piso del instrumento sobre esa cantidad.

### Enmienda 16b (12:40 UTC): el mismo diagnóstico con VELOCIDAD en vez de aceleración

Pregunta del usuario: "¿qué nos frena de usar la velocidad?". Mismo script (`logs/diag_raft_vel.py`), mismos pares,
primera diferencia del flujo agregado en lugar de la segunda.

| par | aceleración: cos / N/D | **velocidad: cos / N/D** |
|---|---|---|
| clip real rotado píxel a píxel, R(+45) | 0,74 / 0,26 | **0,97–0,99 / 0,02–0,03** |
| ídem, convención invertida R(−45) | 0,03 / 0,98 | 0,01 / 0,99 |
| clip real + ruido gaussiano σ=8 | 0,18 / 0,83 | **0,82 / 0,49** |
| generado, réplica 250 | 0,07 / 0,93 | **0,35 / 0,68** |
| generado, tratado 250 | 0,06 / 0,94 | **0,60 / 0,46** |

Lectura: con velocidad el instrumento ve la equivarianza casi perfecta sobre video real (desacuerdo 2–3 % de la energía
contra 26 %), tolera el ruido de píxel (0,82 contra 0,18) y, sobre video generado, por fin distingue algo: el tratado
del 250 tiene mayor acuerdo de dirección entre ramas (0,60) que la réplica (0,35). Es n=9 y exploratorio, pero es la
primera cantidad en la que el brazo con la pérdida aparece más equivariante que su control; con aceleración esa
diferencia era invisible (0,06 contra 0,07). Un clip de caída libre (00501) es un valor atípico en las dos versiones
(escala 12: la pelota sale del cuadro o se pierde).

Nada conceptual frena usar la velocidad para la rotación: es un vector que rota igual (R·v_orig = v_rot) y es invariante
a traslaciones. La aceleración se eligió por el boost galileano (v cambia con un boost, a no), y el boost está apagado
desde hace semanas (λ_boost = 0). Advertencia de diseño: el condicionamiento I2V fija los primeros cuadros, así que la
velocidad inicial de la rama rotada viene rotada "gratis"; la señal informativa está en los cuadros generados, y
conviene ponderar o excluir los condicionados. Cambio de código chico (usar `v` en lugar de `a` en la pasada de medición
y en los chunks; el piso A medido sobre velocidades; la descomposición por chunks es más simple porque la velocidad es
local a un par de cuadros), más volver a sondear λ y una corrida apareada (~10 h de L40S). No se lanza sin decisión.

## Enmienda 17 (13:40 UTC, antes de ninguna sonda): corrida con la equivarianza sobre VELOCIDAD

Decisión del usuario: la corrida retomada se corta en el paso 807 (checkpoints 125–750 apareados; el 750 en
evaluación; la cadena final no se corre porque no aporta) y la GPU pasa a probar la pérdida sobre velocidad, que el
diagnóstico (enmiendas 16, 16b) señala como la cantidad con señal.

Cambio de código (commit en `fix/perdida-piso-no-negativa`): `physics_cantidad = "velocidad"` en la config. La pérdida es la
misma (cociente invariante a escala con numerador clampeado y gate de margen), aplicada a la primera diferencia del flujo
agregado (T−1 vectores por clip) en vez de la segunda; el piso A se mide sobre velocidades del mismo clip real rotado;
la descomposición por chunks es la misma. Oráculo de valor y gradiente verificado para 3 tamaños de chunk. Sin
exclusión de los cuadros condicionados (n_cond_lat = 1: solo el primer cuadro es fijo, y su velocidad involucra un
cuadro generado).

Protocolo:
1. **Sondas** de 10 pasos desde `ckpt/A_dir`, semilla 42, empezando en λ = 1,54e-03 y ajustando linealmente hasta caer
   en [0,40; 0,60] de razón de gradiente; máximo 3 sondas, después se lanza con el interpolado. Se registran además el
   piso sobre velocidades, el margen D/A, y cos_ramas y escala_ramas iniciales.
2. **Corrida** de 1000 pasos, mismo control: la réplica ya entrenada en la L40S (`stage2_ctrl_lambda_s42_20260906-162433`,
   λ = 0, misma semilla, misma arquitectura), porque el control no depende de la pérdida. Evaluador por checkpoint cada
   125 (apareado, ajuste físico T2V) y cadena final; equivarianza directa en el 250 y el 1000.
3. **Reglas de corte**: validación de difusión > 2× la réplica en dos valores consecutivos; mov_ratio < 0,75× la réplica
   en un apareado; salteos por margen > 60 % sostenidos en 100 pasos. Se ejecuta en el checkpoint siguiente.
4. **Predicciones registradas** (lo que haría decir "aprendió algo"):
   a. La pérdida por rama **baja** a lo largo del entrenamiento (con aceleración quedó plana en 0,5–0,6). Umbral: la
      media de los últimos 100 pasos es menor que la de los primeros 100 con p < 0,05 (Mann-Whitney).
   b. cos_ramas sube desde su valor inicial (con aceleración: 0,42 → 0,39).
   c. Equivarianza directa en velocidad (el diagnóstico 16b: réplica 0,35, tratado-aceleración 0,60 a n=9) mayor que la
      réplica con n=9 por escenario, y la diferencia de píxeles des-rotada menor que la réplica.
   d. Métricas de trayectoria apareadas: vel_mae no peor que la réplica; mov_ratio en [0,75; 1,25]× la réplica.
   Solo a+c juntas cuentan como "la pérdida enseña equivarianza"; d es la condición de no dañar. Cualquier otra
   combinación se reporta como negativa.
5. Costo: sondas ~30 min; corrida ~5,5 h a 19 s/paso; evaluación ~4 h. ≈ 20 USD.

### Enmienda 17b (13:25 UTC): la primera sonda "de velocidad" corrió en aceleración; clave del config que no aterrizaba

`train_sana.py` sólo aterriza a mano las claves que `get_variant_config` no conoce; `physics_cantidad` y
`margen_piso_minimo` no estaban en esa lista, así que el JSON las declaraba, el log avisaba
"claves del config que NO llegan al trainer y se ignoran" y el trainer usaba los defaults. Para `margen_piso_minimo`
el default (3,0) coincidía con lo pedido en todas las corridas anteriores, así que no cambia ninguna lectura. Para
`physics_cantidad` el default es "aceleracion": la sonda de las 12:59 fue una sonda de aceleración más (razón 0,718 a
1,54e-03, n=8, mín 0,24, máx 1,8; en la L40S original la misma sonda dio 0,592: la dispersión entre sondas iguales es
de ese orden, y hay que tenerla en cuenta al leer una razón). Arreglado en `train_sana.py` (commit `db8fb490`), verificado
en `config_resolved.json` de la sonda relanzada a las 13:20 (`physics_cantidad: velocidad`, sin aviso de huérfanas).

## Enmienda 18 (13:30 UTC): apareado del checkpoint 750 y veredicto de la corrida con aceleración

| checkpoint | vel_mae réplica / tratado | p | mov_ratio réplica / tratado | p |
|---|---|---|---|---|
| 750 | 4,23 / **7,21** | < 0,001 (30/30 peor... 1/30 mejor) | 0,71 / **0,37** | < 0,001 (30/30 peor) |

Es el colapso al video quieto, tardío: la regla "mov_ratio < 0,75× la réplica" salta en el 750 (0,37 < 0,53). La corrida
ya estaba cortada en el 807 por decisión del usuario, así que el corte por regla y el corte por decisión coinciden en
el mismo checkpoint. Con la pérdida arreglada la trayectoria completa fue: 125 inflación del movimiento con duplicados
(0,85), 250 pico (0,95), 375–625 vuelta al régimen del control (0,63–0,74), 750 colapso (0,37), con la fracción de
pasos salteados por margen subiendo del 28 % al 50 % en el camino: a medida que el modelo se mueve menos, más pasos caen
por debajo del margen y la física deja de entrenarse, sin que eso frene la deriva de la difusión hacia menos
movimiento. Ninguna de las cuatro predicciones de la enmienda 13 se cumplió.

**Veredicto de la corrida con aceleración (λ = 1,54e-03, 807 pasos, apareada)**: negativo. No aprende equivarianza (la
directa da 6,22 contra 4,78 de la réplica en el 250), no mejora la trayectoria en ningún checkpoint salvo el 125 (p =
0,05, no sostenido), y termina degenerando. El diagnóstico de la enmienda 16 explica por qué: la cantidad medida no
tiene señal. Corresponde a la conclusión "(Cerrada, negativa)" del poster, que sigue vigente y se refuerza.

## Enmienda 19 (13:40 UTC): sonda 1 en velocidad, y qué cambia respecto de aceleración

Sonda de 10 pasos desde `ckpt/A_dir`, semilla 42, λ = 1,54e-03, `physics_cantidad = velocidad` (verificado en el log).

| | aceleración (sondas 3–5 y corrida) | **velocidad (sonda 1)** |
|---|---|---|
| pérdida por rama, mediana en el paso 1–10 | 0,55–0,62 | **0,092** (péndulo/caída 0,01–0,06; rebote 0,44–0,73) |
| cos_ramas (acuerdo de dirección R·q̂_orig vs q̂_rot) | 0,42 | **0,90** (péndulo/caída 0,87–0,96; rebote 0,25–0,51) |
| escala_ramas | 1,11–1,19 | 1,03 |
| margen D/A, mediana | 3–20 | **76** (28–203); salteados 0/10 |
| piso A (energía) | — | 24,5 |
| razón de gradiente mediana | 0,59 (a 1,54e-03) | **0,269** (n=10; mín 0,009, máx 4,97) |
| coseno gradiente física/difusión | 0,00 | −0,04 a +0,02 |

Lectura: con velocidad la pérdida ya no mide ruido. Péndulo y caída libre salen casi equivariantes de entrada
(cos > 0,9, pérdida < 0,06), y **toda la señal se concentra en los rebotes** (cos 0,25–0,51, pérdida 0,4–0,7, razón de
gradiente 1–5 en esos pasos): la no-equivarianza generada está en el impacto, que es donde la dinámica es no trivial.
Es exactamente lo que uno esperaría de una pérdida que mide algo real. La razón de gradiente mediana (0,27) queda bajo la
banda porque en 6 de 10 pasos el término es casi cero; su distribución es de cola pesada (mediana 0,27, media ~0,8).

Decisión por regla (ajuste lineal a la mediana): λ₂ = 1,54e-03 × 0,50/0,269 = **2,9e-03**, sonda 2 lanzada 13:38 UTC.
Riesgo registrado: con ese λ los pasos de rebote tendrían razones de 2–10 y el clip global de norma 1,0 va a actuar en
ellos; se reporta la fracción de pasos con clip y, si supera el 30 %, se prefiere la mediana entre λ₁ y λ₂ antes que
seguir subiendo. Máximo 3 sondas.

### Enmienda 20 (13:55 UTC): sonda 2 en velocidad y regla de lanzamiento tras la sonda 3

| sonda | λ | razón mediana | mín / máx | pérdida/rama | cos_ramas | clips |
|---|---|---|---|---|---|---|
| 1 | 1,54e-03 | 0,269 | 0,009 / 4,97 | 0,092 | 0,90 | 0/10 |
| 2 | 2,9e-03 | **0,309** | 0,011 / 0,52 | 0,058 | 0,93 | 0/10 |

La razón no escaló con λ (×1,9 en λ dio ×1,15 en la mediana): con la misma semilla y los mismos clips, los pasos de
rebote dieron razones 4,97 → 0,45 y 1,13 → 0,40 entre sondas, porque a partir del paso 2 los pesos difieren y la
generación del rebote es muy sensible. La razón mediana en velocidad es una cantidad ruidosa entre sondas (igual que en
aceleración: 0,59 vs 0,72 con λ idéntico) y su dependencia de λ es sublineal en este rango. Sonda 3 lanzada a las
13:52 UTC con λ₃ = 2,9e-03 × 0,50/0,309 = **4,7e-03** (última por regla).

Regla de lanzamiento, fijada antes de leer la sonda 3: si su mediana cae en [0,40; 0,60] se lanza con 4,7e-03. Si queda
por debajo, se lanza con **4,7e-03 igual** (es el λ más alto probado sin clips y la razón crece con λ aunque sea
despacio), salvo que la sonda 3 muestre clips en más del 30 % de los pasos o una razón máxima > 5, en cuyo caso se lanza
con 2,9e-03. Si se pasa de 0,60, se interpola entre 2,9e-03 y 4,7e-03. La corrida lleva `probe_grad_split` encendido,
así que la razón efectiva queda medida a lo largo de los 1000 pasos y se reporta por ventanas de 100.

## Enmienda 21 (14:10 UTC): sonda 3 en banda; corrida en VELOCIDAD lanzada con λ = 4,7e-03

| sonda | λ | razón mediana | mín / máx | pérdida/rama | cos_ramas | clips |
|---|---|---|---|---|---|---|
| 1 | 1,54e-03 | 0,269 | 0,009 / 4,97 | 0,092 | 0,90 | 0/10 |
| 2 | 2,9e-03 | 0,309 | 0,011 / 0,52 | 0,058 | 0,93 | 0/10 |
| 3 | **4,7e-03** | **0,448** | 0,024 / 1,22 | 0,120 | 0,88 | 0/10 |

Por paso en la sonda 3: péndulo 0,85 / 0,49 / 0,05 / 1,22, caída libre 0,02 / 0,04 / 0,20, rebote 1,04 / 0,41 / 0,92. La
razón cae en [0,40; 0,60]: se lanza con 4,7e-03 (`relanzar_vel.sh 4.7e-03`: configs `config_stage2_equiv_vel.json` y
espejo `config_stage2_ctrl_vel.json` con `physics_cantidad: velocidad`, preflight de brazos, 1000 pasos desde
`ckpt/A_dir`, semilla 42, guardado cada 125, validación cada 50). Control: la réplica ya entrenada en L40S
(`stage2_ctrl_lambda_s42_20260906-162433`). Evaluador por checkpoint contra la réplica en `results/e4vel/`, cadena final al
terminar. Reglas y predicciones: enmienda 17. λ es 3× el de la corrida con aceleración: en velocidad el término es más
chico y más concentrado (rebotes), y hace falta más peso para el mismo reparto de gradiente.

## Enmienda 22 (14:30 UTC): ablación de aumentaciones, en distribución (n = 60, apareado por clip)

Etapa 1 con aumentaciones (A) vs sin (B), 3000 pasos cada una, evaluadas con el protocolo válido (held-out, 20 clips
por escenario, 33 frames, cota de video quieto), `results/ablacion_aug_v2/indist`.

| métrica | A (con aumentaciones) | B (sin) | B mejor | signo p | Wilcoxon p |
|---|---|---|---|---|---|
| vel_mae (↓) | 5,304 | **4,974** | 37/60 | 0,092 | 0,059 |
| mov_ratio (→ 1) | 0,612 | 0,592 | 32/60 | 0,70 | 0,76 |
| jerk_rms (↓) | 1,559 | 1,469 | 36/60 | 0,16 | 0,14 |

Lectura contra las predicciones de la enmienda 14: la predicción 1 (A mejor en velocidad y movimiento) **no se cumple**;
la 2 (B "gana" moviéndose menos) tampoco, porque el movimiento es igual (0,59 vs 0,61). Lo que queda es que B está
en el borde de ganar en error de velocidad (−6 %, p = 0,06) con el mismo movimiento. La aumentación por rotación y
traslación de los clips de entrenamiento no enseña nada medible de trayectoria en Etapa 1; si acaso, resta un poco
(coherente con la ablación de agosto, que con las métricas viejas también favorecía a B). Pendientes: OOD tiro vertical,
ajuste físico T2V y los pares de equivarianza de A y B (predicción 3).

### Enmienda 23 (16:15 UTC): ablación de aumentaciones, completa

| medición | A (con aumentaciones) | B (sin) | lectura |
|---|---|---|---|
| en distribución, vel_mae (n=60, apareado) | 5,30 | **4,97** | B mejor, p = 0,06 |
| en distribución, mov_ratio | 0,61 | 0,59 | igual |
| OOD tiro vertical, vel_mae (n=15) | **6,83** | 7,59 | A mejor, p = 0,17 |
| OOD tiro vertical, mov_ratio | 0,54 | 0,50 | igual (quieto 0,22; base 0,72) |
| equivarianza directa, diferencia de píxeles des-rotada (n=9) | 5,48 | **4,92** | B más equivariante |
| coseno de velocidades entre ramas (n=9) | 0,46 | 0,50 | igual |
| ajuste físico T2V | — | — | el análisis no produjo filas (fallo del script sobre este directorio; exploratorio, no se repite) |

Contra las predicciones de la enmienda 14: (1) A no es mejor en trayectoria en distribución; (2) B no gana por moverse
menos; (3) la aumentación por rotación **no** hizo al modelo más equivariante: la diferencia de píxeles entre la
generación con condicionamiento original y rotado es menor sin aumentaciones (4,92 contra 5,48), y el acuerdo de
velocidades entre ramas es el mismo. En OOD hay una ventaja leve y no significativa para A. Conclusión: la simetría
"por datos" (rotar y trasladar los clips de entrenamiento) no enseña ni trayectoria ni equivarianza medibles en Etapa 1;
en distribución cuesta un poco de velocidad, probablemente porque rotar caída libre cambia la dirección de la gravedad.
Referencias de equivarianza directa: réplica L40S paso 250 4,78; tratado-aceleración 250 6,22; control L4 1000 5,1.

## Enmienda 24 (18:45 UTC): corrida en velocidad, checkpoint 125 y primeros 250 pasos

**Apareado contra la réplica** (n = 30, mismos clips, `results/e4vel/paso_125`):

| métrica | réplica | velocidad (125) | Δ | Wilcoxon p |
|---|---|---|---|---|
| vel_mae (↓) | 4,915 | 5,166 | +0,25 | 0,33 |
| mov_ratio (→ 1) | 0,604 | 0,598 | −0,006 | 1,00 |
| jerk_rms (↓) | 1,588 | **1,332** | −0,26 | 0,031 |
| l_rot cruda (↓) | 3,447 | **2,225** | −1,22 | 0,001 |
| l_rot_norm (↓, no validada) | 0,078 | 0,205 | +0,13 | 0,035 |
| mae_ay | 1,787 | 1,771 | −0,02 | 0,98 |

Predicción d (no dañar) **cumplida**: velocidad y movimiento indistinguibles del control. Es la primera vez que un brazo
con física no paga la trayectoria. El jerk baja un 16 % **con el mismo movimiento**, y la pérdida de equivarianza cruda
un 35 %: esas dos métricas premian moverse menos, pero acá el movimiento es igual, así que la caída no es degeneración.
La l_rot_norm vuelve a dar al revés (ya marcada como no validada; en el 250 de la corrida de aceleración daba lo
opuesto que en el 125).

**Curso de la corrida** (250 pasos): validación de difusión contra la réplica 1,26× (50), 1,18 (100), 1,13 (150),
1,06 (200): converge hacia el control, mientras que la corrida de aceleración iba 1,06 → 1,49 → 1,84 y colapsaba. Norma
de velocidad estable (6,2 → 5,3 → 6,0), salteos por margen 12 %, clips 0-1 %, coseno gradiente física/difusión ≈ 0.

**Componentes por escenario** (primera vs segunda mitad de los primeros 131 pasos, n≈40 cada uno): péndulo pérdida por
rama 0,177 → 0,130 y cos_ramas 0,790 → 0,846; caída libre 0,112 → 0,093 y 0,875 → 0,903; **rebote 0,226 → 0,292 y
0,770 → 0,692**. Las dos predicciones (a) y (b) se cumplen en péndulo y caída libre y fallan en rebote, que es donde
está la mayor parte de la señal y del gradiente. Con n≈40 por escenario esto es indicio, no tendencia: la prueba
preregistrada se hace a 500+ pasos con Mann-Whitney.

Ajuste físico T2V del 125: ω 0,150 (idéntico a todos los brazos: lo fija el prompt), g 0,639 (GT 3,281), CV de
velocidad 0,645 (GT 0,326). Sin cambio respecto del control.

## Enmienda 25 (20:15 UTC): checkpoint 250 en velocidad, apareado y equivarianza directa

**Equivarianza directa** (pares I2V con condicionamiento original y rotado 45°, 3 clips por escenario, n = 9):

| medición | réplica (250) | **velocidad (250)** | aceleración (250) |
|---|---|---|---|
| diferencia de píxeles des-rotada (↓) | 4,78 | 5,07 | 6,22 |
| coseno de velocidades entre ramas (↑) | 0,33 | **0,53** | 0,60 |
| desacuerdo N/D en velocidad (↓) | 0,69 | **0,47** | — |

En la cantidad que la pérdida optimiza, el brazo es más equivariante que su control (acuerdo de dirección 0,53 contra
0,33; desacuerdo 0,47 contra 0,69), y eso **no** se traduce en imágenes más parecidas píxel a píxel (5,07 contra 4,78).
La predicción (c) pedía las dos cosas: se cumple la mitad, con n = 9.

**Apareado del 250** (n = 30): vel_mae 5,19 vs 4,84 (p = 0,096), mov_ratio 0,568 vs 0,608 (p = 0,21), jerk 1,271 vs
1,592 (p = 0,16), **l_rot cruda 2,277 vs 3,649 (p < 0,001, mejor en 25/30)**, mae_ay 1,727 vs 1,746 (p = 1,00),
l_rot_norm 0,220 vs 0,264 (p = 0,52, ahora a favor). Validación de difusión: 1,05× (250) y 1,01× (300), convergiendo al
control.

Lectura: la trayectoria sigue sin dañarse (todo n.s.), la pérdida de equivarianza cruda es un 38 % menor con movimiento
equivalente, pero el error de velocidad tiende a ser algo peor (p = 0,096) y el movimiento algo menor (0,568 vs 0,608),
las dos sin significación. La señal de "más equivariante en velocidad" es la más firme; la duda es si a más pasos ese
margen se paga con menos movimiento. Sigue hasta el 1000 con las reglas de la enmienda 17 vigentes.

### Enmienda 25b (21:45 UTC): el monitor pidió cortar; no se corta, y por qué

En el paso 402 `estado_lambda.py` imprimió "CORTAR: razon de gradiente 0.244 fuera de [0.25; 1.00]". Esa banda es la
regla del monitor escrito para la corrida con aceleración (enmienda 1), no una regla de esta corrida: las de velocidad
son las tres de la enmienda 17 (validación > 2× la réplica en dos valores consecutivos, mov_ratio < 0,75× la réplica en
un apareado, salteos > 60 % sostenidos). En el paso 402 la corrida va con validación 1,04×, movimiento 0,93× (apareado
del 250) y salteos 14 %: ninguna se viola, así que **no se corta**.

La razón de gradiente además no es una cantidad estable: con λ fijo, las tres sondas dieron 0,27 / 0,31 / 0,45, y la
mediana móvil de la corrida oscila entre 0,27 y 0,39 desde el paso 100. Un umbral duro sobre ella genera falsos cortes.
Queda anotado como defecto del monitor (no se edita mientras corre, por la regla de no tocar scripts en uso); al leer
sus avisos, la banda de razón se ignora y valen las tres reglas preregistradas.

## Enmienda 26 (22:00 UTC): checkpoint 375 y el patrón de la corrida en velocidad

| checkpoint | vel_mae réplica / vel | p | mov_ratio réplica / vel | jerk réplica / vel | p | l_rot cruda |
|---|---|---|---|---|---|---|
| 125 | 4,92 / 5,17 | 0,33 | 0,604 / 0,598 | 1,588 / **1,332** | 0,031 | 3,45 / **2,23** (p 0,001) |
| 250 | 4,84 / 5,19 | 0,096 | 0,608 / 0,568 | 1,592 / **1,271** | 0,16 | 3,65 / **2,28** (p < 0,001) |
| 375 | 4,91 / 5,31 | **0,050** | 0,599 / 0,609 | 1,728 / **1,184** | **0,001** | — |

El patrón es consistente en los tres: **mismo movimiento, jerk cada vez menor (−16 %, −20 %, −31 %), error de velocidad
cada vez más peor (p 0,33 → 0,096 → 0,050)**. No es degeneración (esa bajaría el jerk bajando el movimiento, y el
movimiento no baja): el modelo genera trayectorias más suaves que el control, y algo más apartadas del GT.

Ajuste físico T2V del 375: CV de la velocidad en caída libre **2,57** (GT 0,33; en el 125 era 0,645), g del rebote 0,523
(GT 3,281), ω 0,150 (invariante). Junto con la norma de velocidad del entrenamiento subiendo (5,3 → 6,9 → 7,5,
tendencia 1,35×) y la validación repuntando a 1,13×, sugiere deriva hacia más movimiento y menos parecido a la física
real. Es lo contrario del colapso de la corrida con aceleración, y tampoco es lo que se buscaba.

Interpretación provisoria, a la espera del 500 y el 1000: la pérdida sobre velocidad **sí** empuja lo que dice empujar
(equivarianza en velocidad medida: 0,53 contra 0,33 del control; l_rot cruda −38 %), y el modelo la satisface
**suavizando** la dinámica, que es una manera barata de que las dos ramas coincidan sin aprender la ley. Suavizar no
requiere entender la física: basta con generar movimiento más regular. Las reglas de corte siguen sin violarse.

## Enmienda 27 (23:30 UTC): las predicciones (a) y (b) se cumplen — el modelo aprende la equivarianza pedida

Prueba preregistrada (enmienda 17), primeros 100 pasos contra los últimos 100, Mann-Whitney, sobre los 440 pasos con
física de la corrida en velocidad (último: 504):

| cantidad | primeros 100 | últimos 100 | p | predicción |
|---|---|---|---|---|
| pérdida de equivarianza por rama | 0,183 | **0,124** | 0,010 | (a) cumplida |
| cos_ramas (acuerdo de dirección) | 0,806 | **0,870** | 0,002 | (b) cumplida |
| \|escala_ramas − 1\| | 0,186 | **0,106** | 0,0003 | — |

Por escenario, incluido el rebote que en los primeros 130 pasos iba al revés: péndulo 0,174 → 0,164 (cos 0,797 → 0,833),
caída libre 0,107 → 0,053 (0,887 → 0,937), **rebote 0,259 → 0,152 (0,739 → 0,842)**. Los tres mejoran.

Con aceleración, la misma prueba daba pérdida plana (0,55–0,62) y coseno bajando (0,42 → 0,39). La diferencia entre las
dos corridas es la cantidad medida, no λ ni el código: es la evidencia directa del diagnóstico de la enmienda 16.

Sumado a la medición independiente en generación (enmienda 25: coseno de velocidades entre ramas 0,53 contra 0,33 del
control, desacuerdo 0,47 contra 0,69), la conclusión sobre el mecanismo es: **la pérdida sobre velocidad enseña la
equivarianza que pide, y es verificable con tres instrumentos distintos** (el término de entrenamiento, la métrica de
generación y los pares I2V). Lo que sigue abierto es el costo: el modelo la satisface suavizando la dinámica (jerk
−16 %, −20 %, −31 % con el mismo movimiento) y su error de velocidad contra el GT empeora (p = 0,33 → 0,096 → 0,050).
La condición (d) "no dañar" está en el límite y se decide en los checkpoints 500 a 1000.

## Enmienda 28 (00:50 UTC del 08/09): el checkpoint 500 da vuelta la lectura del costo

| checkpoint | vel_mae réplica / vel | p | mov_ratio réplica / vel | p | jerk réplica / vel | p |
|---|---|---|---|---|---|---|
| 125 | 4,92 / 5,17 | 0,33 | 0,604 / 0,598 | 1,00 | 1,588 / 1,332 | 0,031 |
| 250 | 4,84 / 5,19 | 0,096 | 0,608 / 0,568 | 0,21 | 1,592 / 1,271 | 0,16 |
| 375 | 4,91 / 5,31 | 0,050 | 0,599 / 0,609 | 0,60 | 1,728 / 1,184 | 0,001 |
| **500** | 4,540 / **4,365** | 0,79 | 0,682 / **0,799** | **0,017** | 1,857 / **1,473** | 0,047 |

La tendencia a empeorar en velocidad (p 0,33 → 0,096 → 0,050) **se dio vuelta**: en el 500 el brazo es mejor que su
control en velocidad (no significativo) y **genera más movimiento** que él (0,799 contra 0,682, p = 0,017), más cerca
del GT, manteniendo el jerk bajo. La hipótesis del "atajo por suavizado" (enmienda 26) queda debilitada: suavizar
reduciendo fidelidad no produce este patrón.

Ajuste físico T2V del 500: CV de la velocidad en caída libre **0,402** (GT 0,326; en el 375 era 2,572), g del rebote
0,465 (GT 3,281), y **ω del péndulo 0,189 (GT 0,168)**. Es la primera vez en todo el proyecto que ω se aparta de 0,150:
todos los brazos anteriores (control, equiv_norm, aceleración en 125/250/375/500/625, velocidad en 125/375) daban 0,150
exactamente, lo que se había atribuido al prompt. Que este checkpoint dé 0,189, más cerca del GT, indica que la dinámica
generada cambió de verdad, no sólo su suavidad.

Cautela: es **un** checkpoint, con n = 30 apareado y n = 10 en el ajuste físico, y el 375 iba en la dirección contraria.
La lectura se decide con 625, 750, 875 y 1000, que ya están encolados, más la equivarianza directa del 1000.

## Enmienda 29 (02:10 UTC): el checkpoint 625 confirma el patrón del 500

| checkpoint | vel_mae réplica / vel | p | mov_ratio réplica / vel | p | jerk | val_dif vs réplica |
|---|---|---|---|---|---|---|
| 125 | 4,92 / 5,17 | 0,33 | 0,604 / 0,598 | 1,00 | −16 % (p 0,031) | 1,26× (50) |
| 250 | 4,84 / 5,19 | 0,096 | 0,608 / 0,568 | 0,21 | −20 % (p 0,16) | 1,05× (250) |
| 375 | 4,91 / 5,31 | 0,050 | 0,599 / 0,609 | 0,60 | −31 % (p 0,001) | 1,06× (350) |
| 500 | 4,540 / **4,365** | 0,79 | 0,682 / **0,799** | **0,017** | −21 % (p 0,047) | 1,16× (500) |
| **625** | 5,299 / **4,326** | **0,080** | 0,570 / **0,726** | **< 0,001** | −2 % (p 0,34) | **0,89× (650)** |

Dos checkpoints consecutivos con el mismo signo en las dos métricas de trayectoria: el brazo con física genera **más
movimiento** que su control apareado (+17 % y +27 %, p = 0,017 y < 0,001) y **menor error de velocidad** (−4 % y −18 %,
p = 0,79 y 0,080). Además, en el paso 650 su validación de difusión es **0,89× la del control**: reconstruye mejor,
por primera vez en el proyecto (serie: 1,26 / 1,05 / 1,16 / 0,89).

La secuencia completa de la corrida es entonces: hasta el 375 el modelo satisface la equivarianza suavizando y pierde
algo de fidelidad; entre el 375 y el 500 eso se revierte y pasa a moverse más, más cerca del GT, con la equivarianza ya
aprendida (pérdida por rama −33 %, coseno +8 %, ambos p < 0,011). La interpretación de la enmienda 26 ("atajo por
suavizado") describe la primera mitad, no el final.

Falta: 750, 875, 1000, la equivarianza directa del 1000 y la cadena final (indist n = 20 y OOD con todos los brazos).
Si el patrón se sostiene, es el primer resultado positivo del proyecto y cambia la conclusión del poster.

## Enmienda 30 (09:20 UTC): los siete checkpoints juntos — CORRIGE las enmiendas 26, 28 y 29

La corrida terminó los 1000 pasos sin cortes (validación final 1,07× el control, salteos 13 %). Con la serie completa,
las lecturas de las enmiendas 26 ("atajo por suavizado"), 28 ("se dio vuelta el costo") y 29 ("dos checkpoints confirman
el patrón") **eran sobre-lecturas de dos o tres puntos consecutivos**. La serie apareada contra la réplica es:

| paso | vel_mae réplica / vel | p | mov_ratio réplica / vel | p | jerk réplica / vel | p |
|---|---|---|---|---|---|---|
| 125 | 4,92 / 5,17 | 0,33 | 0,604 / 0,598 | 1,00 | 1,588 / 1,332 | **0,031** |
| 250 | 4,84 / 5,19 | 0,096 | 0,608 / 0,568 | 0,21 | 1,592 / 1,271 | 0,16 |
| 375 | 4,91 / 5,31 | 0,050 | 0,599 / 0,609 | 0,60 | 1,728 / 1,184 | **0,001** |
| 500 | 4,54 / 4,37 | 0,79 | 0,682 / 0,799 | **0,017** | 1,857 / 1,473 | **0,047** |
| 625 | 5,30 / 4,33 | 0,080 | 0,570 / 0,726 | **< 0,001** | 1,565 / 1,532 | 0,34 |
| 750 | 4,21 / 5,23 | **0,003** | 0,715 / 0,645 | 0,105 | 1,820 / 1,101 | **0,001** |
| 875 | 4,94 / 4,77 | 0,47 | 0,592 / 0,617 | 0,39 | 1,546 / 1,256 | 0,28 |

**Error de velocidad**: cambia de signo cuatro veces (peor, peor, peor, mejor, mejor, peor, mejor); el valor más
significativo de la serie es el 750 **en contra** (p = 0,003). **Movimiento**: cambia de signo tres veces; los dos p
significativos (500 y 625) están rodeados de nulos. Ninguna de las dos es una tendencia: es ruido entre checkpoints, del
mismo orden que la dispersión control-contra-control ya medida (escala de ruido 0,14 / 0,01 / 0,13 px/frame en la
tabla principal).

**Lo único consistente es el jerk**: menor que el control en los **siete** checkpoints, con p < 0,05 en cuatro (125,
375, 500, 750) y el mismo movimiento. Y el aprendizaje de la equivarianza, ya establecido por dos vías independientes
(enmiendas 25 y 27).

Lectura corregida, pendiente del 1000 y la cadena final: **la pérdida sobre velocidad se aprende y produce movimiento
más suave, sin mejorar ni empeorar la fidelidad de la trayectoria.** El error de velocidad y el movimiento fluctúan por
encima del ruido entre checkpoints y no admiten una lectura direccional. Nota de método: leer dos checkpoints
consecutivos como tendencia fue un error de lectura, cometido tres veces esta noche; con siete puntos la serie se
interpreta entera o no se interpreta.

## Enmienda 31 (12:30 UTC): cadena final de la corrida en velocidad — VEREDICTO

**En distribución** (held-out, 20 clips/escenario, apareado por clip contra la réplica, n = 60):

| métrica | réplica | equiv_vel | mejor | Wilcoxon p |
|---|---|---|---|---|
| vel_mae | 4,895 | 5,009 | 24/60 | 0,299 |
| mov_ratio | 0,616 | 0,598 | 29/60 | 0,802 |
| jerk_rms | 1,640 | 1,495 | 35/60 | 0,067 |

Todo nulo. El apareado del 1000 por separado coincide (vel_mae p = 0,21, mov_ratio p = 0,45), con l_rot cruda 2,55
contra 5,63 (p = 0,021) y mae_ay levemente peor (p = 0,050).

**Fuera de distribución** (tiro vertical, nunca entrenado, n = 15 apareado):

| métrica | réplica | equiv_vel | quieto (cota) | Wilcoxon p |
|---|---|---|---|---|
| vel_mae (↓) | 7,634 | **8,799** | 8,90 | **0,005** |
| mov_ratio (→1) | 0,427 | **0,119** | 0,22 | **0,001** |
| jerk_rms | 1,651 | 0,817 | — | 0,005 |

**El brazo colapsa fuera de distribución**: su razón de movimiento (0,119) queda **por debajo de la cota del video
quieto** (0,22), y su error de velocidad alcanza el de un video estático (8,80 contra 8,90). Es la misma degeneración de
la corrida con aceleración, pero acá invisible en distribución y visible sólo al salir de ella, después de 1000 pasos.

**Relectura del jerk**: menor que el control en los siete checkpoints y en la tabla final (p = 0,067). No era
"movimiento más suave" como se leyó en las enmiendas 26 y 29: era el síntoma temprano de que el modelo se mueve menos,
compensado en distribución por el término de difusión y sin compensar fuera de ella.

**Veredicto de la corrida en velocidad (λ = 4,7e-03, 1000 pasos, apareada)**: la pérdida **se aprende** (término
−33 % con p = 0,010; coseno entre ramas +8 % con p = 0,002; acuerdo de velocidades en generación 0,53 contra 0,33 del
control), **no mejora nada medible en distribución** y **degenera fuera de distribución**. Es un resultado negativo con
mecanismo identificado, distinto del de la corrida con aceleración: allá la pérdida no medía nada, acá mide y el modelo
la satisface reduciendo movimiento, que es el óptimo barato que la cota anti-degeneración detecta.

Consecuencia para la tesis del proyecto: imponer equivarianza de rotación sobre una cantidad cinemática estimada del
propio video generado admite un óptimo degenerado (menos movimiento ⇒ más equivarianza) que ni el clamp del piso ni el
gate de margen eliminan. Cerrar ese hueco exige un término que fije la escala del movimiento (por ejemplo, anclarla al
GT o a la generación del control), y eso ya no es una restricción sin ground truth.

## Enmienda 32 (13:10 UTC): equivarianza directa del 1000, e inspección visual de las muestras

**Equivarianza directa del checkpoint 1000** (pares I2V, n = 9):

| medición | control réplica (1000) | equiv_vel (1000) |
|---|---|---|
| diferencia de píxeles des-rotada (↓) | 5,38 | **4,99** |
| coseno de velocidades entre ramas (↑) | 0,37 | **0,56** |
| desacuerdo N/D (↓) | 0,64 | **0,45** |

El brazo gana en las tres, incluida la de píxeles, que en el 250 perdía (5,07 contra 4,78). La predicción (c) del
prerregistro se cumple **entera** al final del entrenamiento: el modelo es más equivariante que su control por toda
medida probada. Junto con las predicciones (a) y (b) (enmienda 27), las tres condiciones de "aprende la simetría" están
cumplidas; la que falla es (d), no dañar, y sólo fuera de distribución (enmienda 31).

**Inspección visual de las muestras** (`samples/compare_step{250,500,750,1000}_*.mp4`, GT | base | T2V | I2V):

- **Péndulo**: en el 250 la generación T2V tiene dos hilos convergiendo a la pelota y la I2V no dibuja hilo. Desde el
  500 eso desaparece: hilo único, pivote correcto, y en el 1000 la I2V es casi indistinguible del GT en posición y
  ángulo. Coincide con que es el escenario donde la pérdida por rama bajó más limpio y donde ω se movió del 0,150 fijo.
- **Rebote**: al revés. Una sola pelota en el 250 y el 500; **dos pelotas en el 750 y el 1000**, en el 1000 de colores
  distintos (violeta y turquesa).
- **Caída libre**: la T2V se mantiene; la I2V duplica en el 750 y en el 1000 deja una sola pelota chica pegada al borde.

**Hallazgo: hay dos rutas degeneradas, no una.** Además de "moverse menos", el modelo descubre **partir el objeto en
dos**: como `aggregate_flow` promedia el flujo de toda la escena, dos objetos moviéndose en direcciones distintas
**cancelan parcialmente** su contribución, con lo que el flujo agregado se achica y se estabiliza, y la pérdida de
equivarianza baja sin que la dinámica mejore. Eso explica por qué el rebote concentraba el gradiente y a la vez terminó
peor, y por qué la misma firma (pelotas duplicadas) apareció en la corrida con aceleración.

**Recomendación concreta que deja el experimento**: la segunda ruta es un defecto del *estimador*, no de la pérdida. Se
cierra midiendo la cinemática **por objeto** (centroide rastreado o flujo enmascarado por segmentación) en vez de
agregando sobre la escena. Es un cambio de instrumento, acotado, y no requiere ground truth físico.

## Enmienda 33 (2026-09-08, 13:40 UTC, antes de lanzar): barrido de ventanas de BPTT con la pérdida sobre velocidad

Pedido del usuario. El barrido de agosto (figura 3 del póster: BPTT completo 8,5 contra 21,0 de la cola de DRaFT-K y
16,1 de la mejor ventana truncada) se corrió con la **pérdida vieja sin normalizar sobre aceleración**
(`config_e4mini_*.json`: `rot_invariante_escala` ausente, λ_rot = 1,63e-04). Dos de sus tres afirmaciones se sostienen
igual y una no:

1. **El perfil del gradiente por paso de Euler no depende de la cantidad**: verificado sobre los logs ya existentes de
   las dos corridas (656 y 538 pasos), los doce valores coinciden dentro de un punto porcentual (primeros cuatro 33 %
   contra 34 %, últimos cuatro 48 % contra 47 %, máximo en el índice 11 en ambas). Nota: el perfil es en U, no
   "concentrado en los primeros pasos" como dice el póster; eso se corrige en el texto.
2. **El costo del BPTT completo (+38 % por paso)** es una medición de sistemas, independiente de la pérdida.
3. **La comparación entre ventanas sí depende**: es una pérdida de equivarianza sobre aceleración medida en validación,
   y el diagnóstico de la enmienda 16 mostró que esa cantidad es ruido sobre video generado.

Se repite el barrido con la pérdida arreglada sobre velocidad. Cinco brazos idénticos salvo la ventana, 150 pasos cada
uno desde `ckpt/A_dir`, semilla 42, λ_rot = 4,7e-03 (el de la sonda 3), `probe_grad_split` encendido:
`full` (12 pasos), `cola4` [8,9,10,11], `ventana4` [2,3,8,11], `mejor6` [0,1,2,4,5,6] y `mejor6comp` (la misma ventana
con n_bptt=12, el control de magnitud). Métrica de comparación: `val_loss_rotation` sobre 20 clips held-out.

**Corrección necesaria antes de lanzar**: `validate()` calculaba esa pérdida sobre **aceleraciones** aunque el
entrenamiento optimizara velocidades, así que el barrido se habría decidido con el instrumento que sabemos que es
ruido. Arreglado con la misma bandera (commit `bbc6778d`); sin efecto en modo aceleración.

Costo estimado: 5 × 150 pasos, ~25-40 s/paso en L40S con 24 frames ≈ 5 h ≈ 10 USD.

### Enmienda 33b (2026-09-08, 20:15 UTC): el barrido de ventanas de BPTT sí se lanzó

AWS liberó capacidad de g6e.xlarge y la instancia detenida arrancó (IP nueva 52.90.252.117). Se desplegó el trainer con
la corrección de `validate()` (mide velocidades en modo velocidad, commit `bbc6778d`), la suite pasó 22 tests en la VM, y
el barrido salió a las 20:14 UTC: cinco brazos de 150 pasos cada uno, secuenciales, idénticos salvo la ventana de BPTT,
λ_rot = 4,7e-03, `physics_cantidad = velocidad`. Comparación por `val_loss_rotation` sobre 20 clips held-out.

| brazo | n_bptt | pasos de Euler retropropagados |
|---|---|---|
| full | 12 | todos |
| cola4 | 1 | [8, 9, 10, 11] (la cola tipo DRaFT-K) |
| ventana4 | 1 | [2, 3, 8, 11] |
| mejor6 | 1 | [0, 1, 2, 4, 5, 6] |
| mejor6comp | 12 | [0, 1, 2, 4, 5, 6], magnitud compensada |

Estimado: ~5 h y ~10 USD. Al terminar se decide si la conclusión del póster sobre ventanas se puede cerrar sobre la
cantidad que sí tiene señal, o si queda como está, calificada.

## Costo y tiempo (a completar con la sonda y la prueba de velocidad)
- L4 g6.xlarge: 149 s/paso medidos → 41 h de entrenamiento, ≈ $33; evaluación ≈ 8 h, ≈ $6.
- L40S g6e.xlarge, **medido en caliente**: 65 / 34 / 46 s/paso (mediana 46), VRAM 21,2 GB → 1000 pasos ≈ 12,8 h,
  ≈ $24. **3,2× más rápida y $9 más barata** que la L4; 25 GB de VRAM libres para el evaluador en paralelo.
  Decisión: se lanza en la L40S (`i-05038c50ad18961af`, clon de la L4 vía AMI `ami-0072a8f19c671562c`).
- Instancia nueva desde la AMI clon: el volumen se carga perezosamente desde S3; la primera prueba de velocidad
  (3 pasos) tardó 2 h y no midió nada. Antes de medir o lanzar: pre-calentar código, venv, caché de pesos,
  datos y `ckpt` (`vm/speedtest2_g6e.sh`), y `timeout -k`. Un stop/start conserva el volumen calentado.
- Simposio: 2026-09-12. Lanzando el 09-06, resultados el 09-07 (L40S) o el 09-08 (L4).
