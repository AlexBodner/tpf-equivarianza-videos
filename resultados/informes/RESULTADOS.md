# Qué salió del proyecto, y con qué evidencia

Documento de lectura. Los números que sostienen cada afirmación están en el texto; las tablas completas por
checkpoint, escenario y métrica están en el apéndice, y el volcado crudo de todo en `NUMEROS_EN_CRUDO.md`.

## En una página

Se entrenó SANA-Video 2B con LoRA para que respete una simetría física: si se rota la escena, la cinemática de lo
generado tiene que rotar igual. La señal sale de los propios videos generados, sin ground truth físico, estimando el
movimiento con RAFT. Contra un control apareado que entrena idéntico salvo esa pérdida, el resultado es **negativo**,
y las tres corridas fallan por razones distintas que sí quedaron identificadas:

1. **La primera corrida no probó nada**: λ estaba calibrado con una versión con bug, y la física aportaba el 1,8 % del
   gradiente. Nulo esperable.
2. **Con λ efectivo, la pérdida original tiene su mínimo en el video quieto**: con el desacuerdo por debajo del piso de
   ruido del propio RAFT el numerador se vuelve negativo. Colapsó en 100 pasos. Se corrigió con un clamp.
3. **Corregida, sobre aceleración, no aprende nada**, porque la aceleración estimada por RAFT no tiene señal a esta
   escala. **Sobre velocidad sí aprende** la simetría, pero la satisface degenerando: se mueve menos, y descubre que
   partir la pelota en dos cancela el flujo agregado.

La contribución utilizable es el diagnóstico: qué cantidad cinemática se puede medir con este instrumental, qué
óptimos degenerados admite una restricción de equivarianza sin ancla de escala, y qué métricas de la literatura son
inservibles sin cotas.

## 1. El instrumento decide qué se puede aprender

Prueba directa con las mismas funciones del entrenador, sobre pares (original, rotado 45°). El coseno mide acuerdo de
dirección entre R·q̂(original) y q̂(rotado); 1 es equivarianza perfecta. `N/D` es el desacuerdo relativo.

| par | aceleración cos / N/D | velocidad cos / N/D |
|---|---|---|
| mismo clip real, dos pasadas | 1,00 / 0,00 | 1,00 / 0,00 |
| clip real rotado píxel a píxel | 0,74 / 0,26 | **0,97–0,99 / 0,02–0,03** |
| ídem con la convención de signo invertida | 0,03 / 0,98 | 0,01 / 0,99 |
| clip real más ruido gaussiano σ=8 | 0,18 / 0,83 | 0,82 / 0,49 |
| video generado, control | 0,07 / 0,93 | 0,35 / 0,68 |
| video generado, brazo entrenado | 0,06 / 0,94 | 0,60 / 0,46 |

RAFT no está roto: es determinista, la convención de rotación es la correcta (invertirla destruye el acuerdo), y sobre
rebotes reales, donde las aceleraciones son grandes, ve equivarianza casi perfecta. El problema es que la aceleración
es la segunda diferencia del flujo agregado, y a esta escala (pelota del 1 % del cuadro) su ruido es del tamaño de la
señal: sobre el **mismo video real** rotado ya desacuerda un 26 %, y sobre generado un 93 %, o sea nada.

**Consecuencia**: la pérdida sobre aceleración estaba optimizando ruido. Sobre velocidad, la misma pérdida mide algo.

## 2. Qué hizo cada corrida

Todas contra un control apareado que entrena con el mismo clip, el mismo ruido y los mismos pesos iniciales, y difiere
sólo en λ. Verificado: pérdida idéntica en el paso 1.

| corrida | λ efectivo | término de equivarianza | trayectoria en distribución | fuera de distribución |
|---|---|---|---|---|
| E4-norm2 | 0,018 | no medido | nulo (n=60, p≥0,14) | movimiento −32 % (p=0,005) |
| pérdida original, λ efectivo | 0,50 | colapsa | movimiento 0,13 vs 0,60 (30/30) | no se evaluó (cortada) |
| corregida, aceleración | 0,45 | plano en 0,5–0,6 | nulo, luego colapso en el 750 | no se evaluó (cortada) |
| corregida, velocidad | 0,45 | **baja 33 % (p=0,010)** | nulo (n=60, p≥0,29) | **movimiento 0,12 vs 0,43 (p=0,001)** |

## 3. La corrida sobre velocidad, que es la única que aprende

**Aprende la simetría, y se comprueba con tres instrumentos independientes.**

| evidencia | al empezar | al terminar | p |
|---|---|---|---|
| término de la pérdida por rama (entrenamiento) | 0,183 | 0,124 | 0,010 |
| acuerdo de dirección entre ramas (entrenamiento) | 0,806 | 0,870 | 0,002 |
| desvío de escala entre ramas | 0,186 | 0,106 | <0,001 |

En generación libre, contra el control en el mismo paso (n=9 pares por checkpoint):

| medición | control | brazo (paso 250) | brazo (paso 1000) |
|---|---|---|---|
| acuerdo de velocidades entre ramas | 0,37 | 0,53 | **0,56** |
| desacuerdo N/D | 0,64 | 0,47 | **0,45** |
| diferencia de píxeles des-rotada | 5,38 | 5,07 | **4,99** |

Los tres escenarios mejoran, incluido el rebote (0,259 → 0,152 en el término, 0,739 → 0,842 en el acuerdo).
Con la pérdida sobre aceleración, la misma prueba daba término plano (0,55–0,62) y acuerdo bajando (0,42 → 0,39).

**No sirve de nada en distribución.** Tabla final, 20 clips por escenario, apareado por clip contra el control:

| métrica | control | brazo | mejor | p |
|---|---|---|---|---|
| vel_mae | 4,895 | 5,009 | 24/60 | 0,299 |
| mov_ratio | 0,616 | 0,598 | 29/60 | 0,802 |
| jerk_rms | 1,640 | 1,495 | 35/60 | 0,067 |

**Y degenera fuera de distribución.** Tiro vertical, escenario nunca entrenado, 15 clips:

| métrica | control | brazo | cota del video quieto | p |
|---|---|---|---|---|
| vel_mae | 7,634 | 8,799 | 8,899 | 0,005 |
| mov_ratio | 0,427 | 0,119 | 0,216 | 0,001 |
| jerk_rms | 1,651 | 0,817 | 0,000 | 0,005 |

La razón de movimiento del brazo (0,119) queda **por debajo** de la de un video estático (0,216), y su error de
velocidad alcanza al de ese video. Eso es degeneración, no un efecto pequeño.

**Por qué el jerk bajo no era una buena noticia.** El jerk del brazo es menor que el del control en los ocho
checkpoints. Lo leí primero como movimiento más suave; con la evaluación fuera de distribución se entiende que era el
síntoma temprano de que el modelo se mueve menos, compensado en distribución por el término de difusión.

**Las métricas de trayectoria por checkpoint no tienen tendencia**, y conviene decirlo porque invita a engañarse: el
error de velocidad cambia de signo cuatro veces a lo largo de los ocho checkpoints y el movimiento tres. Dos
checkpoints consecutivos favorables (500 y 625) no eran una tendencia; el siguiente (750) es el peor de la serie.

## 4. Las dos rutas degeneradas

La restricción de equivarianza no fija la escala del movimiento, así que admite óptimos que la satisfacen sin física:

1. **Moverse menos.** El desacuerdo entre ramas se achica solo si todo se achica. Es lo que hizo la pérdida original
   (colapso en 100 pasos) y lo que hace la corregida fuera de distribución.
2. **Partir el objeto en dos.** Visible en las muestras de rebote de los checkpoints 750 y 1000, con dos pelotas de
   colores distintos. `aggregate_flow` promedia el flujo de toda la escena pesado por magnitud, así que dos objetos
   moviéndose en direcciones distintas se cancelan parcialmente: el vector agregado se achica y se estabiliza, y la
   pérdida baja sin que la dinámica mejore.

La primera es una propiedad de la restricción y se ataca con un ancla de escala. La segunda es un defecto del
estimador y se cierra midiendo por objeto, con identidad estable entre cuadros, en vez de agregar sobre la escena.

## 5. Cómo se detecta la pelota, y por qué importa

Hay dos mecanismos distintos, y ninguno maneja más de un objeto:

- **En la pérdida** no hay detección. RAFT da flujo denso; `aggregate_flow` resta el flujo medio global, se queda con
  los píxeles que se mueven más de 0,5 px respecto de ese fondo y promedia pesando por magnitud. Es un promedio de
  escena. Dos objetos en direcciones opuestas se cancelan: de ahí la segunda ruta degenerada.
- **En la evaluación** hay un trazador clásico: fondo por mediana temporal, máscara por saturación o por diferencia
  contra el fondo, y en cada cuadro se toma el **componente conexo de mayor área**, sin asociación entre cuadros. Con
  dos pelotas de tamaño parecido la elección puede alternar. Un filtro descarta saltos de más de 60 píxeles, lo que
  baja la tasa de detección (hasta 37 % en algunos rebotes) en vez de corregir la trayectoria.

**Limitación que esto impone**: en los checkpoints con duplicación, el error de velocidad y la razón de movimiento se
miden sobre una trayectoria que puede mezclar dos objetos. La dirección del resultado no cambia, pero los valores
puntuales de esos checkpoints no son confiables.

## 6. Métricas que no se pueden usar

Tres métricas habituales en este problema **las gana un video estático**, y por eso toda tabla lleva la fila `quieto`:

| métrica | GT | video quieto | qué premia |
|---|---|---|---|
| MAE de aceleración | 0,31 / 0,36 / 12,98 | 0,00 | no moverse |
| jerk | 0,31 / 0,36 / 12,98 | 0,00 | no moverse |
| pérdida de equivarianza cruda | — | 0 | no moverse |

La métrica principal es el **error de velocidad** contra el ground truth del simulador, que castiga al video quieto
7,6 veces por encima del piso del instrumento, más la **razón de movimiento** como cota de degeneración.

## 7. La ablación de aumentaciones

Etapa 1 entrenada con clips rotados y trasladados (A) contra sin ellos (B), 3000 pasos, apareado por clip:

| métrica | con aumentaciones | sin | mejor sin | p |
|---|---|---|---|---|
| vel_mae | 5,304 | 4,974 | 37/60 | 0,059 |
| mov_ratio | 0,612 | 0,592 | 32/60 | 0,757 |
| jerk_rms | 1,559 | 1,469 | 36/60 | 0,135 |

Sin aumentaciones no es peor: mejora el error de velocidad en el borde de la significación con el mismo movimiento, y
además queda **más** equivariante (diferencia de píxeles 4,92 contra 5,48). La simetría por datos tampoco enseña lo
que la pérdida intenta enseñar.

## 8. Qué haría falta para cerrar la pregunta

1. **Anclar la escala del movimiento**, porque sin eso la restricción siempre admite el óptimo degenerado. Deja de ser
   una restricción sin ground truth, que era el atractivo original.
2. **Medir por objeto** en vez de agregar sobre la escena. La parte cara, emparejar objetos entre cuadros, hace falta
   sólo para el trazador de evaluación; para cerrar la ruta degenerada de la pérdida alcanza con que `aggregate_flow`
   tome el **componente conexo mayor** de la máscara de movimiento en cada par de cuadros, que es local y no necesita
   identidad temporal. El emparejamiento completo queda como trabajo futuro.
3. **Repetir el barrido de ventanas de BPTT** con la pérdida sobre velocidad: el original se midió con la pérdida
   vieja sobre aceleración. El perfil del gradiente por paso de Euler sí es independiente de la cantidad (verificado
   sobre los logs: los doce valores coinciden dentro de un punto porcentual).

---

# Apéndice: series por checkpoint

## Corrida sobre velocidad (λ=4,7e-03)

Apareado por clip contra el control en el mismo paso, n=30. Formato: control → brazo (p).

| paso | error de velocidad | razón de movimiento | jerk |
|---|---|---|---|
| 125 | 4,915 → 5,166 (0,328) | 0,604 → 0,598 (1,000) | 1,588 → 1,332 (0,031) |
| 250 | 4,838 → 5,190 (0,096) | 0,608 → 0,568 (0,213) | 1,592 → 1,271 (0,164) |
| 375 | 4,913 → 5,312 (0,050) | 0,599 → 0,609 (0,598) | 1,728 → 1,184 (0,001) |
| 500 | 4,540 → 4,365 (0,792) | 0,682 → 0,799 (0,017) | 1,857 → 1,473 (0,047) |
| 625 | 5,299 → 4,326 (0,080) | 0,570 → 0,726 (<0,001) | 1,565 → 1,532 (0,339) |
| 750 | 4,212 → 5,225 (0,003) | 0,715 → 0,645 (0,105) | 1,820 → 1,101 (<0,001) |
| 875 | 4,939 → 4,767 (0,465) | 0,592 → 0,617 (0,393) | 1,546 → 1,256 (0,280) |
| 1000 | 4,619 → 4,900 (0,213) | 0,651 → 0,613 (0,452) | 1,907 → 1,503 (0,253) |

## Corrida sobre aceleración (λ=1,54e-03, cortada en el 807)

Apareado por clip contra el control en el mismo paso, n=30. Formato: control → brazo (p).

| paso | error de velocidad | razón de movimiento | jerk |
|---|---|---|---|
| 125 | 4,908 → 4,277 (0,050) | 0,604 → 0,852 (<0,001) | 1,560 → 2,378 (<0,001) |
| 250 | 4,816 → 4,792 (0,871) | 0,610 → 0,946 (<0,001) | 1,587 → 2,502 (<0,001) |
| 375 | 4,893 → 4,859 (0,984) | 0,601 → 0,735 (<0,001) | 1,803 → 2,182 (0,096) |
| 500 | 4,519 → 5,504 (<0,001) | 0,684 → 0,634 (0,198) | 1,897 → 1,885 (0,808) |
| 625 | 5,276 → 5,299 (0,598) | 0,571 → 0,643 (0,031) | 1,541 → 2,034 (0,213) |
| 750 | 4,229 → 7,212 (<0,001) | 0,713 → 0,374 (<0,001) | 1,775 → 0,978 (<0,001) |

## Ajuste de parámetros físicos (generación libre, n=10 por escenario)

Mediana por escenario. Ningún brazo de ninguna corrida recupera la gravedad del rebote ni la frecuencia del péndulo.

| corrida | paso | ω péndulo (GT 0,168) | g rebote (GT 3,281) | CV velocidad caída libre (GT 0,326) |
|---|---|---|---|---|
| velocidad | 125 | 0,150 | 0,639 | 0,645 |
| velocidad | 375 | 0,257 | 0,523 | 2,572 |
| velocidad | 500 | 0,194 | 0,641 | 0,402 |
| velocidad | 625 | 0,275 | nan | 0,472 |
| velocidad | 750 | 0,162 | 0,216 | 1,242 |
| velocidad | 1000 | 0,262 | 0,749 | 0,285 |
| aceleración | 125 | 0,150 | nan | 0,600 |
| aceleración | 250 | 0,150 | 1,811 | 3,134 |
| aceleración | 375 | nan | 0,849 | 2,323 |
| aceleración | 500 | 0,187 | 0,883 | 0,631 |
| aceleración | 625 | nan | 0,790 | 0,407 |

`nan`: el ajuste no convergió en la mediana de ese escenario, porque el trazador pierde la pelota en demasiados
cuadros. Los valores de ω que se apartan de 0,150 corresponden a los checkpoints donde el trazador alterna entre dos
pelotas, así que no se leen como recuperación de la dinámica.

Tablas completas de todas las métricas, todos los brazos y todos los escenarios: `NUMEROS_EN_CRUDO.md`.
