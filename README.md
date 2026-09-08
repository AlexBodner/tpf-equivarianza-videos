# Qué se ve en cada video

Videos de un trabajo que intentó enseñarle física a un modelo de difusión de video sin ground truth
físico, imponiendo **equivarianza a rotaciones**: si se rota la escena, la cinemática de lo generado
tiene que rotar igual. El movimiento se estima con RAFT (flujo óptico) sobre los propios videos
generados, y todo se retropropaga hasta los pesos.

El resultado global fue **negativo**, pero los modos de falla son informativos y es lo que muestran
estos videos. Detalle completo en el informe del proyecto.

## Cómo leer los paneles

- Los videos `01` a `05` y `11` tienen cuatro paneles: **ground truth del simulador · modelo base sin
  fine-tuning · generación por texto del modelo entrenado · generación condicionada en el primer
  cuadro**. Fondo Perlin gris, pelota de color, 33 cuadros a 384 px.
- Los videos `08` y `10` tienen tres paneles por brazo: **generación con condicionamiento original ·
  generación con el condicionamiento rotado 45° y luego des-rotada · diferencia absoluta**. Si el
  modelo fuera equivariante, los dos primeros paneles serían iguales y el tercero negro.
- El video `06` muestra las dos ramas que la pérdida compara en cada paso de entrenamiento.

## Resultados que salieron bien

**`01_pendulo_bien_paso1000.mp4`** — Péndulo al final del entrenamiento con la pérdida sobre
velocidad. La generación condicionada sigue el ground truth de cerca: pivote en su lugar, hilo único,
ángulo y fase razonables. Es el escenario donde la pérdida bajó más limpio.

**`06_dos_ramas_de_la_perdida.mp4`** — Las dos generaciones que la pérdida compara: mismo clip, mismo
ruido, una con la escena original y otra rotada. La inconsistencia entre sus aceleraciones es toda la
señal de entrenamiento, y no necesita anotaciones.

**`08_equivarianza_*_velocidad.mp4`** — Equivarianza medida directamente en el brazo entrenado sobre
velocidad. El tercer panel (la diferencia) es más oscuro que el del control, y numéricamente el brazo
termina más equivariante que su control por las tres medidas que probamos. La pérdida sí enseña lo
que dice enseñar.

## Modos de falla

**`03_rebote_pelota_duplicada_paso1000.mp4`** — El hallazgo más interesante. Aparecen **dos pelotas**
de colores distintos. No es un artefacto casual: el estimador promedia el flujo óptico de toda la
escena, así que dos objetos moviéndose en direcciones distintas se cancelan parcialmente, el vector
agregado se achica y **la pérdida de equivarianza baja sin que la física mejore**. El modelo encontró
que partir el objeto es más barato que aprender la simetría. Comparar con
`04_rebote_una_pelota_paso250.mp4`, el mismo escenario 750 pasos antes, con una sola pelota.

**`02_pendulo_hilo_doble_paso250.mp4`** — Versión temprana del mismo truco: dos hilos convergiendo a
la pelota, y en el panel condicionado la pelota flota sin hilo. Desaparece más adelante en el
entrenamiento, a diferencia del caso del rebote.

**`05_caida_libre_paso1000.mp4`** — La otra ruta degenerada, más sutil: la pelota se vuelve pequeña y
se pega al borde. Menos movimiento significa menos desacuerdo entre las dos ramas, así que la
restricción también se satisface **moviéndose menos**. Fuera de distribución esto se vuelve severo: el
modelo entrenado genera menos movimiento que un video estático de referencia.

**`07_fuera_de_dominio_rodando.mp4`** — Un prompt que nunca se entrenó ("una pelota rodando"). Es
donde la degeneración se ve sin ambigüedad.

**`11_aceleracion_pendulo_paso250.mp4`** — La misma pérdida aplicada a la **aceleración** en vez de la
velocidad. Acá el modelo no aprende nada, y por una razón medible: a esta escala, la aceleración que
estima RAFT es ruido. Sobre el *mismo video real* rotado píxel a píxel, el instrumento ya se
contradice un 26 % con aceleración contra un 2 % con velocidad; sobre video generado, un 93 % contra
un 45 %.

**`10_equivarianza_*_ablacion_aumentaciones.mp4`** — Control del método: entrenar con clips rotados y
trasladados (aumentación de datos) tampoco produce un modelo más equivariante. La simetría por datos
no reemplaza a la restricción, y la restricción tampoco alcanzó.

## Qué quedó como recomendación

1. Medir la cinemática **por objeto** en vez de promediar la escena. Para cerrar la ruta degenerada
   del rebote ni siquiera hace falta emparejar objetos entre cuadros: alcanza con tomar el componente
   conexo mayor de la máscara de movimiento en cada par.
2. **Anclar la escala del movimiento**, porque sin eso la restricción siempre admite el óptimo de
   moverse menos. El costo es que deja de ser una restricción sin ground truth, que era el atractivo.

## Créditos

Alexander Bodner y Mateo Costantini, Universidad de San Andrés, Visión Artificial Avanzada.
Modelo base: SANA-Video 2B. Flujo óptico: RAFT. Los clips del simulador son sintéticos y propios.
