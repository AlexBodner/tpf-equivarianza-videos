# Equivarianza como prior físico en difusión de video — los videos

Videos del TPF de Visión Artificial Avanzada (UdeSA). El trabajo intenta enseñarle física a un modelo
de difusión de video **sin ground truth físico**, imponiendo una simetría: si se rota la escena, la
cinemática de lo generado tiene que rotar igual. El movimiento se estima con RAFT (flujo óptico) sobre
los propios videos generados, y todo se retropropaga hasta los pesos.

Con el método tal como está hoy **no logramos mejorar la física generada**, y las secciones de abajo
cuentan por qué, experimento por experimento. La conclusión no es que la idea no sirva, sino algo más
concreto y más accionable: la pérdida ve una versión **parcial y cruda** de lo que el modelo genera
—de los 32 vectores de velocidad del clip mira 24 en péndulo y rebote, pero sólo 12 o 16 en caída
libre, y los calcula sobre una generación de 12 pasos de Euler y no de los 20 de la inferencia—, y
contra esa señal incompleta el entrenamiento termina introduciendo
**corrupciones** en la imagen que bajan la pérdida sin mejorar la dinámica. Cada sección indica a qué
parte del póster corresponde.

---

## 1. Cómo leer los paneles

- Videos de cuatro paneles: **ground truth del simulador · modelo base sin fine-tuning · generación
  por texto · generación condicionada**. Fondo Perlin gris, pelota de color, 33 cuadros a 384 px.
  El condicionamiento son los **2 primeros cuadros latentes** del clip real, que en píxeles son los
  primeros 8 cuadros; el resto lo genera el modelo.
- Videos de equivarianza, tres paneles: **generación con el condicionamiento original · con el
  condicionamiento rotado 45° y luego des-rotada · diferencia absoluta**. Si el modelo fuera
  equivariante, los dos primeros serían iguales y el tercero negro.
- Todos los MP4 están en [`videos/`](videos); los GIF de abajo son los mismos, reducidos.
- **Cada video dice de qué checkpoint sale.** Los dos brazos siempre se comparan en el *mismo* paso, y
  el nombre del archivo lo lleva.
- Los JSON de todas las evaluaciones, los logs de entrenamiento y las configuraciones están en
  [`resultados/`](resultados), para que los números se puedan verificar sin la máquina de entrenamiento.

<a id="metodo"></a>

## 2. El método: las dos ramas que compara la pérdida

*Póster: sección «Método».*

![Las dos ramas de la pérdida](gifs/ramas_de_la_perdida.gif)

*Checkpoint 250 del brazo con física. Las dos generaciones del mismo clip, con el mismo ruido: a la
izquierda la escena original, a la derecha la escena rotada 45°. La pérdida compara la cinemática de una contra la de la otra rotada.*

En cada paso de entrenamiento el modelo genera el mismo clip dos veces, con el mismo ruido: una con la
escena original y otra rotada. Si fuera equivariante, la cinemática de la segunda sería la de la
primera rotada. La inconsistencia entre ambas es toda la señal de entrenamiento, y no requiere
anotaciones. El costo es que hay que generar, decodificar y estimar flujo **dentro** del paso de
entrenamiento, y retropropagar por todo eso.

Archivo: `videos/06_dos_ramas_de_la_perdida.mp4`. Una aclaración para que no confunda: el
entrenamiento decodifica las dos ramas **de a un latente por vez** para que entren en memoria, y cada
uno de esos trozos son unos 5 cuadros. Eso es el troceado del decodificador, no el largo de la rama:
las dos ramas cubren el clip entero, 33 cuadros, que es lo que muestra este video.

<a id="colapso"></a>

## 3. Primer experimento: la pérdida original colapsa al video quieto

*Póster: sección «Diagnóstico y corrección».*

![Colapso contra el control](gifs/colapso_vs_control.gif)

Con λ calibrado para que la física pese la mitad del gradiente, el modelo **deja de mover la pelota**
en 100 pasos. A la izquierda el brazo con la pérdida, a la derecha su control entrenado en paralelo
sin ella, **ambos en el paso 100**. Medido en el paso 125: razón de movimiento 0,13 contra 0,60 del control,
peor en los 30 clips.

**Por qué pasa, en una ecuación.** La pérdida compara la cinemática de las dos ramas, normalizada por
la energía total para que no dependa de la escala, y le resta el piso de ruido $A$ del propio
estimador (lo que RAFT se contradice a sí mismo al rotar un video real):

$$\mathcal{L}_{\text{rot}} \;=\; \frac{N - A}{\max\left(D - A,\; A\right)}
\qquad
N = \lVert R(\theta)\,\hat a_{\text{orig}} - \hat a_{\text{rot}} \rVert^2
\qquad
D = \lVert \hat a_{\text{orig}} \rVert^2 + \lVert \hat a_{\text{rot}} \rVert^2$$

Restar $A$ es correcto: sin eso la pérdida premia generar aceleraciones enormes, porque el ruido pesa
proporcionalmente menos. El problema es qué pasa cuando el modelo **deja de moverse**. Ahí
$N \to 0$ y $D \to 0$, y la pérdida tiende a

$$\mathcal{L}_{\text{rot}} \;\longrightarrow\; \frac{0 - A}{\max(0 - A,\; A)} \;=\; \frac{-A}{A} \;=\; -1,$$

que es su **mínimo global**. Un modelo perfectamente equivariante que sí se mueve da
$(0-A)/(D-A) \approx -A/D \to 0^{-}$: *peor* que quedarse quieto. Con $A = 1$: la pérdida vale $0$ en
$D = 2A$, $-0{,}40$ en $D = 1{,}5A$ y $-0{,}90$ en $D = 0{,}5A$. El descenso por gradiente encuentra
eso antes que la simetría.

La corrección es recortar el numerador en cero, con lo que por debajo del piso la pérdida vale $0$ y
deja de empujar:

$$\mathcal{L}_{\text{rot}}^{\text{corregida}} \;=\; \frac{\max(N - A,\; 0)}{\max(D - A,\; A)}$$

Archivo: `videos/12_colapso_vs_control_paso100.mp4`

<a id="aceleracion"></a>

## 4. Segundo experimento: sobre aceleración no hay señal que aprender

*Póster: sección «Diagnóstico y corrección».*

Ya corregida la fórmula, la pérdida se aplicó a la **aceleración** estimada por RAFT. Acá no hay nada
que mostrar en video, y eso es exactamente el punto: **la falla no es visual, es que el término nunca
baja**. Se ve en las curvas de entrenamiento, no en un cuadro.

![La misma pérdida sobre aceleración y sobre velocidad](figuras/aceleracion_vs_velocidad.png)

*Izquierda: el término de la pérdida, normalizado igual en los dos casos (0 sería equivarianza
perfecta). Sobre aceleración se queda plano entre 0,5 y 0,6 durante 800 pasos; sobre velocidad baja un
31 % (p = 0,043 de una cola). Derecha: el acuerdo de dirección entre las dos ramas. Sobre aceleración arranca en
0,42 y baja; sobre velocidad arranca en 0,81 y sube a 0,86 (p = 0,062 de una cola). La curva de aceleración empieza
en el paso 280 porque el registro de esa cantidad se agregó cuando la corrida se retomó.*

La razón es del instrumento, no del modelo. La aceleración es la segunda diferencia del flujo, y a esta
escala su ruido es del tamaño de la señal: sobre el *mismo video real* rotado píxel a píxel, el
estimador ya se contradice un 26 %; sobre video generado, un 93 %, o sea que no mide nada. Con
velocidad, sobre video real el desacuerdo baja al 2 %.

**Y sin embargo la corrida se ve bien.** Sus muestras en el paso 250 son indistinguibles de las de
cualquier otro brazo, porque un término que no aprende tampoco rompe nada. Recién en el paso 750
colapsa (razón de movimiento 0,37 contra 0,71 del control) y ahí se cortó.

**Qué terminó haciendo.** Esto sí se ve:

![La corrida sobre aceleración, paso a paso](gifs/aceleracion_evolucion_bouncing.gif)

*Generación condicionada del mismo clip de rebote en los pasos 250, 500 y 750. En el 250 hay varias
pelotas a la vez, en el 500 quedan dos, y en el 750 una sola y pálida: la razón de movimiento cae a
0,37 contra 0,71 del control y ahí se cortó la corrida. La pérdida no bajó en ningún momento; lo que
cambió fue la imagen.*

Archivos: `videos/15_aceleracion_evolucion_*.mp4` y `videos/11_aceleracion_pendulo_paso250.mp4`, este
último para comprobar que en el paso 250 no se distingue de cualquier otro brazo.

<a id="velocidad"></a>

## 5. Tercer experimento: sobre velocidad el modelo sí se vuelve más equivariante

*Póster: sección «Resultados».*

La **velocidad** (primera diferencia del flujo) sí tiene señal: sobre video real rotado el estimador
se contradice sólo un 2 %. Con la pérdida aplicada ahí, el modelo sí se mueve en la dirección pedida.
Antes de los números, cómo se decidió: el criterio de "la pérdida enseña equivarianza" se escribió
**antes de correr**, y pedía dos cosas juntas — que el término baje durante el entrenamiento, y que el
modelo resulte más equivariante que su control en una medición hecha fuera del bucle de entrenamiento.
Las dos se cumplen.

- **El término baja**: media de los últimos 100 pasos contra los primeros 100, 0,1834 → 0,1266, un
  31 % menos (Mann-Whitney de una cola, **p = 0,043**; la dirección estaba preregistrada, así que el
  test es unilateral — a dos colas sería p = 0,085). Junto con eso, el acuerdo de dirección entre las
  dos ramas sube de 0,81 a 0,86 (p = 0,062 de una cola). Conviene aclarar que estas dos cantidades **no
  son medidas independientes**: son la norma y el ángulo de la misma diferencia, calculadas en la misma
  función sobre los mismos vectores.
- **La medición externa**, sobre 9 pares condicionados del checkpoint 1000, donde el brazo gana en las
  tres cantidades: coseno entre ramas 0,56 contra 0,37 del control, desacuerdo N/D 0,45 contra 0,64, y
  diferencia de píxeles des-rotada 4,99 contra 5,38. Es n = 9 y sin test estadístico, pero es la única
  lectura que no viene del propio término que se está optimizando. En el checkpoint 250 esta misma
  medición daba a favor del control; se da vuelta recién al final.

Salida cruda: `resultados/evaluaciones/e4vel__equiv_1000__diagnostico_completo.log`. La prueba de
tendencia se reproduce con `scripts_figuras/gen_aceleracion_vs_velocidad.py`.

![Equivarianza del brazo entrenado](gifs/equivarianza_pendulo.gif)

*Checkpoint 250. Condicionamiento original · rotado 45° y des-rotado · diferencia. Cuanto más oscuro el
tercer panel, más equivariante. En el 250 esta medición todavía daba a favor del control (5,07 contra
4,78 de diferencia de píxeles); recién en el 1000 se da vuelta (4,99 contra 5,38), y ahí el coseno
entre ramas es 0,56 contra 0,37. Las dos mediciones están en
`resultados/evaluaciones/e4vel__equiv_1000__diagnostico_completo.log`.*

![Péndulo al paso 1000](gifs/pendulo_bien.gif)

*Checkpoint 1000. El péndulo es el escenario donde mejor sale: la generación condicionada sigue al
ground truth con el pivote en su lugar y el hilo único.*

Archivos: `videos/08_equivarianza_*_velocidad.mp4`, `videos/01_pendulo_bien_paso1000.mp4`

<a id="ladoalado"></a>

## 6. Lado a lado: el control contra el brazo con física

*Póster: sección «Resultados». Es el contraste que decide todo el trabajo.*

Los dos brazos entrenan con el mismo clip, el mismo ruido y los mismos pesos iniciales; lo único que
los separa es la pérdida de equivarianza. Acá generan el **mismo clip**, así que la diferencia que se
ve es atribuible a la pérdida y a nada más.

Los videos de esta sección son del **checkpoint 250 de ambos brazos**, y conviene ser claro sobre por
qué: no porque sea el mejor (no lo es, ver sección 8), sino porque es el único paso donde se generaron
los pares de ambos brazos sobre los mismos clips. En el checkpoint 1000, que es la elección primaria, la
comparación existe en números pero todavía no en video.

![Los tres escenarios, control contra brazo con física](gifs/tres_escenarios_i2v.gif)

*Checkpoint 250 de los dos brazos. Los tres escenarios a la vez, generación condicionada en los 2 primeros latentes. Arrancan idénticos porque
el condicionamiento es el mismo. En **caída libre** las dos trayectorias son parecidas y el brazo con
física termina más cerca del ground truth. En **péndulo** también. En **rebote** la pelota del brazo con
física se desdibuja y queda atrás: es el escenario donde empeora, y donde después aparecen los
duplicados.*

![Los tres escenarios, generación libre](gifs/tres_escenarios_t2v.gif)

*Checkpoint 250, generando sólo desde el texto, sin condicionamiento: acá las dos ramas no tienen por qué
coincidir en posición, y se ve mejor la diferencia de dinámica.*

Y en detalle, dos casos:

![Rebote: control contra brazo con física](gifs/lado_a_lado_rebote.gif)

*Rebote, checkpoint 250 de los dos brazos, generación condicionada. Arrancan idénticos, porque el
condicionamiento es el mismo, y divergen: la pelota del brazo con física recorre menos y hacia el final se desdibuja.
Es la degeneración empezando, en un caso donde las métricas en distribución todavía no la marcan.*

![Péndulo: control contra brazo con física](gifs/lado_a_lado_pendulo.gif)

*Péndulo, checkpoint 250, generación libre por texto. Acá el brazo con física dibuja el pivote y el
hilo, que el control no pone; es el escenario donde la pérdida ayudó más.*

Los videos completos: `videos/16_tres_escenarios_*.mp4` para las grillas y
`videos/13_control_vs_fisica_*.mp4` para los seis casos individuales (tres escenarios × condicionada y
libre).

<a id="degeneracion"></a>

## 7. Tercer experimento, la otra cara: cómo satisface la simetría

*Póster: sección «Resultados» (fuera de distribución) y «Conclusiones».*

Aprender la simetría no mejoró la física. En distribución nada se distingue del control (n = 60, todo
p > 0,29), y **fuera de distribución el modelo degenera**: su razón de movimiento cae a 0,12 contra
0,43 del control, por debajo de la de un video estático. La restricción no fija la escala del
movimiento, así que admite dos atajos.

**Atajo 1: moverse menos.** Si todo se achica, el desacuerdo entre las dos ramas se achica solo.

![Caída libre, la pelota se achica](gifs/caida_libre_encoge.gif)

**Atajo 2: partir el objeto en dos.** Este es el hallazgo más interesante. El estimador promedia el
flujo de toda la escena, así que dos objetos moviéndose en direcciones distintas se cancelan
parcialmente: el vector agregado se achica y la pérdida baja sin que la dinámica mejore.

![Pelota duplicada en el rebote](gifs/pelota_duplicada.gif)

*Checkpoint 1000. Dos pelotas de colores distintos en los paneles generados. Comparar con
`videos/04_rebote_una_pelota_paso250.mp4`, el mismo escenario 750 pasos antes.*

Archivos: `videos/03_rebote_pelota_duplicada_paso1000.mp4`, `videos/05_caida_libre_paso1000.mp4`,
`videos/07_fuera_de_dominio_rodando.mp4`

### Cómo se cerraría cada atajo

El primero, moverse menos, es una propiedad de la restricción: sin un ancla de escala siempre está
disponible.

El segundo es un defecto del **estimador**, y es más difícil de cerrar de lo que parece. La dificultad
de fondo es que las dos ramas son **generaciones independientes**: el modelo genera dos videos a partir
de dos condicionamientos distintos, no rota un video ya hecho. Nada garantiza que la pelota esté en la
misma fase ni en la posición correspondiente en ambos. Por eso:

- **Comparar los campos de flujo densos** píxel a píxel, $R\,\Phi_{\text{orig}}(x)$ contra
  $\Phi_{\text{rot}}(Rx)$, **no alcanza**: supone correspondencia espacial entre las dos generaciones.
  Si la pelota quedó en otra fase, la diferencia es enorme aunque la dinámica sea perfectamente
  equivariante. Confunde "está en otro lugar" con "se mueve distinto".
- **Quedarse con el objeto más grande** tampoco: es esencialmente lo que ya hace el agregador, no está
  justificado, y se cae apenas hay varios objetos o cuando el movimiento del fondo es informativo, que
  además hoy se descarta al restar el flujo medio global.

Las dos salidas que quedan:

1. **Comparar la distribución de velocidades** de las dos ramas (momentos, histograma o transporte
   óptimo) en vez de un promedio o de una comparación posición a posición. Es invariante a dónde esté
   cada objeto y a permutarlos, así que **no necesita correspondencia**, y tolera que las dos ramas
   generen distinta cantidad de objetos. Es la única de la lista que esquiva el problema de raíz.
2. **Emparejar las velocidades entre los dos videos generados**, con segmentación y asociación de
   identidad. Es lo que hace falta para cualquier comparación que no sea distribucional, y es un
   problema en sí mismo: hay que resolver correspondencia entre dos generaciones que pueden diferir en
   fase, en posición y hasta en cuántos objetos tienen. **Queda como trabajo futuro**, y es la pieza
   que hoy falta para que la restricción se pueda imponer sobre escenas con más de un objeto.

<a id="numeros"></a>

## 8. Los números, y cómo se eligió el checkpoint

*Póster: sección «Resultados».*

**El problema de elegir.** Las métricas de trayectoria oscilan entre checkpoints (ver la última tabla),
así que elegir para cada brazo el que mejor le va en la métrica de interés inflaría el resultado: sería
seleccionar sobre el desenlace.

**La elección primaria es el checkpoint final** (paso 1000): es la preregistrada y no mira ningún
resultado. Es la única defendible acá, y conviene decir por qué las alternativas no lo son.

Se intentó también elegir *el mejor por validación de difusión*, un criterio independiente de lo que se
quiere probar. Los dos brazos caen en el paso 250. Pero esa elección **no es sólida**, por tres razones:

- La validación se mide cada 50 pasos y los checkpoints se guardan cada 125, así que sólo **4 de los 8**
  checkpoints tienen un valor exacto (250, 500, 750, 1000). La regla nunca pudo rankear la otra mitad.
- El mínimo real de la validación está en el **paso 650**, que no quedó guardado como checkpoint.
- La diferencia entre el 250 (0,0698) y el 1000 (0,0724) es de 3,7 %, mientras que el desvío de la
  validación a lo largo del entrenamiento es del 8,6 % y el salto típico entre mediciones consecutivas
  es mayor que esa diferencia. Está **dentro del ruido**.

Lo que sí juega a favor de esa regla: el *ranking* por validación de difusión es idéntico en los dos
brazos (250, 1000, 500, 750), así que la comparación sigue apareada — los dos se leen en el mismo paso.

**La tercera opción, y la más interesante: elegir por lo que el brazo realmente minimiza.** Si lo que
se quiere es el mejor modelo *de la pérdida física*, la regla natural es la suma de los dos términos de
validación, que es el objetivo de entrenamiento. Estos son los checkpoints donde hay validación y
evaluación a la vez:

| paso | validación de difusión | validación de rotación (MSE crudo) | suma (dif + λ·rot) | movimiento medio |
|---|---|---|---|---|
| 250 | **0,0698** | 11,2507 | 0,1227 | — |
| 500 | 0,0808 | 10,3254 | 0,1293 | 0,987 |
| **750** | 0,0846 | **6,7278** | **0,1162** | **0,853** |
| 1000 | 0,0724 | 12,8185 | 0,1326 | 1,030 |

![Elegir por la pérdida elige el que menos se mueve](figuras/seleccion_checkpoint.png)

*Izquierda: los dos términos de validación y su suma, que es el objetivo de entrenamiento; el mínimo
cae en el paso 750. Derecha: cuánto se mueve el video generado en la evaluación, contra el ground
truth; el mínimo cae en el mismo paso. Reproducible con
`scripts_figuras/gen_seleccion_checkpoint.py`.*

La suma elige el **paso 750**. Y el 750 es, de los siete checkpoints con evaluación, **el que menos se
mueve** (0,853 contra 0,90–1,03 en todos los demás). No es casualidad: la validación de rotación es un
MSE crudo en px²/cuadro², no el cociente normalizado que se optimiza, así que baja cuando el video se
mueve menos, y la suma hereda ese defecto.

**Cada brazo en su propio óptimo.** Si el criterio es "cada método en su mejor punto", cada brazo se
elige por *su* objetivo de validación: el de física por la suma (paso 750) y el control por su
validación de difusión, que es todo lo que entrena (paso 250, con 0,0667). Los dos toman un máximo
sobre los mismos cuatro checkpoints, así que la ventaja de selección es simétrica. Sobre los mismos 30
clips held-out:

| métrica | control (250) | con física (750) | gana | p |
|---|---|---|---|---|
| error de velocidad (↓) | 4,838 | 5,225 | 10/30 | **0,184** |
| jerk (↓) | 1,592 | **1,101** | 22/30 | 0,004 |
| MAE de aceleración (↓) | 1,746 | 1,755 | 15/30 | 0,968 |
| pérdida de equivarianza cruda (↓) | 3,649 | **1,441** | 28/30 | &lt;0,001 |

En la métrica principal no hay diferencia (p = 0,18), y por escenario tampoco (0,064 · 0,77 · 0,28).
Las dos que sí dan significativas —jerk y la pérdida de equivarianza— son exactamente las dos que
**gana un video quieto**, así que apuntan a la ruta degenerada, no a física aprendida.

**Las tres reglas dan la misma conclusión, y la única que da significativo es la que elige mal.**

| regla | control | con física | p |
|---|---|---|---|
| preregistrada, sin selección (1000 contra 1000) | 4,619 | 4,900 | 0,213 |
| mismo paso, en el óptimo de la pérdida (750 contra 750) | 4,212 | 5,225 | 0,003 |
| cada uno en su óptimo (250 contra 750) | 4,838 | 5,225 | 0,184 |

La fila del medio es la que hay que leer con cuidado: compara contra el control **en el punto donde el
control está en su mejor momento** (4,212 es su mínimo de los ocho), así que exagera la brecha por los
dos lados. Es además una observación post hoc sobre una tabla de 8 checkpoints × 3 métricas: con
Bonferroni el umbral son 0,0021 y ese p = 0,003 no lo pasa. Con el control elegido por su propio
criterio, la diferencia se va.

Lo que sobrevive a las tres lecturas es el hecho de partida: el checkpoint que la pérdida señala como
su óptimo es también el que menos se mueve. Y eso no se apoya sólo en estos cuatro puntos — el mismo
mecanismo está medido aparte, sobre los cuatro brazos del barrido de la sección 12, donde la métrica
cruda ordena al revés que la normalizada.

### ¿Y qué tan bien medimos el error de velocidad?

La pregunta es obligada: si el modelo a veces genera dos pelotas, puede que el error no mida física
sino que el instrumento se rompe. Tres comprobaciones.

**No lo arruinan unos pocos clips.** Descartando el peor clip de cada brazo, después los dos peores, y
así hasta cinco, la brecha entre brazos queda igual: +0,39 · +0,35 · +0,32 · +0,34 · +0,41. La
**mediana** de la diferencia por clip (+0,65) es incluso mayor que la media (+0,39). El brazo con
física está parejamente un poco peor, no arrastrado por dos clips rotos. Y la brecha más grande no está
en rebote, que es donde aparece la pelota duplicada, sino en caída libre.

**El número sí distingue.** Contra la cota de un video congelado, el modelo base queda pegado a ella
(1,01 a 1,15× mejor, o sea prácticamente indistinguible de no moverse) mientras que los dos brazos
afinados están a 1,5-2,8×. La métrica separa un modelo que hace algo de uno que no.

**Pero está dominado por cuánto se mueve el modelo**, y ahí sí hay una advertencia seria:

| brazo | escenario | error de velocidad | razón de movimiento |
|---|---|---|---|
| control (250) | péndulo | 3,74 | 0,65 |
| con física (750) | péndulo | **3,48** | **0,88** |
| control (250) | caída libre | **4,61** | **0,66** |
| con física (750) | caída libre | 5,58 | 0,57 |
| control (250) | rebote | **6,16** | 0,51 |
| con física (750) | rebote | 6,62 | 0,48 |

Todos los brazos se mueven **menos** que el ground truth, y el único escenario donde el brazo con
física gana es el único donde se mueve más que el control. El error de velocidad y la razón de
movimiento no son independientes: la duplicación de la pelota actúa justamente por ahí, porque el
agregador promedia el flujo de toda la escena y dos objetos en direcciones opuestas se cancelan, con lo
que la rapidez estimada baja y el error contra el ground truth sube. El mecanismo existe; lo que los
datos dicen es que no es lo que explica la diferencia entre brazos.

**Una advertencia sobre caída libre.** En la ventana que se evalúa el ground truth ya está en velocidad
terminal, así que la cota de "extrapolar velocidad constante" da **0,00**: la respuesta correcta es una
recta. Cualquier modelo es infinitamente peor que lo trivial ahí, y ese escenario no debería pesar en
ninguna conclusión sobre física aprendida.

**Lo que arreglaría el instrumento** es medir la cinemática **por objeto** en vez de agregando sobre la
escena, que es el primer punto de la sección 11.

**Qué habría que cambiar para la próxima corrida.** La validación corre cada 50 pasos y los checkpoints
se guardan cada 125: sólo coinciden en cuatro puntos, así que la mitad de los checkpoints nunca pudo
entrar en ninguna regla. Alcanza con alinear las dos cadencias. Y hay que registrar en validación la
cantidad **normalizada** con el piso restado, además de la energía del movimiento, para que la regla no
se pueda ganar generando menos.

**Por qué no se puede elegir por la validación de equivarianza**, que sería lo natural dado lo que se
quiere probar. Por dos razones distintas, y las dos son decisivas:

- **El control no la tiene.** Con λ_rot = 0 el término nunca se calcula, así que su registro es 0,0000
  en los 1000 pasos. Una regla que sólo se puede evaluar en un brazo no se puede aplicar a los dos, y
  comparar brazos elegidos con criterios distintos no es comparar.
- **Es un MSE sin normalizar**, así que premia generar menos movimiento. Lo verificamos sobre los cuatro
  brazos del barrido de ventanas de la sección 12: esa métrica los ordena **al revés** que el cociente
  normalizado que es lo que efectivamente se minimiza.

Así que el paso 250 se reporta abajo como control de sensibilidad, no como "el mejor modelo". Lo
relevante es que **las dos elecciones dan la misma respuesta**: nada se distingue del control.

### Checkpoint final (paso 1000), en distribución

| métrica | control | con física | mejor | p |
|---|---|---|---|---|
| error de velocidad (↓) | 4,895 | 5,009 | 24/60 | 0,299 |
| razón de movimiento (→1) | 0,616 | 0,598 | 29/60 | 0,802 |
| jerk (↓) | 1,640 | 1,495 | 35/60 | 0,067 |
| MAE de aceleración (↓) | 1,828 | 1,880 | 24/60 | 0,126 |

Cota del video quieto, error de velocidad por escenario: 11,498, 7,519, 9,953.

### Checkpoint final (paso 1000), fuera de distribución: tiro vertical

| métrica | control | con física | cota del video quieto | p |
|---|---|---|---|---|
| error de velocidad (↓) | 7,634 | **8,799** | 8,899 | 0,005 |
| razón de movimiento (→1) | 0,427 | **0,119** | 0,216 | 0,001 |
| jerk (↓) | 1,651 | **0,817** | 0,000 | 0,005 |

### Control de sensibilidad: paso 250, en distribución

| métrica | control | con física | mejor | p |
|---|---|---|---|---|
| error de velocidad (↓) | 4,838 | 5,190 | 13/30 | 0,096 |
| razón de movimiento (→1) | 0,608 | 0,568 | 16/30 | 0,213 |
| jerk (↓) | 1,592 | 1,271 | 18/30 | 0,164 |
| pérdida de equivarianza cruda (↓) | 3,649 | 2,277 | 25/30 | &lt;0,001 |

### Por checkpoint y por escenario

![Error y movimiento por checkpoint, abierto por escenario](figuras/por_checkpoint_y_escenario.png)

**El promedio escondía el resultado más interesante.** Agrupando los tres escenarios, el brazo con
física no se distingue del control (p = 0,299). Abriendo por escenario, en el checkpoint final los
efectos son opuestos y significativos:

| escenario | error de velocidad (control → física) | p | razón de movimiento | p |
|---|---|---|---|---|
| caída libre | 4,274 → **3,234** | 0,001 | 0,695 → **0,797** | &lt;0,001 |
| péndulo | 4,325 → 4,825 | 0,044 | 0,621 → 0,588 | 0,522 |
| rebote | 6,084 → **6,967** | 0,007 | 0,533 → **0,410** | 0,015 |

En **caída libre** el brazo con física es mejor: se acerca más al ground truth y además **se mueve
más**, o sea que no es degeneración. En **rebote** es peor y se mueve menos, que es exactamente donde
aparece la pelota duplicada. El péndulo queda en el medio. Con seis tests, una corrección de
Bonferroni deja en pie la mejora en caída libre y el empeoramiento en velocidad del rebote.

La lectura que esto sugiere: la pérdida ayuda donde la cinemática es simple y constante (caída libre
tras la velocidad terminal, que es movimiento uniforme) y estorba donde hay impactos, que es donde el
estimador se rompe y el modelo encuentra el atajo de partir el objeto. Es un resultado **post-hoc**:
no estaba preregistrado por escenario, aunque la metodología sí exige reportar los escenarios por
separado y nunca promediados.

### Todos los checkpoints, para que se vea que no hay tendencia

Formato: control → con física (p). Apareado por clip, n=30.

| paso | error de velocidad | razón de movimiento | jerk |
|---|---|---|---|
| 125 | 4,915 → 5,166 (0,328) | 0,604 → 0,598 (1,000) | 1,588 → 1,332 (0,031) |
| 250 | 4,838 → 5,190 (0,096) | 0,608 → 0,568 (0,213) | 1,592 → 1,271 (0,164) |
| 375 | 4,913 → 5,312 (0,050) | 0,599 → 0,609 (0,598) | 1,728 → 1,184 (0,001) |
| 500 | 4,540 → 4,365 (0,792) | 0,682 → 0,799 (0,017) | 1,857 → 1,473 (0,047) |
| 625 | 5,299 → 4,326 (0,080) | 0,570 → 0,726 (&lt;0,001) | 1,565 → 1,532 (0,339) |
| 750 | 4,212 → 5,225 (0,003) | 0,715 → 0,645 (0,105) | 1,820 → 1,101 (&lt;0,001) |
| 875 | 4,939 → 4,767 (0,465) | 0,592 → 0,617 (0,393) | 1,546 → 1,256 (0,280) |
| 1000 | 4,619 → 4,900 (0,213) | 0,651 → 0,613 (0,452) | 1,907 → 1,503 (0,253) |

Todas las evaluaciones son apareadas por clip: los dos brazos generan **los mismos clips held-out** con
la misma semilla, y el test es un Wilcoxon sobre las diferencias por clip. Las tablas completas, con
todos los escenarios y todas las métricas, están en el repositorio principal del proyecto.

<a id="ood"></a>

## 9. Fuera de dominio: las corrupciones aparecen con los pasos

*Póster: sección «Resultados», fila fuera de distribución.*

Los prompts de abajo nunca se entrenaron. Es donde mejor se ve el mecanismo: no es que el modelo
"aprenda mal la física", es que **la imagen se corrompe** a medida que avanza el entrenamiento, y la
corrupción es justamente lo que baja la pérdida.

![Rodando, evolución con los pasos](gifs/ood_rolling.gif)

*Prompt «una pelota rodando por una superficie plana», el mismo en los cuatro paneles, generado por el
modelo entrenado en distintos pasos. En el 250 y el 500 hay una pelota limpia; en el 750 aparecen
**dos**; en el 1000 la pelota está deshecha. La pérdida no penaliza nada de eso: dos objetos que se
mueven en direcciones distintas se cancelan en el flujo agregado, y el desacuerdo entre ramas baja.*

![Péndulo fotográfico, evolución con los pasos](gifs/ood_pendulum_photo.gif)

*Mismo efecto con otro prompt fuera de dominio.*

![Fuera de dominio: control contra brazo con física](gifs/ood_control_vs_fisica.gif)

*Y contra el control en el mismo paso: el control mantiene un objeto coherente.*

En números, sobre el escenario fuera de distribución con ground truth (tiro vertical): la razón de
movimiento del brazo con física cae a 0,12 contra 0,43 del control, por debajo de la de un video
estático (0,22). Es el mismo fenómeno, medido.

**Por qué esto apunta a una limitación del método y no a la hipótesis.** La pérdida se calcula sobre
una generación truncada, de 12 pasos de Euler en vez de 20, y sobre una parte de los 32 vectores de
velocidad del clip: 24 en péndulo y rebote, 12 o 16 en caída libre (contado sobre los 1000 pasos de la
corrida). El modelo puede degradar lo que la pérdida no mira. Conviene decir hasta dónde llega esta
explicación: **no está medido** que la corrupción viva en los cuadros excluidos, y la degeneración más
fuerte aparece fuera de distribución, donde la pérdida no vio ningún vector. Lo que sí está medido son
los dos atajos de la sección 7, que operan sobre cuadros que la pérdida **sí** mira. Las correcciones
de la última sección apuntan a las dos cosas.

## 10. Control: la simetría por datos tampoco enseña

*Póster: no aparece; el contraste quedó confundido y se reporta como pendiente.*

Entrenar con clips rotados y trasladados (aumentación de datos) en vez de imponer la restricción
tampoco produce un modelo más equivariante. El contraste no es limpio, porque el brazo sin
aumentaciones agrega además un objetivo de texto a video, así que el dato queda como indicio y no como
resultado.

Archivos: `videos/10_equivarianza_*_ablacion_aumentaciones.mp4`

## 11. Qué quedó como recomendación

*Póster: sección «Trabajo futuro».*

1. **Que la pérdida vea todo lo que el modelo genera**: hoy mira 24 de 32 vectores en péndulo y
   rebote, 12 o 16 en caída libre, y una generación de 12 pasos de Euler en vez de 20. Es una hipótesis
   razonable —no comprobada— que la corrupción se aloje en lo que queda fuera.
2. **Emparejar las velocidades entre las dos generaciones**, que es la pieza que falta para escenas con
   más de un objeto. La única alternativa que esquiva el emparejamiento es comparar la **distribución**
   de velocidades, invariante a posición y a permutar objetos.
3. **Anclar la escala del movimiento**, porque sin eso la restricción siempre admite el atajo de
   moverse menos. El costo es que deja de ser una restricción sin ground truth, que era el atractivo.

## 12. Cuántos pasos de Euler hay que retropropagar (la ventana de BPTT)

*Póster: sección «Análisis del gradiente».*

La pérdida se calcula sobre un video que el modelo genera **dentro** del paso de entrenamiento, con 12
pasos de Euler. Retropropagar por los doce cuesta memoria y tiempo, así que la pregunta práctica es si
alcanza con una ventana. Ese parámetro es `physics_n_bptt` (cuántos pasos llevan gradiente) junto con
`physics_bptt_steps` (cuáles).

**Lo que está medido y no depende de la pérdida elegida:**

- **El perfil del gradiente por paso de Euler es en U**, no concentrado al principio: los primeros
  cuatro pasos aportan un 33-34 % de la norma y los últimos cuatro un 47-48 %, con el máximo en el
  paso 11. Verificado sobre dos corridas independientes (656 y 538 pasos), coincidiendo dentro de un
  punto porcentual.
- **El BPTT completo cuesta un 29 % más por paso** que cualquier ventana truncada (27,8 s/paso contra
  21,6-21,7 en los brazos de abajo).

**Lo que el barrido no logra decidir.** Se corrieron cinco brazos de 150 pasos, idénticos salvo la
ventana, sobre la pérdida arreglada y la cantidad que sí tiene señal. El resultado es que **ninguna
ventana se distingue de las otras** sobre lo que efectivamente se minimiza, y que los instrumentos
disponibles se contradicen entre sí:

| brazo | pasos retropropagados | val_rot cruda | ρ = N/D apareado | energía de movimiento | s/paso |
|---|---|---|---|---|---|
| completo | los 12 | **10,31** (el mejor) | el peor de los cuatro | referencia | 27,8 |
| ventana4 | 2, 3, 8, 11 | 11,17 | −0,014 (p = 0,21) | +2,2 % | 21,7 |
| cola4 | 8, 9, 10, 11 | 13,59 | −0,012 (p = 0,15) | −0,9 % | 21,6 |
| mejor6 | 0, 1, 2, 4, 5, 6 | **14,66** (el peor) | −0,020 (p = 0,005), el mejor | +9,6 % | 23,4 |

Los dos instrumentos ordenan los brazos **al revés**, y la explicación es la energía del movimiento:
el brazo que más se mueve gana en el cociente normalizado y pierde en el error crudo. Al restar el piso
de RAFT —o sea, mirando exactamente la cantidad que se optimiza— la diferencia entre `mejor6` y el
completo se va a cero (+0,0000, p = 0,66), y lo mismo pasa si se comparan sólo los pasos con energía
pareja al 5 % (p = 0,57).

**Conclusión honesta:** con 150 pasos por brazo y sin guardar pesos, este barrido no ordena ventanas.
Lo que sí deja es la advertencia metodológica de la sección 8: ninguna de las métricas internas sirve
para comparar brazos sin controlar por cuánto se mueve el video. Para decidir la ventana haría falta
guardar checkpoints y evaluar el error de trayectoria contra el ground truth, que es la única métrica
que no se puede ganar moviéndose más o menos.

## 13. Parámetros

Todo lo de abajo sale de `config_resolved.json` de la corrida, publicado en
[`resultados/configs/`](resultados/configs).

| | |
|---|---|
| Modelo base | SANA-Video 2B, 480p (`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`) |
| Adaptación | LoRA rango 32, alfa 64, sobre `to_q`, `to_k`, `to_v`, `to_out.0` |
| Optimización | lr 1e-4, batch 1, bf16, gradient checkpointing, 1000 pasos, semilla 42 |
| Datos | 500 clips sintéticos propios, 384×384, **33 cuadros**, con aumentaciones; validación de 20 clips fija (`split_seed` 42) |
| Objetivo | difusión + condicionado (λ = 1,5) + equivarianza rotacional (λ_rot = 4,7e-03) |
| Términos apagados | boost galileano, traslación, ground truth, flow matching y jerk, todos en λ = 0 |
| Equivarianza | rotación de 45°, sobre **velocidades**, cociente invariante a escala con piso de RAFT restado y margen mínimo 3,0 |
| Generación dentro del paso | 12 pasos de Euler, condicionamiento de 2 latentes, decodificación troceada de a 1 latente |
| BPTT | los 12 pasos (ver sección 12) |
| Evaluación | 20 pasos de Euler, 33 cuadros, 384 px, 10 clips held-out por escenario, rotación de 45° |
| Costo | L40S (AWS g6e.xlarge), 21,3 GB de VRAM en la mediana (pico 28,5), 46,7 s/paso, 16 h por corrida |

**Cómo se calibra λ_rot.** No se elige a mano: se corren sondas de 10 pasos midiendo la razón entre la
norma del gradiente físico y la del gradiente de difusión, y se ajusta λ hasta que esa razón caiga en
[0,40; 0,60], o sea que el término físico pese la mitad que el objetivo generativo. Para esta corrida
hicieron falta cinco sondas y quedó en 4,7e-03. **Cualquier cambio en la fórmula de la pérdida invalida
la calibración anterior** y obliga a repetir las sondas.

**Un límite del montaje que conviene tener presente.** El período del péndulo es de 37,4 cuadros y
generamos 33: nunca se ve una oscilación completa, ni en entrenamiento ni en evaluación.

---

Alexander Bodner y Mateo Costantini, Universidad de San Andrés, Visión Artificial Avanzada.
Modelo base: SANA-Video 2B. Flujo óptico: RAFT. Los clips del simulador son sintéticos y propios.
