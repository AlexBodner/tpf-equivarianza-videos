# Qué queda abierto

Lista de trabajo, no de resultados. Sale de una auditoría hecha el 9 de septiembre de 2026 contra los
JSON crudos, más lo que fuimos encontrando al escribir. Está ordenada por lo que más cambiaría el
trabajo si se resolviera.

## Cosas que no podemos explicar hoy

**El quinto brazo del barrido de BPTT.** El prerregistro lo llamaba «magnitud compensada» y lo señalaba
como la única comparación legible del barrido: la misma ventana de seis pasos que `mejor6` pero con λ
re-escalado para igualar el gradiente del BPTT completo. Eso no pasó: los cinco brazos corrieron con el
mismo λ = 0,0047, y su gradiente físico (0,0065) queda lejos del completo (0,0183). Además, según el
código, cuando se especifica la lista de pasos ésta le gana al parámetro que supuestamente los
diferencia, así que los dos brazos deberían ser idénticos en la rama física y no lo son. La
configuración exacta está en el volumen de la instancia detenida (`checkpoints_sana/bpttvel_*/config_resolved.json`).

**Qué mide `l_rot_norm` cuando satura.** La cota del video congelado da exactamente cero en 45 de los 88
clips del test final, y el modelo base en más todavía. Cuando el desacuerdo cae por debajo del piso del
estimador, el numerador se recorta y la métrica deja de distinguir. Hoy lo esquivamos apoyándonos en la
razón de movimiento, pero convendría una versión que no sature.

## Mediciones que faltan

**El alcance del fine-tuning.** Generar con prompts que no tienen nada que ver con el dominio (un perro
corriendo, una pelota en un parque) con los tres brazos, incluido el modelo base sin LoRA. Mide si la
adaptación destruyó los priors generales o si fue lo bastante chica como para no tocarlos. Los pesos
están en Hugging Face, así que no depende de la instancia detenida.

**Fuera de dominio con varias semillas.** Todo lo que tenemos de generación fuera de dominio es de una
generación por celda, y medimos que con tres semillas la saturación del mismo checkpoint va de 26 a 136
y el número de objetos de 0 a 9. Con una sola muestra no hay señal que leer. Cualquier afirmación sobre
esto necesita repetir.

**Equivarianza por checkpoint con clips independientes.** La serie de ocho checkpoints se midió sobre
tres escenas con sus cuatro variantes aumentadas, no sobre doce clips distintos, así que sus p no valen.
La evidencia fuerte está en el test final (72 de 88 clips); rehacer la serie con clips distintos la
convertiría en una curva citable.

**Test final en el checkpoint preregistrado.** El test corrió con la regla «cada brazo en su óptimo»
(control 125, física 875), que se fijó después del prerregistro. Correrlo también en 1000 contra 1000,
que es la elección que no selecciona nada, cerraría la duda de si la regla favorece a alguno.

## Problemas del instrumental

**El volcado por clip del evaluador.** En `run_eval.py`, la lista `clips` se llena antes del `continue`
que saltea los clips demasiado cortos, así que en el test final hay 30 nombres y 28 valores por métrica.
Los promedios y los tests están bien (los dos brazos saltean los mismos clips), pero cualquier análisis
que asocie un valor a un nombre de clip queda corrido.

**Cadencia de validación y guardado.** La validación corría cada 50 pasos y los checkpoints se guardaban
cada 125: sólo cuatro de los ocho tenían medición hasta que los revalidamos a mano. Alinear las dos
cadencias es gratis y evita todo ese trabajo.

**Registrar en validación la cantidad normalizada.** Hoy `validate()` guarda un MSE crudo, que se puede
bajar generando menos movimiento. Registrar el cociente con el piso restado, y la energía del
movimiento, haría que la regla de selección no se pueda ganar quedándose quieto.

## Cosas de escritura

- El bloque «Diagnóstico» del póster sigue escribiendo la pérdida con notación de aceleración, que es
  correcto porque describe el desvío, pero conviene decirlo explícitamente para que no choque con el
  bloque de método, que ya está sobre velocidades.
- El 13 % de los pasos de entrenamiento no aportan gradiente físico (96 porque el desacuerdo cae bajo el
  piso, 28 por falta de pares de RAFT, 7 por el margen). No está dicho en ningún lado.
- La tabla de generaciones a 65 cuadros es n = 1 o n = 2 por celda y no lo declara.
