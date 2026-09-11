"""Cuanta direccion del gradiente conserva truncar la ventana de BPTT.

El entrenador registra, por paso, el aporte exacto de cada paso de Euler al
gradiente de la perdida fisica respecto de los pesos: su norma (`v_grad_norm`) y
el coseno contra cada uno de los otros (`cos_sims`). Con eso se reconstruye la
matriz de Gram y se puede calcular, para cualquier subconjunto de pasos, el
coseno entre la suma del subconjunto y la suma completa. Eso es exactamente
"cuanto de la direccion verdadera retiene esa ventana".

Importa medirlo sobre la corrida que se reporta: las cifras viejas salian de las
corridas sobre aceleracion, que es la cantidad que despues resulto ser ruido.

Uso:  python3 scripts/coseno_truncamiento_bptt.py <training_log.jsonl>
"""
import itertools
import json
import math
import sys

import numpy as np

VENTANAS = {
    "BPTT completo (12)": list(range(12)),
    "no contigua [2,3,8,11]": [2, 3, 8, 11],
    "mejor-6 [0,1,2,4,5,6]": [0, 1, 2, 4, 5, 6],
    "cola de 4 (DRaFT-K)": [8, 9, 10, 11],
    "primeros 4": [0, 1, 2, 3],
    "solo el ultimo": [11],
    "solo el primero": [0],
}


def gram(entradas):
    k = len(entradas)
    normas = np.array([e["v_grad_norm"] for e in entradas], float)
    cos = np.eye(k)
    for e in entradas:
        i = e["bptt_step"]
        for j, v in (e.get("cos_sims") or {}).items():
            cos[i, int(j)] = cos[int(j), i] = v
    return np.outer(normas, normas) * cos


def coseno(G, sub):
    todos = list(range(G.shape[0]))
    num = G[np.ix_(sub, todos)].sum()
    a, b = G[np.ix_(sub, sub)].sum(), G.sum()
    return None if a <= 0 or b <= 0 else num / math.sqrt(a * b)


def main(log):
    acumulado = {k: [] for k in VENTANAS}
    primeros, ultimos = [], []
    for linea in open(log):
        entradas = json.loads(linea).get("bptt_grad_norms")
        if not entradas or len(entradas) != 12:
            continue
        G = gram(entradas)
        if G.sum() <= 0:
            continue
        for nombre, sub in VENTANAS.items():
            c = coseno(G, sub)
            if c is not None:
                acumulado[nombre].append(c)
        n = np.array([e["v_grad_norm"] for e in entradas])
        if n.sum() > 0:
            primeros.append(n[:4].sum() / n.sum())
            ultimos.append(n[-4:].sum() / n.sum())

    n_pasos = len(acumulado["BPTT completo (12)"])
    print(f"n = {n_pasos} pasos con gradiente fisico\n")
    print(f"{'ventana':26s} {'media':>8s} {'mediana':>9s}")
    for nombre in VENTANAS:
        v = np.array(acumulado[nombre])
        print(f"{nombre:26s} {v.mean():8.3f} {np.median(v):9.3f}")
    print(f"\nnorma: primeros 4 pasos {np.mean(primeros) * 100:.0f} %, "
          f"ultimos 4 {np.mean(ultimos) * 100:.0f} %")


if __name__ == "__main__":
    main(sys.argv[1])


def carga_gram(log):
    """Todas las matrices de Gram del log, una por paso de entrenamiento."""
    Gs = []
    for linea in open(log):
        e = json.loads(linea).get("bptt_grad_norms")
        if not e or len(e) != 12:
            continue
        G = gram(e)
        if G.sum() > 0:
            Gs.append(G)
    return np.stack(Gs)


def direccion_por_tamano(log, k_max=12):
    """Cuanta direccion explica la mejor ventana de cada tamano, fuera de muestra.

    La ventana se elige maximizando el coseno medio sobre la PRIMERA mitad de los
    pasos y se reporta sobre la SEGUNDA. Elegirla y medirla en los mismos datos da
    un optimo dentro de muestra, que no es una afirmacion sobre nada.

    Devuelve {k: (ventana, coseno fuera de muestra, coseno de la cola de k)}.
    """
    G = carga_gram(log)
    mitad = len(G) // 2
    subs = [s for k in range(1, k_max + 1) for s in itertools.combinations(range(12), k)]
    V = np.zeros((len(subs), 12))
    for a, s in enumerate(subs):
        V[a, list(s)] = 1

    def coseno_medio(Gx):
        num = Gx.sum(2) @ V.T
        quad = np.einsum("si,pij,sj->ps", V, Gx, V)
        return num / np.sqrt(np.maximum(quad, 1e-30) * Gx.sum((1, 2))[:, None])

    dentro, fuera = coseno_medio(G[:mitad]).mean(0), coseno_medio(G[mitad:]).mean(0)
    out = {}
    for k in range(1, k_max + 1):
        idx = [a for a, s in enumerate(subs) if len(s) == k]
        mejor = idx[int(np.argmax(dentro[idx]))]
        cola = subs.index(tuple(range(12 - k, 12)))
        out[k] = (list(subs[mejor]), float(fuera[mejor]), float(fuera[cola]))
    return out
