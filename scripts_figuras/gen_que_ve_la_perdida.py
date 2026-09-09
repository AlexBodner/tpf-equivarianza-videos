"""Video explicativo: qué ve la pérdida cuando mira un clip.

Tres paneles: el video, el campo de flujo que devuelve RAFT, y el único vector que
queda después de agregar la escena — que es la cantidad sobre la que se impone la
simetría. La idea es que se vea que la pérdida no mira píxeles, mira UN vector por
cuadro.
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np, cv2, torch

sys.path.insert(0, ".")
from raft_pipeline.wrapper import RAFTWrapper
from raft_pipeline.kinematics import aggregate_flow

BARRA, SEP, FPS = 36, 4, 8
BLANCO, VERDE, AMARILLO = (255,255,255), (140,255,170), (80,230,255)

def leer(p, n=33):
    c = cv2.VideoCapture(str(p)); fr=[]
    while len(fr) < n:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)

def flujo_a_color(fl):
    """Codificación estándar: el tono es la dirección y el brillo la magnitud."""
    mag, ang = cv2.cartToPolar(fl[...,0], fl[...,1])
    hsv = np.zeros((*fl.shape[:2], 3), np.uint8)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    clip = Path(sys.argv[1]); salida = Path(sys.argv[2])
    u8 = leer(clip)
    V = torch.from_numpy(u8).permute(0,3,1,2).float()
    raft = RAFTWrapper(variant="small")
    raft.model.eval()
    with torch.no_grad():
        flujos = [raft.extract_flow(V[i:i+1], V[i+1:i+2])[0].permute(1,2,0).numpy()
                  for i in range(len(V)-1)]
        vels = [aggregate_flow(torch.from_numpy(f).permute(2,0,1)[None]).squeeze().numpy()
                for f in flujos]
    print(f"{len(flujos)} pares de cuadros, {len(vels)} vectores agregados")

    H, W = u8.shape[1:3]
    alto, ancho = H + BARRA, W*3 + SEP*2
    tmp = Path(tempfile.mkdtemp())/"crudo.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ancho, alto))
    escala = 40.0   # 40 px de dibujo = 1 px/cuadro de velocidad
    for t, fl in enumerate(flujos):
        z = np.zeros((alto, ancho, 3), np.uint8)
        z[BARRA:, :W] = cv2.cvtColor(u8[t], cv2.COLOR_RGB2BGR)
        col = flujo_a_color(fl)
        paso_flecha = 24
        for y in range(paso_flecha//2, H, paso_flecha):
            for x in range(paso_flecha//2, W, paso_flecha):
                dx, dy = fl[y, x] * 3
                if dx*dx + dy*dy > 1:
                    cv2.arrowedLine(col, (x, y), (int(x+dx), int(y+dy)),
                                    (255,255,255), 1, tipLength=0.35)
        z[BARRA:, W+SEP:2*W+SEP] = col
        # Tercer panel: un dial. El vector agregado NO tiene ubicacion en la escena
        # —es el promedio de todo— asi que dibujarlo sobre la imagen hacia pensar que
        # algo en el centro se movia. Acá va sobre ejes, con la estela de los cuadros
        # anteriores para que se vea cómo evoluciona.
        dial = np.full((H, W, 3), 18, np.uint8)
        c0 = (W//2, H//2)
        for r_px, etq in ((40, "1"), (80, "2"), (120, "3")):
            cv2.circle(dial, c0, r_px, (55,55,55), 1)
            cv2.putText(dial, etq, (c0[0]+r_px-10, c0[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110,110,110), 1, cv2.LINE_AA)
        cv2.line(dial, (0,c0[1]), (W,c0[1]), (55,55,55), 1)
        cv2.line(dial, (c0[0],0), (c0[0],H), (55,55,55), 1)
        for k in range(max(0, t-8), t):
            vk = vels[k]; f = (k - max(0, t-8) + 1) / 9
            cv2.circle(dial, (int(c0[0]+vk[0]*escala), int(c0[1]+vk[1]*escala)),
                       2, (int(60*f), int(140*f), int(160*f)), -1)
        v = vels[t]
        cv2.arrowedLine(dial, c0, (int(c0[0]+v[0]*escala), int(c0[1]+v[1]*escala)),
                        AMARILLO, 3, tipLength=0.22)
        cv2.putText(dial, "los circulos son 1, 2 y 3 px/cuadro", (10, H-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110,110,110), 1, cv2.LINE_AA)
        z[BARRA:, 2*(W+SEP):] = dial
        for x, txt, col in ((8, "1. el video", BLANCO),
                            (W+SEP+8, "2. flujo de RAFT", VERDE),
                            (2*(W+SEP)+8, "3. el unico vector que ve la perdida", AMARILLO)):
            cv2.putText(z, txt, (x, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.putText(z, f"cuadros {t+1} y {t+2}   |v| = {np.linalg.norm(v):.2f} px/cuadro",
                    (ancho-360, alto-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLANCO, 1, cv2.LINE_AA)
        vw.write(z)
    vw.release()
    salida.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(tmp),
                    "-c:v","libx264","-profile:v","baseline","-level","3.0",
                    "-pix_fmt","yuv420p","-movflags","+faststart",str(salida)], check=True)
    print(f"{salida}: {ancho}x{alto}")
