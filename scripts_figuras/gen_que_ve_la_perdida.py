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
    escala = 40 / (np.median([np.linalg.norm(v) for v in vels]) + 1e-6)
    for t, fl in enumerate(flujos):
        z = np.zeros((alto, ancho, 3), np.uint8)
        z[BARRA:, :W] = cv2.cvtColor(u8[t], cv2.COLOR_RGB2BGR)
        z[BARRA:, W+SEP:2*W+SEP] = flujo_a_color(fl)
        # tercer panel: el cuadro en gris con el vector agregado dibujado desde el centro
        gris = cv2.cvtColor(cv2.cvtColor(u8[t], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)//2
        v = vels[t]; c0 = (W//2, H//2)
        c1 = (int(c0[0] + v[0]*escala), int(c0[1] + v[1]*escala))
        cv2.arrowedLine(gris, c0, c1, AMARILLO, 3, tipLength=0.25)
        z[BARRA:, 2*(W+SEP):] = gris
        for x, txt, col in ((8, "1. el video", BLANCO),
                            (W+SEP+8, "2. flujo de RAFT", VERDE),
                            (2*(W+SEP)+8, "3. lo unico que ve la perdida", AMARILLO)):
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
