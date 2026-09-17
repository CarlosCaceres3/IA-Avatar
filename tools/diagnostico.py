"""Chequeo previo al evento.

Corre esto antes de montar el stand. Verifica en orden: paquetes, modelo,
camaras disponibles, velocidad real del pipeline y camara virtual. Guarda
un cuadro de muestra en diagnostico.png para ver que se esta produciendo.

    python tools/diagnostico.py
    python tools/diagnostico.py --camera 1 --frames 120
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2                                               # noqa: E402
import numpy as np                                       # noqa: E402


def check_packages():
    print("== Paquetes ==")
    ok = True
    for name in ("cv2", "mediapipe", "numpy"):
        try:
            mod = __import__(name)
            print("  OK  " + name + " " + str(getattr(mod, "__version__", "?")))
        except ImportError as exc:
            print("  --  " + name + " FALTA (" + str(exc) + ")")
            ok = False
    try:
        import pyvirtualcam
        print("  OK  pyvirtualcam " + str(getattr(pyvirtualcam, "__version__", "?")))
    except ImportError:
        print("  --  pyvirtualcam no instalado (la camara virtual no funcionara)")
    return ok


def check_model():
    print("== Modelo ==")
    path = os.path.join(ROOT, "models", "pose_landmarker_full.task")
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1e6
        print("  OK  pose_landmarker_full.task (" + str(round(mb, 1)) + " MB)")
        return path
    print("  --  falta el modelo. Corre: python tools/descargar_modelo.py")
    return None


def list_cameras(max_index=4):
    print("== Camaras ==")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print("  OK  camara " + str(i) + " -> " +
                      str(frame.shape[1]) + "x" + str(frame.shape[0]))
                found.append(i)
        cap.release()
    if not found:
        print("  --  ninguna camara respondio")
    return found


def check_virtualcam(width, height, fps):
    print("== Camara virtual ==")
    try:
        import pyvirtualcam
        from pyvirtualcam import PixelFormat
    except ImportError:
        print("  --  pyvirtualcam no instalado")
        return
    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                 fmt=PixelFormat.BGR, print_fps=False) as cam:
            print("  OK  dispositivo: " + str(cam.device))
    except Exception as exc:
        print("  --  no disponible: " + str(exc))
        print("      Instala OBS Studio para tener el driver OBS Virtual Camera.")


def run_pipeline(model_path, camera, frames, width, height, exposure=None):
    """Mide el rendimiento real de deteccion + render sobre la camara.

    Usa exactamente la misma configuracion que avatar_cam.py (MJPG, buffer
    corto, composicion limitada al ROI). Si midiera de otra forma, el numero
    que reporta aqui no seria el que van a ver en el stand.
    """
    print("== Rendimiento (" + str(frames) + " cuadros) ==")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    import avatar as av
    import skeleton as sk
    import stage
    from smoothing import OneEuroFilter

    state = {"landmarks": None, "hits": 0}

    def on_result(result, image, ts):
        if result.pose_landmarks:
            state["landmarks"] = result.pose_landmarks[0]
            state["hits"] += 1
        else:
            state["landmarks"] = None

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=on_result,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if not cap.isOpened():
        print("  --  no pude abrir la camara " + str(camera))
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    packs = av.load_packs(os.path.join(ROOT, "assets", "packs"))
    renderer = av.AvatarRenderer()
    backgrounds = stage.Backgrounds(os.path.join(ROOT, "assets", "backgrounds"), (w, h))
    smoother = OneEuroFilter()

    screen = np.empty((h, w, 3), np.uint8)
    out = None
    t0 = time.perf_counter()
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (480, int(480 * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((time.perf_counter() - t0) * 1000))

        bg = backgrounds.get(frame)
        if state["landmarks"] is not None:
            skel = sk.from_landmarks(state["landmarks"], w, h, smoother=smoother)
            canvas = renderer.render(skel, packs[0])
            out = stage.composite(bg, canvas, packs[0].theme.glow,
                                  packs[0].theme.glow_strength,
                                  roi=skel.bbox(), out=screen)
        else:
            np.copyto(screen, bg)
            out = screen
            stage.draw_banner(out, "Sin persona detectada")

    elapsed = time.perf_counter() - t0
    if exposure is not None:                 # devolver la camara a automatico
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        for _ in range(5):
            cap.read()
    cap.release()
    landmarker.close()

    fps = frames / elapsed if elapsed > 0 else 0
    print("  " + str(w) + "x" + str(h) + " -> " + str(round(fps, 1)) + " FPS")
    print("  cuadros con persona detectada: " + str(state["hits"]) + "/" + str(frames))
    if state["hits"] == 0:
        print("  (normal si no habia nadie frente a la camara)")

    if out is not None:
        path = os.path.join(ROOT, "diagnostico.png")
        cv2.imwrite(path, out)
        print("  muestra guardada en " + os.path.normpath(path))
    return w, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--exposure", type=float, default=None,
                   help="medir con exposicion manual, igual que avatar_cam.py")
    args = p.parse_args()

    check_packages()
    model = check_model()
    cams = list_cameras()

    camera = args.camera if args.camera is not None else (cams[0] if cams else None)
    size = None
    if model and camera is not None:
        size = run_pipeline(model, camera, args.frames, args.width, args.height,
                            args.exposure)

    w, h = size if size else (args.width, args.height)
    check_virtualcam(w, h, 30)
    print("== Fin ==")


if __name__ == "__main__":
    main()
