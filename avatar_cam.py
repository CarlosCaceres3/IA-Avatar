"""Avatar en vivo por camara - demo de stand.

Captura la camara, detecta la pose de la persona con MediaPipe y dibuja un
avatar articulado que imita sus movimientos. La persona real no aparece:
en pantalla solo se ve el avatar sobre el fondo elegido.

Uso:
    python avatar_cam.py
    python avatar_cam.py --camera 1 --width 1280 --height 720
    python avatar_cam.py --no-virtualcam

Teclas:
    A / D      avatar anterior / siguiente
    F          fondo siguiente
    G          espejo (on/off)
    E          esqueleto de depuracion (on/off)
    H          ocultar/mostrar los datos en pantalla
    V          activar/pausar la camara virtual
    P          guardar una foto PNG en capturas/
    TAB        pantalla completa
    Q o ESC    salir
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import mediapipe as mp                                   # noqa: E402
from mediapipe.tasks import python as mp_python          # noqa: E402
from mediapipe.tasks.python import vision                # noqa: E402

import avatar as av                                      # noqa: E402
import skeleton as sk                                    # noqa: E402
import stage                                             # noqa: E402
from camera_out import VirtualCamera                     # noqa: E402
from smoothing import Hysteresis, OneEuroFilter          # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Avatar IA - Festech"

# El detector no gana precision con mas resolucion (internamente reescala),
# pero si cuesta mas convertir el cuadro. Se le manda una version chica.
INFER_WIDTH = 480


class LatestResult:
    """Buzon del ultimo resultado del detector.

    En modo LIVE_STREAM MediaPipe corre en su propio hilo y avisa por
    callback. El bucle principal nunca espera: toma lo ultimo que haya.
    """

    def __init__(self):
        self.landmarks = None
        self.stamp = 0.0

    def push(self, result, output_image, timestamp_ms):
        if result.pose_landmarks:
            self.landmarks = result.pose_landmarks[0]
        else:
            self.landmarks = None
        self.stamp = timestamp_ms


def build_landmarker(model_path, mailbox, num_poses=1):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=mailbox.push,
    )
    return vision.PoseLandmarker.create_from_options(options)


def open_camera(index, width, height, fps, exposure=None):
    """Abre la webcam. En Windows DSHOW arranca mucho mas rapido que MSMF.

    Sobre 'exposure': con exposicion automatica y poca luz, la camara alarga
    el tiempo de cada toma y el FPS se desploma (medido: 16.6 en vez de 30).
    Fijar la exposicion a mano devuelve los 30 FPS, pero oscurece la imagen,
    asi que solo sirve si el stand esta bien iluminado.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # Buffer chico = menos retardo entre el movimiento real y la pantalla.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if exposure is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual en DSHOW
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Avatar en vivo por camara")
    p.add_argument("--camera", type=int, default=0, help="indice de la webcam")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float, default=None,
                   help="exposicion manual (ej. -5). Sube el FPS si hay buena luz.")
    p.add_argument("--model", default=os.path.join(ROOT, "models", "pose_landmarker_full.task"))
    p.add_argument("--no-virtualcam", action="store_true", help="no abrir la camara virtual")
    p.add_argument("--fullscreen", action="store_true", help="arrancar en pantalla completa")
    p.add_argument("--avatar", type=int, default=0, help="indice del avatar inicial")
    p.add_argument("--max-frames", type=int, default=0,
                   help="salir tras N cuadros (0 = sin limite). Util para probar.")
    p.add_argument("--headless", action="store_true",
                   help="no abrir ventana; sirve para verificar sin pantalla")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print("No encuentro el modelo:", args.model)
        print("Descargalo con: python tools/descargar_modelo.py")
        return 1

    cap = open_camera(args.camera, args.width, args.height, args.fps, args.exposure)
    if cap is None:
        print("No pude abrir la camara", args.camera,
              "- prueba con --camera 1 o cierra la app que la este usando.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    print("Camara " + str(args.camera) + " abierta a " + str(width) + "x" + str(height))

    packs = av.load_packs(os.path.join(ROOT, "assets", "packs"))
    pack_index = args.avatar % len(packs)
    renderer = av.AvatarRenderer()
    backgrounds = stage.Backgrounds(os.path.join(ROOT, "assets", "backgrounds"), (width, height))

    mailbox = LatestResult()
    landmarker = build_landmarker(args.model, mailbox)
    smoother = OneEuroFilter(freq=args.fps, min_cutoff=1.1, beta=0.018)
    presence = Hysteresis(on_frames=2, off_frames=10)

    vcam = VirtualCamera(width, height, args.fps)
    if not args.no_virtualcam:
        if vcam.open():
            print("Camara virtual activa: " + str(vcam.device))
        else:
            print("Camara virtual no disponible (" + str(vcam.error) + ").")
            print("La ventana sigue funcionando. Ver README para activarla.")

    fullscreen = args.fullscreen
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Un solo bufer de salida reutilizado: evita pedirle 2.7 MB al sistema
    # operativo treinta veces por segundo.
    screen = np.empty((height, width, 3), dtype=np.uint8)

    mirror = True
    show_hud = True
    show_bones = False
    fps_avg = float(args.fps)
    t_prev = time.perf_counter()
    t_start = t_prev
    frame_no = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Se perdio la senal de la camara.")
                break

            # Se voltea ANTES de detectar: asi los landmarks ya vienen en las
            # coordenadas de lo que se ve en pantalla y no hay que invertir nada.
            if mirror:
                frame = cv2.flip(frame, 1)

            scale = INFER_WIDTH / float(frame.shape[1])
            small = cv2.resize(frame, (INFER_WIDTH, max(int(frame.shape[0] * scale), 1)),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.perf_counter() - t_start) * 1000.0)
            landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                    timestamp_ms)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 1e-4:
                fps_avg = fps_avg * 0.9 + (1.0 / dt) * 0.1

            pack = packs[pack_index]
            landmarks = mailbox.landmarks
            visible = presence.update(landmarks is not None)
            background = backgrounds.get(frame)

            if visible and landmarks is not None:
                skel = sk.from_landmarks(landmarks, width, height,
                                         smoother=smoother, fps=fps_avg)
                canvas = renderer.render(skel, pack)
                if show_bones:
                    av.draw_debug_skeleton(canvas, skel)
                out = stage.composite(background, canvas,
                                      pack.theme.glow, pack.theme.glow_strength,
                                      roi=skel.bbox(), out=screen)
            else:
                smoother.reset()
                np.copyto(screen, background)
                out = screen
                stage.draw_banner(out, "Ponte frente a la camara",
                                  "cuerpo completo, a 2 metros")

            if show_hud:
                stage.draw_hud(out, [
                    "Avatar: " + pack.name + "   Fondo: " + backgrounds.name,
                    "FPS: " + str(int(fps_avg)) + "   " + vcam.status(),
                    "A/D avatar   F fondo   G espejo   V camara virtual   H ocultar   Q salir",
                ])

            vcam.send(out)

            frame_no += 1
            if args.headless:
                if args.max_frames and frame_no >= args.max_frames:
                    cv2.imwrite(os.path.join(ROOT, "autoprueba.png"), out)
                    print("Autoprueba guardada en autoprueba.png")
                    break
                continue

            cv2.imshow(WINDOW, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                pack_index = (pack_index + 1) % len(packs)
            elif key == ord("a"):
                pack_index = (pack_index - 1) % len(packs)
            elif key == ord("f"):
                backgrounds.next()
            elif key == ord("g"):
                mirror = not mirror
            elif key == ord("e"):
                show_bones = not show_bones
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("v"):
                vcam.toggle()
            elif key == ord("p"):
                folder = os.path.join(ROOT, "capturas")
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, time.strftime("avatar_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, out)
                print("Foto guardada en " + path)
            elif key == 9:                                  # TAB
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

            # Si el usuario cierra la ventana con la X.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            if args.max_frames and frame_no >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # La exposicion manual queda grabada en el driver y sobrevive al cierre:
        # si no se restaura, la camara sigue oscura para la proxima aplicacion
        # que la use. Se devuelve a automatico antes de soltarla.
        if args.exposure is not None:
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = auto en DSHOW
                for _ in range(5):
                    cap.read()
            except Exception:
                pass
        cap.release()
        landmarker.close()
        vcam.close()
        cv2.destroyAllWindows()

    print("Listo. Cuadros procesados: " + str(frame_no))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
