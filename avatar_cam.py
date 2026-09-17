"""Avatar en vivo por camara - demo de stand.

Captura la camara, detecta la pose de la persona con MediaPipe y dibuja un
avatar articulado que imita sus movimientos. La persona real no aparece:
en pantalla solo se ve el avatar sobre el fondo elegido.

Uso:
    python avatar_cam.py
    python avatar_cam.py --camera 1 --width 1280 --height 720
    python avatar_cam.py --no-virtualcam

Teclas:
    A / D      avatar anterior / siguiente (a todas las personas)
    F          fondo siguiente
    G          espejo (on/off)
    C          cara real de la persona sobre el avatar (on/off)
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
from people import Crowd                                 # noqa: E402
import realface                                          # noqa: E402
from smoothing import Hysteresis                         # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Avatar IA - Festech"



# Ancho al que se reduce el cuadro antes de detectar. Medido en este
# equipo, el costo de la deteccion es casi plano entre 480 y 960 px
# (11-13 ms) porque MediaPipe reescala a su tamano interno de todas
# formas. Pero con DOS personas cada una ocupa la mitad del cuadro, y
# darle mas pixeles al recorte de cada una si mejora los puntos. Por eso
# el valor por defecto es holgado: sale casi gratis.
INFER_WIDTH = 640


class LatestResult:
    """Buzon del ultimo resultado del detector.

    En modo LIVE_STREAM MediaPipe corre en su propio hilo y avisa por
    callback. El bucle principal nunca espera: toma lo ultimo que haya.
    """

    def __init__(self):
        self.poses = []
        self.stamp = 0.0

    def push(self, result, output_image, timestamp_ms):
        self.poses = list(result.pose_landmarks) if result.pose_landmarks else []
        self.stamp = timestamp_ms


def build_landmarker(model_path, mailbox, num_poses=1, confidence=0.5):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=confidence,
        min_pose_presence_confidence=confidence,
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
    p.add_argument("--personas", type=int, default=2,
                   help="cuantas personas seguir a la vez (1 a 4)")
    p.add_argument("--deteccion", type=float, default=0.5,
                   help="confianza minima para dar por detectada a una persona. "
                        "Bajarla (0.4) ayuda si la segunda persona no aparece; "
                        "subirla evita detecciones fantasma.")
    p.add_argument("--ancho-deteccion", type=int, default=INFER_WIDTH,
                   help="ancho al que se reduce el cuadro para detectar")
    p.add_argument("--calidad", type=float, default=0.0,
                   help="resolucion de dibujo del avatar (0 = automatica). "
                        "0.6 por defecto en los temas con volumen, 1.0 en los planos.")
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

    max_people = max(1, min(int(args.personas), 4))
    infer_w = max(320, min(int(args.ancho_deteccion), 1280))
    mailbox = LatestResult()
    landmarker = build_landmarker(args.model, mailbox, num_poses=max_people,
                                  confidence=args.deteccion)
    presence = Hysteresis(on_frames=2, off_frames=10)

    # Cada persona lleva su propio filtro y su tamano de cuerpo; el reparto
    # por cercania evita que los avatares se intercambien cuando MediaPipe
    # cambia el orden de las poses.
    crowd = Crowd(max_people, args.fps)
    print("Siguiendo hasta " + str(max_people) + " persona(s) a la vez"
          + "  (deteccion a " + str(infer_w) + " px, confianza " + str(args.deteccion) + ")")

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
    render_q = 0.0             # resolucion de dibujo en uso

    mirror = True
    real_face = False          # pegar la cara de la persona sobre el avatar
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

            scale = infer_w / float(frame.shape[1])
            small = cv2.resize(frame, (infer_w, max(int(frame.shape[0] * scale), 1)),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.perf_counter() - t_start) * 1000.0)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp_ms)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 1e-4:
                fps_avg = fps_avg * 0.9 + (1.0 / dt) * 0.1

            pack = packs[pack_index]
            activas = crowd.update(mailbox.poses)
            visible = presence.update(bool(activas))
            background = backgrounds.get(frame)

            if visible and activas:
                # Los temas con volumen se dibujan mas chico y se amplian al
                # componer: el sombreado es suave y a media resolucion cuesta
                # la cuarta parte, sin diferencia visible en un proyector.
                q = args.calidad if args.calidad > 0 else (0.6 if pack.theme.volume else 1.0)
                if q != render_q:
                    # Cambiar de escala mueve todos los puntos: los filtros
                    # lo verian como un salto de la persona. Solo se olvidan
                    # los filtros; la pose de este cuadro se va a dibujar.
                    for persona in crowd.people:
                        persona.reset_filters()
                    render_q = q
                rw = max(int(width * q), 32)
                rh = max(int(height * q), 32)

                # Todas las personas se dibujan en el MISMO lienzo y se
                # componen de una sola vez: componer por persona costaria
                # una copia del fondo por cada una.
                canvas = renderer.blank(rw, rh)
                caja = None
                glow_theme = pack.theme

                for persona in activas:
                    # Cada persona usa un avatar distinto para que se
                    # distingan entre si; A y D los corren a todas.
                    suyo = packs[(pack_index + persona.slot) % len(packs)]
                    skel = sk.from_landmarks(persona.landmarks, rw, rh,
                                             smoother=persona.smoother,
                                             fps=fps_avg, state=persona.body)
                    # El recorte de cara usa el cuadro COMPLETO, pero el
                    # esqueleto puede estar reducido: se devuelve a escala.
                    renderer.render(skel, suyo, canvas=canvas)
                    if real_face:
                        # Va despues del avatar: tapa la cara dibujada.
                        realface.paste(canvas, frame, skel.head_c, skel.head_r, q)
                    if show_bones:
                        av.draw_debug_skeleton(canvas, skel)

                    b = skel.bbox()
                    caja = b if caja is None else (min(caja[0], b[0]), min(caja[1], b[1]),
                                                   max(caja[2], b[2]), max(caja[3], b[3]))
                    if persona.slot == 0:
                        glow_theme = suyo.theme

                out = stage.composite(background, canvas,
                                      glow_theme.glow, glow_theme.glow_strength,
                                      roi=caja, out=screen)
            else:
                for persona in crowd.people:
                    persona.reset()
                np.copyto(screen, background)
                out = screen
                stage.draw_banner(out, "Ponte frente a la camara",
                                  "cuerpo completo, a 2 metros")

            if show_hud:
                stage.draw_hud(out, [
                    "Avatar: " + pack.name + "   Fondo: " + backgrounds.name +
                    "   Personas: " + str(len(activas)) + "/" + str(max_people) +
                    ("   CARA REAL" if real_face else ""),
                    "FPS: " + str(int(fps_avg)) + "   " + vcam.status(),
                    "A/D avatar   F fondo   C cara real   G espejo   V camara   Q salir",
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
            elif key == ord("c"):
                real_face = not real_face
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
