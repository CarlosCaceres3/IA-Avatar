"""Descarga los modelos de pose de MediaPipe a models/.

    python tools/descargar_modelo.py            # el modelo 'full' (recomendado)
    python tools/descargar_modelo.py --todos    # tambien lite y heavy

lite  = mas rapido, menos preciso. Sirve si la maquina del stand es lenta.
full  = el equilibrio que usa la demo por defecto.
heavy = mas preciso, bastante mas lento. Solo con GPU o un equipo potente.
"""

import argparse
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"

MODELS = {
    "lite": BASE + "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": BASE + "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": BASE + "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def download(name, url, folder):
    target = os.path.join(folder, "pose_landmarker_" + name + ".task")
    if os.path.exists(target):
        print("ya existe: " + os.path.basename(target))
        return
    print("descargando " + name + " ...")
    urllib.request.urlretrieve(url, target)
    print("  guardado (" + str(round(os.path.getsize(target) / 1e6, 1)) + " MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--todos", action="store_true", help="descargar lite, full y heavy")
    args = p.parse_args()

    folder = os.path.join(ROOT, "models")
    os.makedirs(folder, exist_ok=True)

    names = list(MODELS) if args.todos else ["full"]
    for name in names:
        try:
            download(name, MODELS[name], folder)
        except Exception as exc:
            print("  fallo " + name + ": " + str(exc))


if __name__ == "__main__":
    main()
