# Avatar IA en vivo — demo para stand

Una persona se para frente a la cámara y en pantalla aparece un avatar articulado
que copia sus movimientos en tiempo real, **incluida la expresión de la cara**: si
cierra los ojos, el avatar los cierra; si abre la boca, el avatar la abre. La persona
real no se ve: solo el avatar sobre el fondo elegido. La salida se puede enviar a una **cámara virtual** para que
Zoom, Meet, OBS o el proyector la tomen como si fuera una webcam más.

Todo corre **local, en CPU y sin internet** una vez descargado el modelo.

> ¿Buscando un avatar **3D estilo anime** en vez del muñeco 2D? Ese es otro stack
> (VRoid + Warudo) y está documentado en [AVATAR_3D.md](AVATAR_3D.md). Esta app
> sigue sirviendo como plan B: arranca en segundos y no depende de GPU ni de Steam.

---

## Instalación

Requiere Python 3.9–3.13.

```bash
pip install -r requirements.txt
```

Los modelos (9 MB el de pose, 3.6 MB el de cara) no vienen en el repositorio:

```bash
python tools/descargar_modelo.py
```

## Arranque rápido

```bash
python avatar_cam.py
```

O doble clic en `INICIAR.bat` (arranca en pantalla completa).

Antes del evento, corre el chequeo:

```bash
python tools/diagnostico.py
```

Verifica paquetes, modelo, cámaras, FPS reales y cámara virtual, y guarda un cuadro
de muestra en `diagnostico.png`.

---

## Varias personas a la vez

Por defecto sigue hasta **2 personas**, y le da a cada una **un avatar distinto**
para que se distingan. Las teclas `A` / `D` cambian los avatares de todas a la vez.

```bash
python avatar_cam.py --personas 3      # hasta 3 (1 a 4)
python avatar_cam.py --personas 1      # una sola, lo mas rapido
```

Cada persona lleva su propio filtro de suavizado, su propio tamaño de cuerpo y su
propio detector de cara.

**Por qué no basta con pedirle varias poses a MediaPipe:** el detector no garantiza
devolverlas siempre en el mismo orden, así que asignar el avatar por posición en la
lista haría que dos personas se intercambiaran los avatares constantemente. En cada
cuadro las poses se emparejan con las personas ya conocidas **por cercanía**, así que
quien ya estaba conserva el suyo. Si alguien desaparece un instante, su avatar lo
espera 12 cuadros antes de liberarse.

**Costo medido** en este equipo, a 720p con un tema con volumen:

| Personas | Por cuadro | FPS teóricos |
|----------|-----------|--------------|
| 1 | 13.0 ms | 77 |
| 2 | 23.4 ms | 43 |
| 3 | 35.3 ms | 28 |

Con dos va sobrado. **Con tres ya queda por debajo de los 30 FPS de la cámara**: si
necesitan tres o cuatro, usen un tema plano (los que no empiezan con "3D") o bajen
la calidad con `--calidad 0.5`.

---

## Teclas

| Tecla | Acción |
|-------|--------|
| `A` / `D` | Avatar anterior / siguiente |
| `F` | Cambiar fondo |
| `G` | Espejo on/off |
| `E` | Mostrar el esqueleto detectado (depuración) |
| `R` | Mostrar los valores de la cara en vivo (depuración) |
| `H` | Ocultar/mostrar los datos en pantalla |
| `V` | Activar/pausar la cámara virtual |
| `P` | Guardar una foto PNG en `capturas/` |
| `TAB` | Pantalla completa |
| `Q` o `ESC` | Salir |

---

## Cómo funciona

```
            ┌─► PoseLandmarker ──► 33 puntos del cuerpo ──► One Euro ─┐
webcam ──►──┤                                                         ├─► avatar
            └─► FaceLandmarker ──► expresión (ojos, boca, cejas) ─────┘     │
                                                                            │
                                        fondo + halo ──► ventana + cámara virtual
```

Los dos detectores corren **en su propio hilo** y el bucle principal nunca los
espera: toma el último resultado disponible de cada uno. Por eso agregar la cara no
baja los FPS del cuerpo.

- **`src/skeleton.py`** convierte los 33 puntos normalizados de MediaPipe en
  píxeles y calcula lo derivado: centro de hombros y cadera, radio y giro de la
  cabeza, escala del cuerpo y qué lado está más cerca de la cámara.
- **`src/smoothing.py`** aplica un filtro One Euro. Sin él, el avatar vibra 1–3 px
  aunque la persona esté quieta, y en proyección grande eso se nota mucho.
- **`src/avatar.py`** dibuja el cuerpo de atrás hacia adelante (pierna lejana,
  brazo lejano, torso, pierna cercana, brazo cercano, cabeza) para que las partes
  se tapen en el orden correcto.
- **`src/face.py`** traduce los 52 *blendshapes* de MediaPipe (valores de 0 a 1 por
  expresión) a los cinco que se dibujan: apertura de cada ojo, boca, sonrisa y cejas.
- **`src/stage.py`** genera el fondo, pega el avatar y le agrega el halo.
- **`src/camera_out.py`** envía el resultado a la cámara virtual.

---

## Avatares

Cada avatar es una carpeta en `assets/packs/`. Se cambian en vivo con `A` y `D`.

### Futbolista configurable

```bash
python tools/crear_pack_futbolista.py --nombre "Futbolista Verde" \
    --camiseta "#1B7F3B" --pantaloneta "#FFFFFF" --medias "#1B7F3B" --numero 9
```

Genera un futbolista completo con camiseta, número al pecho, pantaloneta, medias y
botines en los colores que le pasen. Cada llamada crea una carpeta nueva, así que
pueden tener varios equipos y alternarlos durante el evento.

Opciones: `--camiseta --pantaloneta --medias --botines --piel --pelo --numero --nombre`.

### Avatares con volumen (3D)

Los temas que empiezan con **3D** (`3D Azul`, `3D Heroe`, `3D Robot`, `3D Oro`) no
se pintan con colores lisos: cada extremidad se sombrea como un **cilindro** y la
cabeza como una **esfera**, con luz difusa, brillo especular, luz de contorno y
sombra de contacto en el piso. Es el mismo esqueleto de siempre — lo que cambia es
cómo se pinta cada parte.

Se cambian con `A` / `D` como cualquier otro avatar.

Para crear uno propio, basta agregar un `Theme` en `src/avatar.py` con
`volume=True`, elegir los colores y el color de la luz de contorno (`rim`).

**Sobre el costo:** sombrear píxel a píxel cuesta caro. A resolución completa eran
55 ms por cuadro (18 FPS). Por eso los temas con volumen se dibujan a **0.6 de la
resolución** y se amplían al componer: el sombreado es suave, así que a distancia de
proyector no se nota, y baja a 22 ms (46 FPS). Se puede forzar con:

```bash
python avatar_cam.py --calidad 1.0     # máxima calidad, más lento
python avatar_cam.py --calidad 0.5     # más rápido, un poco más suave
```

### Poner sus propias imágenes

```bash
python tools/crear_pack_ejemplo.py     # crea un pack de muestra para copiar
```

**Reemplacen los PNG por sus dibujos manteniendo el nombre** y el avatar los usa,
sin tocar código. Archivos que reconoce (todos opcionales — lo que falte se dibuja
con figuras del tema):

```
head.png  torso.png  upper_arm.png  forearm.png  thigh.png  shin.png  hand.png  foot.png
```

Para una pieza distinta en cada lado: `forearm_l.png`, `forearm_r.png`.

**Lo único que hay que respetar:**

- Fondo transparente (canal alfa de verdad, no blanco).
- En los huesos (brazos, piernas, torso), el dibujo va **de arriba hacia abajo**:
  arriba la articulación que queda del lado del cuerpo, abajo la del extremo.
- Mínimo unos 200 px de alto por pieza, o se verá pixelado en el proyector.

**No hace falta recortar al píxel.** Las articulaciones se deducen solas del canal
alfa: el programa mide dónde empieza y termina el dibujo dentro del PNG e ignora el
margen transparente. Pueden exportar con el espacio que les quede cómodo.

Revisen el resultado sin pararse frente a la cámara:

```bash
python tools/probar_pack.py mi_personaje
```

Avisa qué partes faltan, detecta los errores típicos (sin transparencia, imagen muy
chica, pieza mal orientada) y genera `pack_revision.png` con el avatar en 4 poses.

### theme.json

Los colores y ajustes van en `theme.json` dentro de la carpeta del pack:

```json
{
  "name": "Mi personaje",
  "skin": "#C68642",
  "suit": "#D32F2F",
  "accent": "#D32F2F",
  "outline": "#2A2520",
  "glow": "#D32F2F",
  "glow_strength": 0.35,
  "draw_face": true,
  "parts": {
    "head": { "size": 1.30 }
  }
}
```

- `skin` cubre las partes sin PNG (por ejemplo el cuello): pónganla en el tono del pack.
- `draw_face: true` hace que el programa pinte los ojos **encima** de `head.png`, y
  esos ojos siguen a la persona. Si su dibujo ya trae cara, pónganlo en `false`.
- `parts` es opcional y sirve para retocar: `size` agranda la pieza, `width` la
  ensancha, y `a` / `b` / `anchor` permiten fijar las articulaciones a mano si la
  detección automática no acierta. Lo que declaren se combina con lo detectado.

### Sobre usar personas reales

Si piensan usar la cara de alguien conocido, tengan en cuenta que eso toca sus
derechos de imagen y, si la foto es de prensa, también los derechos del fotógrafo.
Alternativas que funcionan igual de bien para un stand: personajes propios, el
futbolista configurable de arriba, personajes de dominio público, o la mascota de
la marca si tienen una.

**Fondos:** cualquier `.jpg` o `.png` que dejen en `assets/backgrounds/` aparece en
el ciclo de la tecla `F`.

---

## Cámara virtual (para el proyector, Zoom u OBS)

**Ya está instalada y verificada en este equipo** (OBS Studio 32.2.1). La app detecta
*OBS Virtual Camera* automáticamente al arrancar y lo confirma en pantalla:
`VirtualCam: OBS Virtual Camera`. Desde ahí la salida aparece como una webcam más en
Zoom, Meet, Teams, OBS o cualquier programa que liste cámaras.

Probado leyendo la salida desde otro proceso: llega **1280×720 exacto**, y el
consumidor puede pedir la resolución que quiera (640×480, 720p o 1080p) — OBS la
negocia del lado de quien recibe.

Para montarlo en **otro** equipo:

```bash
winget install --id OBSProject.OBSStudio -e
```

No hace falta abrir OBS: el instalador registra el driver solo.

Si el driver no está, **la app igual funciona**: muestra la ventana normal y avisa en
pantalla que la cámara virtual no está disponible. Para el stand, la ventana en
pantalla completa (`TAB`) puede bastar.

### Cómo se genera el video, técnicamente

La cámara virtual no graba un archivo: publica cada cuadro ya renderizado en un
dispositivo de video del sistema operativo. El bucle es:

```python
cam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR)
...
out = stage.composite(background, canvas, ...)   # el cuadro final, un array BGR
cam.send(out)                                    # se publica como webcam
cam.sleep_until_next_frame()                     # mantiene el ritmo de 30 FPS
```

Se usa `PixelFormat.BGR` porque es el formato nativo de OpenCV: así no hay que
convertir el color en cada cuadro.

Si más adelante quieren **grabar a archivo** en vez de (o además de) transmitir, es
el mismo array y tres líneas:

```python
writer = cv2.VideoWriter("clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
writer.write(out)    # dentro del bucle
writer.release()     # al terminar
```

---

## Rendimiento

Medido en este equipo, a 1280×720:

| Etapa | Costo por cuadro |
|-------|------------------|
| Detección de pose (hilo aparte) | 10.8 ms |
| Render del avatar | 2.3 ms |
| Fondo + halo + composición | 7.3 ms |

El límite real termina siendo **la webcam**, no el procesamiento.

### El detalle de la exposición — y su trampa

Con exposición automática y poca luz, la cámara alarga el tiempo de cada toma y el
FPS se desploma. Fijarla a mano lo arregla… **pero solo si hay luz suficiente.**

Medido en este equipo, en una sala con poca luz:

| Exposición | FPS | Personas detectadas |
|------------|-----|---------------------|
| Automática | 12.4 | 27 de 40 |
| Manual `-5` | 27.1 | **0 de 40** |

Con `-5` la demo corre al doble de velocidad y **no detecta absolutamente a nadie**,
porque la imagen queda casi negra y el detector no encuentra un cuerpo ahí. Unos FPS
altos con cero detecciones se ven en pantalla como una app "fluida" que no reacciona.

Por eso la exposición manual **no viene activada por defecto**. Si el stand está bien
iluminado, arranquen con:

```bash
python avatar_cam.py --exposure -5
```

y comprueben con el diagnóstico que siguen detectando personas:

```bash
python tools/diagnostico.py --exposure -5
```

Si la cuenta de "cuadros con persona detectada" baja, falta luz: suban el valor
(`-4`, `-3`) o quiten la opción. **Prueben esto con la luz real del stand antes de
abrir**, porque cambia el resultado por completo.

> Ojo: la exposición manual **queda grabada en el driver de la cámara y sobrevive al
> cierre del programa**. Si no se restaura, la webcam sigue oscura para la siguiente
> aplicación que la use (y el detector deja de encontrar personas). `avatar_cam.py` y
> el diagnóstico la devuelven a automático al salir; si alguna vez matan el proceso a
> la fuerza y la cámara queda oscura, se arregla abriéndola una vez sin `--exposure`.

### Si aún va lento

1. Bajen la resolución: `--width 960 --height 540`.
2. Usen el modelo liviano:
   ```bash
   python tools/descargar_modelo.py --todos
   python avatar_cam.py --model models/pose_landmarker_lite.task
   ```
3. Elijan un avatar sin halo (el tema *Cartoon* tiene `glow_strength` en 0).

---

## Opciones de línea de comandos

```
--camera N        índice de la webcam (por defecto 0)
--width / --height  resolución de captura (1280x720)
--fps N           FPS objetivo (30)
--exposure V      exposición manual, ej. -5
--model RUTA      archivo .task del modelo de pose
--avatar N        avatar inicial
--no-virtualcam   no abrir la cámara virtual
--fullscreen      arrancar en pantalla completa
--max-frames N    salir tras N cuadros (para probar)
--headless        no abrir ventana (para verificar sin pantalla)
```

---

## Problemas frecuentes

**"No pude abrir la cámara 0"** — otra aplicación la está usando (Teams, Zoom,
el navegador). Ciérrala, o prueba `--camera 1`.

**El avatar no aparece** — la persona debe verse de cuerpo completo o al menos de
medio cuerpo. A menos de metro y medio de la cámara el detector pierde las piernas.
Presiona `E` para ver si el esqueleto se está detectando.

**El avatar tiembla** — sube el suavizado bajando `min_cutoff` en la creación del
`OneEuroFilter` en `avatar_cam.py` (por ejemplo de `1.1` a `0.7`). A cambio, el
movimiento rápido se siente un poco más lento.

**El avatar va con retraso** — es lo contrario: sube `min_cutoff`.

**Se ve todo espejado al revés** — presiona `G`.

---

## Expresión de la cara

El avatar copia la cara de la persona: si cierra un ojo, el avatar cierra ese ojo;
si abre la boca o levanta las cejas, el avatar también.

MediaPipe entrega 52 *blendshapes* (un valor de 0 a 1 por cada expresión del rostro).
`src/face.py` se queda con cinco y los convierte a algo directo de dibujar:

| Valor | De dónde sale | Qué dibuja |
|-------|---------------|------------|
| apertura de cada ojo | `eyeBlinkLeft` / `eyeBlinkRight`, invertido | ojo abierto, entornado o cerrado |
| boca | `jawOpen` | óvalo que crece con la mandíbula |
| sonrisa | `mouthSmileLeft` + `mouthSmileRight` | curva de la boca |
| cejas | `browInnerUp`, `browOuterUp*` | cejas que suben |

**Presiona `R` para ver los valores en vivo** sobre el video. Sirve para confirmar
que la cara se está detectando y para ajustar la iluminación del stand.

El suavizado de los ojos es corto a propósito: un parpadeo dura unos 100 ms, o sea
tres cuadros a 30 FPS. Filtrar más fuerte se los comería y el efecto se pierde.

**Si los guiños salen del lado cambiado**, pon `SWAP_EYES = True` al inicio de
`src/face.py`. Los blendshapes vienen nombrados desde el punto de vista de la
persona y el cuadro se voltea antes de detectar, así que el lado depende de la cámara.

**Si la cara no se detecta**, casi siempre es una de tres: la persona está muy lejos
(el cuerpo se detecta a más distancia que la cara), está muy de perfil, o falta luz
en el rostro. La tecla `R` lo dice al instante.

Si el equipo del stand va justo de CPU, se puede apagar:

```bash
python avatar_cam.py --no-face
```

El avatar queda con la cara neutra de siempre y todo lo demás sigue igual.
