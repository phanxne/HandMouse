# HandMouse

Use your laptop webcam to track your hands and use them as an input device. Two parts:

- **Hand Sandbox** (`launch_sim.py`): a 3D physics playground in the browser (three.js + cannon-es + MediaPipe) that you control with your hands.
- **handmouse** (`handmouse.py`): drives the real macOS cursor, clicks, scroll and pinch-zoom from your hand.

Hand tracking is [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). Everything runs locally.

## Setup (macOS, Python 3.10+)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The hand model (`hand_landmarker.task`, ~7 MB) is downloaded on first run.

## Hand Sandbox

```bash
python3 launch_sim.py
```

Opens `http://localhost:8765/`; click **Start camera** and allow access. Needs internet on first load (MediaPipe WASM and three.js come from a CDN).

| Gesture | Action |
| --- | --- |
| Move hand | Cursor follows your index finger; hand toward the webcam pushes it deeper |
| Pinch (thumb + index) near an object | Grab it; release to throw |
| Pinch on empty space, drag | Orbit camera |
| Pinch with both hands, spread / squeeze | Zoom (moving both orbits) |

Buttons: **+ Cube**, **+ Ball**, **Reset**, **Flip X** (if left/right feels mirrored).

## handmouse (system mouse)

```bash
.venv/bin/python handmouse.py
```

Grant Camera, Accessibility and Input Monitoring to the app running Python (Terminal, iTerm, ...).

| Gesture | Action |
| --- | --- |
| Index fingertip | Move cursor (inside the blue box) |
| Quick pinch | Left click |
| Quick double pinch | Right click |
| Pinch and hold, then move | Drag |
| Peace sign (index + middle up) | Scroll, with momentum |
| Pinch with both hands, spread / squeeze | Zoom |

Preview keys: `q` quit, `p` pause, `i` invert scroll, `z` invert zoom, `g` cycle zoom mode (magnify / ctrl+scroll / cmd +-).
Tune sensitivity, smoothing and click timing in the constants at the top of `handmouse.py`.

## Notes

- Two-hand system zoom sends synthetic trackpad pinch events; if an app ignores them, press `g` to use ctrl+scroll or cmd +/-.
- macOS only for `handmouse.py` (uses Quartz). The sandbox runs in any modern browser with a webcam.
