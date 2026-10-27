#!/usr/bin/env python3
"""handmouse: webcam hand tracking as a full pointing device (macOS).

One hand (dominant, default right):
  index fingertip              move cursor (inside the blue active box)
  quick pinch (thumb+index)    left click
  quick double pinch           right click
  pinch and hold, then move    drag
  peace sign (index+middle up, others curled)   scroll, with momentum
Two hands:
  pinch (thumb+index) with both, move apart / together   zoom
Keys in preview: q quit, p pause, i invert scroll, z invert zoom, g cycle zoom mode.
"""
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ---- tuning ---------------------------------------------------------------
DOMINANT = "Right"      # which hand drives the cursor when two are visible
CAM_W, CAM_H = 1280, 720
BOX = 0.78              # fraction of camera frame (centered) mapped to the screen; bigger = calmer
PINCH_ON, PINCH_OFF = 0.38, 0.55   # pinch distance / palm size (hysteresis)
CURSOR_MINCUTOFF, CURSOR_BETA = 0.45, 0.004   # One-Euro: lower mincutoff = steadier
CURSOR_DEADZONE_PX = 3.0    # ignore sub-3px wobble when slow
TAP_MAX = 0.32          # pinch shorter than this = tap; longer = drag
TAP_MIN = 0.04          # ignore pinch flickers shorter than this
DOUBLE_GAP = 0.30       # second tap within this = right click (left click waits this long)
DRAG_START_PX = 16      # moving this far during a pinch also starts a drag
LOST_GRACE = 0.30       # s of lost tracking before held buttons are released
SCROLL_GAIN = 1.4       # screen heights of scroll per frame-height of hand travel
SCROLL_FRICTION = 0.92  # momentum decay per frame
ZOOM_MODE = "magnify"   # magnify (trackpad pinch) | scroll (ctrl+scroll) | keys (cmd +/-)
ZOOM_GAIN = 2.0         # magnification per unit relative change in hand separation
ZOOM_SCROLL_GAIN = 900.0
ZOOM_DEADZONE = 0.006   # relative change before an update is sent
ZOOM_GRACE = 0.20       # keep zoom alive through brief pinch dropouts
ZOOM_MODES = ["magnify", "scroll", "keys"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

WRIST, THUMB_TIP, INDEX_TIP, MID_MCP, MID_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 9, 12, 16, 20
PIP = {8: 6, 12: 10, 16: 14, 20: 18}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
               (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ---- smoothing --------------------------------------------------------------
class OneEuro:
    """Adaptive low-pass: smooth when slow, responsive when fast."""

    def __init__(self, mincutoff=CURSOR_MINCUTOFF, beta=CURSOR_BETA, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff) / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        self.dx += self._alpha(self.dcutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.mincutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


# ---- OS mouse (Quartz) -----------------------------------------------------
class QuartzMouse:
    def __init__(self):
        import Quartz
        self.q = Quartz
        b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        self.width, self.height = b.size.width, b.size.height
        self.pos = (self.width / 2, self.height / 2)
        self.down = {"left": False, "right": False}
        self._acc = {"sx": 0.0, "sy": 0.0, "zy": 0.0}

    def _post(self, ev):
        self.q.CGEventPost(self.q.kCGHIDEventTap, ev)

    def _types(self, btn):
        q = self.q
        if btn == "left":
            return (q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp,
                    q.kCGEventLeftMouseDragged, q.kCGMouseButtonLeft)
        return (q.kCGEventRightMouseDown, q.kCGEventRightMouseUp,
                q.kCGEventRightMouseDragged, q.kCGMouseButtonRight)

    def move(self, x, y):
        q = self.q
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        self.pos = (x, y)
        drag = next((b for b in ("left", "right") if self.down[b]), None)
        if drag:
            _, _, dtype, bcode = self._types(drag)
            ev = q.CGEventCreateMouseEvent(None, dtype, (x, y), bcode)
        else:
            ev = q.CGEventCreateMouseEvent(None, q.kCGEventMouseMoved, (x, y), 0)
        self._post(ev)

    def press(self, btn, now=0.0):
        q = self.q
        dn, _, _, bcode = self._types(btn)
        ev = q.CGEventCreateMouseEvent(None, dn, self.pos, bcode)
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventClickState, 1)
        self.down[btn] = True
        self._post(ev)

    def release(self, btn):
        q = self.q
        _, up, _, bcode = self._types(btn)
        ev = q.CGEventCreateMouseEvent(None, up, self.pos, bcode)
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventClickState, 1)
        self.down[btn] = False
        self._post(ev)

    def scroll(self, dx, dy, zoom=False):
        """Pixels; positive dy = content moves down. zoom=True holds ctrl (app zoom)."""
        q = self.q
        key = "zy" if zoom else "sy"
        self._acc[key] += dy
        if not zoom:
            self._acc["sx"] += dx
        iy, ix = int(self._acc[key]), int(self._acc["sx"]) if not zoom else 0
        if iy == 0 and ix == 0:
            return
        self._acc[key] -= iy
        if not zoom:
            self._acc["sx"] -= ix
        ev = q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitPixel, 2, iy, ix)
        if zoom:
            q.CGEventSetFlags(ev, q.kCGEventFlagMaskControl)
        self._post(ev)

    def magnify(self, phase, amount):
        """Trackpad pinch-zoom gesture. phase: 1 begin, 2 change, 4 end."""
        q = self.q
        ev = q.CGEventCreate(None)
        q.CGEventSetType(ev, 29)                     # kCGSEventGesture
        q.CGEventSetIntegerValueField(ev, 110, 8)    # gesture type: magnify
        q.CGEventSetIntegerValueField(ev, 132, phase)
        q.CGEventSetDoubleValueField(ev, 113, amount)
        q.CGEventSetLocation(ev, self.pos)
        self._post(ev)

    def key(self, keycode, flags):
        q = self.q
        for down in (True, False):
            ev = q.CGEventCreateKeyboardEvent(None, keycode, down)
            q.CGEventSetFlags(ev, flags)
            self._post(ev)

    def zoom_key(self, zoom_in):
        self.key(24 if zoom_in else 27, self.q.kCGEventFlagMaskCommand)   # cmd+= / cmd+-


# ---- gesture engine ---------------------------------------------------------
@dataclass
class Hand:
    pts: list          # [(x, y)] normalised, 21 landmarks, selfie-view coords
    label: str


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_up(p, tip):
    return d2(p[tip], p[WRIST]) > d2(p[PIP[tip]], p[WRIST]) * 1.05


def tc(p):
    """Thumb/index pinch centre."""
    return ((p[THUMB_TIP][0] + p[INDEX_TIP][0]) / 2, (p[THUMB_TIP][1] + p[INDEX_TIP][1]) / 2)


class Engine:
    def __init__(self, mouse, aspect=16 / 9):
        self.m, self.aspect = mouse, aspect
        self.enabled = True
        self.scroll_dir = 1
        self.zoom_dir = 1
        self.zoom_mode = ZOOM_MODE
        self.fx, self.fy, self.fz = OneEuro(), OneEuro(), OneEuro(1.5, 0.01)
        self.pinch = {"p": False, "s": False}
        self.last_seen = 0.0
        self.scroll_prev = None
        self.vel = [0.0, 0.0]
        self.out = None                 # last cursor position sent
        # click state machine
        self.mode = "idle"              # idle | pending | drag
        self.t_start = 0.0
        self.origin = None
        self.second = False             # this pinch is the 2nd of a double pinch
        self.wait = None                # (t_release, pos): tap awaiting possible 2nd tap
        # zoom
        self.zoom_on = False
        self.zoom_ref = None
        self.zoom_acc = 0.0
        self.zoom_last = 0.0
        self.need_release = False       # swallow the pinch that ended a zoom

    # -- helpers
    def _hyst(self, key, ratio):
        s = self.pinch[key]
        self.pinch[key] = ratio < PINCH_ON if not s else ratio <= PINCH_OFF
        return self.pinch[key]

    def release_all(self):
        for b in ("left", "right"):
            if self.m.down[b]:
                self.m.release(b)
        self._end_zoom()
        self.mode, self.wait, self.second = "idle", None, False

    def _click(self, btn, pos):
        self.m.move(*pos)
        self.m.press(btn)
        self.m.release(btn)

    def _inertia(self):
        if self.scroll_prev is None and (abs(self.vel[0]) > 0.5 or abs(self.vel[1]) > 0.5):
            self.m.scroll(self.vel[0], self.vel[1])
            self.vel = [v * SCROLL_FRICTION for v in self.vel]
        elif self.scroll_prev is None:
            self.vel = [0.0, 0.0]

    def _end_zoom(self):
        if self.zoom_on and self.zoom_mode == "magnify":
            self.m.magnify(4, 0.0)
        self.zoom_on, self.zoom_ref, self.zoom_acc = False, None, 0.0
        self.fz.reset()

    def _screen(self, x, y):
        lo = (1 - BOX) / 2
        nx = min(max((x - lo) / BOX, 0), 1)
        ny = min(max((y - lo) / BOX, 0), 1)
        return nx * (self.m.width - 1), ny * (self.m.height - 1)

    # -- zoom
    def _zoom(self, p, s, t):
        pc, sc = tc(p), tc(s)
        if not self.zoom_on:
            self.zoom_on = True
            self.zoom_ref = None
            self.m.move(*self._screen((pc[0] + sc[0]) / 2, (pc[1] + sc[1]) / 2))  # zoom about here
            if self.zoom_mode == "magnify":
                self.m.magnify(1, 0.0)
        self.zoom_last = t
        dist = self.fz(math.hypot((pc[0] - sc[0]) * self.aspect, pc[1] - sc[1]), t)
        if self.zoom_ref is None:
            self.zoom_ref = dist
            return
        rel = (dist / max(self.zoom_ref, 1e-3) - 1.0) * self.zoom_dir
        if abs(rel) < ZOOM_DEADZONE:
            return
        self.zoom_ref = dist
        if self.zoom_mode == "magnify":
            self.m.magnify(2, rel * ZOOM_GAIN)
        elif self.zoom_mode == "scroll":
            self.m.scroll(0, rel * ZOOM_SCROLL_GAIN, zoom=True)
        else:
            self.zoom_acc += rel
            while abs(self.zoom_acc) > 0.10:
                self.m.zoom_key(self.zoom_acc > 0)
                self.zoom_acc -= math.copysign(0.10, self.zoom_acc)

    # -- main
    def update(self, hands, t):
        if not self.enabled:
            self.release_all()
            return "PAUSED"
        if not hands:
            if t - self.last_seen > LOST_GRACE:
                self.release_all()
                self.fx.reset(); self.fy.reset()
                self.scroll_prev = None
                self.pinch = dict.fromkeys(self.pinch, False)
            self._inertia()
            return "no hand"
        self.last_seen = t

        prim = next((h for h in hands if h.label == DOMINANT), hands[0])
        sec = next((h for h in hands if h is not prim), None)
        p = prim.pts
        palm = max(d2(p[WRIST], p[MID_MCP]), 1e-3)
        pi = self._hyst("p", d2(p[THUMB_TIP], p[INDEX_TIP]) / palm)
        si = False
        if sec:
            sp = sec.pts
            si = self._hyst("s", d2(sp[THUMB_TIP], sp[INDEX_TIP]) /
                            max(d2(sp[WRIST], sp[MID_MCP]), 1e-3))
        else:
            self.pinch["s"] = False

        # ---- two-hand zoom (survives short dropouts)
        if sec and pi and si:
            self.mode, self.wait, self.second = "idle", None, False   # cancel any pending click
            if self.m.down["left"]:
                self.m.release("left")
            self.need_release = True
            self.scroll_prev = None
            self._zoom(p, sec.pts, t)
            return "ZOOM"
        if self.zoom_on:
            if t - self.zoom_last < ZOOM_GRACE:
                return "ZOOM"
            self._end_zoom()
        if self.need_release:
            if pi or si:
                return "..."
            self.need_release = False

        # ---- scroll (peace sign)
        scroll_mode = (finger_up(p, INDEX_TIP) and finger_up(p, MID_TIP)
                       and not finger_up(p, RING_TIP) and not finger_up(p, PINKY_TIP)
                       and not pi and self.mode == "idle")
        if scroll_mode:
            pos = ((p[INDEX_TIP][0] + p[MID_TIP][0]) / 2, (p[INDEX_TIP][1] + p[MID_TIP][1]) / 2)
            if self.scroll_prev is not None:
                f = SCROLL_GAIN * self.m.height * self.scroll_dir
                sx = (pos[0] - self.scroll_prev[0]) * self.aspect * f
                sy = (pos[1] - self.scroll_prev[1]) * f
                self.m.scroll(sx, sy)
                self.vel = [0.6 * self.vel[0] + 0.4 * sx, 0.6 * self.vel[1] + 0.4 * sy]
            self.scroll_prev = pos
            return "SCROLL"
        self.scroll_prev = None
        self._inertia()

        # ---- pointer target (blend toward pinch centre as fingers close: stays put at click)
        w = max(0.0, min(1.0, (1.0 - d2(p[THUMB_TIP], p[INDEX_TIP]) / palm) / 0.6))
        ax = p[INDEX_TIP][0] * (1 - 0.5 * w) + p[THUMB_TIP][0] * 0.5 * w
        ay = p[INDEX_TIP][1] * (1 - 0.5 * w) + p[THUMB_TIP][1] * 0.5 * w
        sx, sy = self._screen(ax, ay)
        tx, ty = self.fx(sx, t), self.fy(sy, t)
        if self.out and math.hypot(tx - self.out[0], ty - self.out[1]) < CURSOR_DEADZONE_PX:
            tx, ty = self.out                     # kill sub-pixel wobble
        pos = (tx, ty)

        # ---- click / double-pinch / drag state machine
        if self.wait and self.mode == "idle" and not pi and t - self.wait[0] > DOUBLE_GAP:
            self._click("left", self.wait[1])     # no 2nd tap came: it was a plain click
            self.wait = None

        if self.mode == "idle":
            if pi:
                self.mode, self.t_start = "pending", t
                self.second = self.wait is not None
                self.origin = self.wait[1] if self.wait else pos
                return "PINCH"
            if self.wait:
                return "click?"                   # cursor held still while waiting
        elif self.mode == "pending":
            held = t - self.t_start
            if not pi:
                if held >= TAP_MIN and held <= TAP_MAX:
                    if self.second:
                        self._click("right", self.origin)
                        self.wait = None
                    else:
                        self.wait = (t, self.origin)
                elif self.second:
                    self.wait = None
                self.mode, self.second = "idle", False
                return "TAP" if self.wait else ("RIGHT CLICK" if held <= TAP_MAX else "move")
            if held > TAP_MAX or math.hypot(tx - self.origin[0], ty - self.origin[1]) > DRAG_START_PX:
                if self.wait:                     # tap, then hold: the tap was a click
                    self._click("left", self.wait[1])
                    self.wait = None
                self.m.move(*self.origin)
                self.m.press("left")
                self.mode, self.second = "drag", False
            else:
                return "PINCH"
        if self.mode == "drag":
            if not pi:
                self.m.release("left")
                self.mode = "idle"
                self.out = pos
                return "move"
            self.m.move(tx, ty)
            self.out = pos
            return "DRAG"

        self.m.move(tx, ty)
        self.out = pos
        return "move"


# ---- app --------------------------------------------------------------------
WINDOW = "handmouse  q quit | p pause | i invert scroll | z invert zoom | g zoom mode"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading hand model -> {MODEL_PATH.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def raise_app():
    """A bare Python process opens its window behind other apps; pull it forward."""
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyRegular
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        app.activateIgnoringOtherApps_(True)
    except Exception as e:
        print("could not raise window:", e)


def main():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    ensure_model()
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        sys.exit("Cannot open webcam. Grant Camera permission to your terminal/app.")

    engine = Engine(QuartzMouse(), aspect=CAM_W / CAM_H)
    raised = False
    t0, last_ts, fps, fps_t = time.perf_counter(), -1, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)          # selfie view; MediaPipe handedness assumes this
        h, w = frame.shape[:2]
        engine.aspect = w / h
        now = time.perf_counter()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)

        hands = [Hand([(l.x, l.y) for l in lms], cats[0].category_name)
                 for lms, cats in zip(res.hand_landmarks, res.handedness)]
        status = engine.update(hands, now)

        lo = (1 - BOX) / 2
        cv2.rectangle(frame, (int(lo * w), int(lo * h)), (int((1 - lo) * w), int((1 - lo) * h)),
                      (255, 160, 0), 2)
        for hd in hands:
            col = (0, 255, 0) if hd.label == DOMINANT else (255, 0, 255)
            pt = [(int(x * w), int(y * h)) for x, y in hd.pts]
            for a, b in CONNECTIONS:
                cv2.line(frame, pt[a], pt[b], col, 2)
            for q in pt:
                cv2.circle(frame, q, 3, (255, 255, 255), -1)
        if status == "ZOOM" and len(hands) == 2:
            mid = [tc(a.pts) for a in hands]
            cv2.line(frame, (int(mid[0][0] * w), int(mid[0][1] * h)),
                     (int(mid[1][0] * w), int(mid[1][1] * h)), (0, 200, 255), 3)

        fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
        fps_t = now
        cv2.putText(frame, f"{status} | zoom:{engine.zoom_mode} | {fps:.0f}fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        if not raised:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)   # keep preview above other apps
            raise_app()
            raised = True
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("p"):
            engine.enabled = not engine.enabled
        elif k == ord("i"):
            engine.scroll_dir *= -1
        elif k == ord("z"):
            engine.zoom_dir *= -1
        elif k == ord("g"):
            engine._end_zoom()
            engine.zoom_mode = ZOOM_MODES[(ZOOM_MODES.index(engine.zoom_mode) + 1) % 3]

    engine.release_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
