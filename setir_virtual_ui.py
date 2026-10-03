"""
Setir Virtual - versi tampilan keren (HUD).

Tampilan:
- Kamera di atas dengan kerangka tangan, garis setir bercahaya, dan label BUKA/KEPAL.
- Panel HUD di bawah: setir yang ikut berputar, panah belok kiri/kanan menyala,
  pedal GAS (hijau) dan REM (merah) yang terisi, status teks, dan FPS.

Kontrol:
- Dua tangan mengepal (memegang setir)      -> GAS otomatis
- Setir miring ke kanan (tangan kanan turun) -> tombol KANAN
- Setir miring ke kiri  (tangan kiri turun)  -> tombol KIRI
- Tangan TERBUKA                             -> REM (gas dilepas)
- Tangan hilang dari kamera                  -> semua tombol dilepas

Instalasi (Python 3.12):
    py -3.12 -m pip install mediapipe==0.10.14 opencv-python pydirectinput

Jalankan:
    py -3.12 setir_virtual_ui.py

Setelah jendela muncul, KLIK di dalam area game.
Untuk keluar: klik jendela kecil lalu tekan 'q'.
Tombol di jendela: 'n' = ganti kamera, 'q' = keluar.
"""

import math
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import pydirectinput

pydirectinput.PAUSE = 0

# ====== PENGATURAN ======
TEST_TOMBOL = False     # True = tes tombol saja tanpa kamera

KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_GAS = "up"
KEY_BRAKE = "down"

DEAD_ZONE_DEG = 12      # sudut minimal sebelum belok
AUTO_GAS = True         # gas otomatis selama 2 tangan terlihat dan tidak mengerem
SMOOTHING = 0.5         # penghalusan sudut (0 - 0.9)
DETECTION_CONF = 0.3
TRACKING_CONF = 0.3
CAMERA_INDEX = 0
MAX_CAMERAS = 4

# Rem dengan tangan terbuka
BRAKE_MODE = "salah_satu"   # "salah_satu" atau "keduanya"
OPEN_MIN_FINGERS = 3        # minimal jari lurus agar tangan dianggap terbuka (1-4)
OPEN_RATIO = 1.15           # makin besar makin ketat
BRAKE_HOLD_FRAMES = 2       # anti kedip

# Jendela
PREVIEW_W = 420             # lebar jendela (tinggi dihitung otomatis)
POSISI = "kanan-bawah"      # "kanan-bawah", "kanan-atas", "kiri-bawah", "kiri-atas"
MARGIN = 20
# ========================

WINDOW_NAME = "Setir Virtual"

# ukuran gambar kamera dan panel HUD
CAM_W, CAM_H = 640, 480
PANEL_H = 130
PREVIEW_H = int(PREVIEW_W * (CAM_H + PANEL_H) / CAM_W)

# warna (BGR)
C_BG = (30, 27, 26)
C_DIM = (85, 80, 76)
C_TEXT = (235, 235, 235)
C_ACCENT = (255, 190, 0)
C_GREEN = (110, 225, 120)
C_RED = (80, 80, 245)
C_ORANGE = (0, 165, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX
LINE = cv2.LINE_AA

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_SPEC = mp_draw.DrawingSpec(color=C_ACCENT, thickness=2, circle_radius=3)
CONN_SPEC = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]

pressed = {}


# ---------- kontrol tombol ----------
def set_key(key, down):
    if pressed.get(key, False) == down:
        return
    pressed[key] = down
    if down:
        pydirectinput.keyDown(key)
    else:
        pydirectinput.keyUp(key)


def release_all():
    for k in (KEY_LEFT, KEY_RIGHT, KEY_GAS, KEY_BRAKE):
        set_key(k, False)


def is_hand_open(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    wx, wy = lm[0].x * w, lm[0].y * h
    extended = 0
    for tip, pip in FINGERS:
        d_tip = math.hypot(lm[tip].x * w - wx, lm[tip].y * h - wy)
        d_pip = math.hypot(lm[pip].x * w - wx, lm[pip].y * h - wy)
        if d_tip > d_pip * OPEN_RATIO:
            extended += 1
    return extended >= OPEN_MIN_FINGERS


def tes_tombol():
    print("MODE TES: klik di dalam area game sekarang (balapan harus sudah jalan)...")
    for i in range(5, 0, -1):
        print(f"Mulai dalam {i} ...")
        time.sleep(1)
    try:
        print("GAS 2 detik")
        set_key(KEY_GAS, True)
        time.sleep(2)
        set_key(KEY_GAS, False)
        print("REM 1 detik")
        set_key(KEY_BRAKE, True)
        time.sleep(1)
        set_key(KEY_BRAKE, False)
        print("KIRI 1 detik")
        set_key(KEY_LEFT, True)
        time.sleep(1)
        set_key(KEY_LEFT, False)
        print("KANAN 1 detik")
        set_key(KEY_RIGHT, True)
        time.sleep(1)
        set_key(KEY_RIGHT, False)
    finally:
        release_all()
    print("Tes selesai.")


# ---------- kamera & jendela ----------
def open_camera(index):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    return cap


def setup_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_W, PREVIEW_H)

    try:
        sw, sh = pydirectinput.size()
    except Exception:
        sw, sh = 1920, 1080

    x = sw - PREVIEW_W - MARGIN if "kanan" in POSISI else MARGIN
    y = sh - PREVIEW_H - MARGIN - 40 if "bawah" in POSISI else MARGIN
    cv2.moveWindow(WINDOW_NAME, x, y)

    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        print("Catatan: versi OpenCV ini belum mendukung 'selalu di atas'. "
              "Atur posisi jendela game dan kamera secara manual.")


# ---------- gambar HUD ----------
def lerp(a, b, t):
    return a + (b - a) * t


def put_text(img, s, org, scale=0.6, color=C_TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, LINE)


def put_text_center(img, s, cx, y, scale=0.6, color=C_TEXT, thick=1):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    put_text(img, s, (int(cx - tw / 2), y), scale, color, thick)


def draw_wheel(panel, cx, cy, r, angle_deg, color):
    """Setir berbentuk T yang ikut berputar sesuai sudut."""
    cv2.circle(panel, (cx, cy), r, color, 7, LINE)
    cv2.circle(panel, (cx, cy), r - 9, C_BG, 1, LINE)
    for base in (180, 0, 90):  # kiri, kanan, bawah
        a = math.radians(base + angle_deg)
        end = (int(cx + (r - 4) * math.cos(a)), int(cy + (r - 4) * math.sin(a)))
        cv2.line(panel, (cx, cy), end, color, 6, LINE)
    # penanda posisi atas setir
    a = math.radians(-90 + angle_deg)
    p_in = (int(cx + (r - 12) * math.cos(a)), int(cy + (r - 12) * math.sin(a)))
    p_out = (int(cx + (r + 3) * math.cos(a)), int(cy + (r + 3) * math.sin(a)))
    cv2.line(panel, p_in, p_out, C_ORANGE, 5, LINE)
    cv2.circle(panel, (cx, cy), 11, color, -1, LINE)
    cv2.circle(panel, (cx, cy), 5, C_BG, -1, LINE)


def draw_arrow(panel, cx, cy, direction, active, color):
    pts = np.array([[cx + direction * 15, cy],
                    [cx - direction * 10, cy - 17],
                    [cx - direction * 10, cy + 17]], np.int32)
    if active:
        cv2.fillPoly(panel, [pts], color, LINE)
    else:
        cv2.polylines(panel, [pts], True, C_DIM, 2, LINE)


def draw_pedal(panel, x, y, w, h, level, color, label):
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_DIM, 2, LINE)
    fh = int((h - 4) * max(0.0, min(1.0, level)))
    if fh > 3:
        cv2.rectangle(panel, (x + 2, y + h - 2 - fh), (x + w - 2, y + h - 2), color, -1)
    put_text_center(panel, label, x + w // 2, y + h + 18, 0.5, color if level > 0.5 else C_DIM)


def build_panel(display_angle, tracking, arah, gas_level, brake_level,
                n_tangan, angle, hand_states, fps, cam_index):
    panel = np.full((PANEL_H, CAM_W, 3), C_BG, np.uint8)
    top_color = C_ACCENT if tracking else C_ORANGE
    cv2.line(panel, (0, 1), (CAM_W, 1), top_color, 3)

    # setir + panah belok
    wheel_color = C_TEXT if tracking else C_DIM
    if tracking and arah != 0:
        wheel_color = C_ACCENT
    draw_wheel(panel, 120, 62, 46, display_angle, wheel_color)
    draw_arrow(panel, 38, 62, -1, tracking and arah < 0, C_ACCENT)
    draw_arrow(panel, 202, 62, +1, tracking and arah > 0, C_ACCENT)

    # teks tengah
    cx = 345
    if tracking:
        arah_text = "KIRI" if arah < 0 else "KANAN" if arah > 0 else "LURUS"
        arah_color = C_ACCENT if arah != 0 else C_TEXT
        put_text_center(panel, arah_text, cx, 58, 1.0, arah_color, 2)
        put_text_center(panel, f"Sudut {angle:+.0f}", cx, 82, 0.5, C_TEXT)
        put_text_center(panel, f"Kiri: {hand_states[0]}   Kanan: {hand_states[1]}",
                        cx, 102, 0.5, C_TEXT)
    else:
        put_text_center(panel, "TUNJUKKAN 2 TANGAN", cx, 58, 0.7, C_ORANGE, 2)
        put_text_center(panel, f"Terbaca: {n_tangan}/2", cx, 84, 0.55, C_TEXT)

    # pedal
    draw_pedal(panel, 495, 18, 42, 72, gas_level, C_GREEN, "GAS")
    draw_pedal(panel, 560, 18, 42, 72, brake_level, C_RED, "REM")

    put_text_center(panel, f"FPS {fps:.0f}  |  kamera {cam_index}  |  n: ganti  |  q: keluar",
                    CAM_W // 2, PANEL_H - 8, 0.4, C_DIM)
    return panel


def draw_hand_label(frame, wx, wy, terbuka):
    label = "BUKA" if terbuka else "KEPAL"
    color = C_RED if terbuka else C_GREEN
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    x, y = int(wx - tw / 2), int(wy + 42)
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), C_BG, -1)
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), color, 2)
    put_text(frame, label, (x, y), 0.7, color, 2)


# ---------- program utama ----------
def main():
    if TEST_TOMBOL:
        tes_tombol()
        return

    cam_index = CAMERA_INDEX
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"Kamera indeks {cam_index} tidak bisa dibuka. "
              "Tutup aplikasi lain yang memakai kamera, lalu coba lagi.")
        return
    print(f"Memakai kamera indeks {cam_index}. Tekan 'n' untuk ganti kamera, 'q' untuk keluar.")

    setup_window()

    smooth_angle = 0.0
    display_angle = 0.0
    gas_level = 0.0
    brake_level = 0.0
    open_counter = 0
    fps = 0.0
    last_time = time.time()

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF,
    ) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Gagal membaca gambar dari kamera.")
                    break

                if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                    frame = cv2.resize(frame, (CAM_W, CAM_H))
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                landmarks = result.multi_hand_landmarks or []
                n_tangan = len(landmarks)

                for lm in landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                           HAND_SPEC, CONN_SPEC)

                tracking = n_tangan == 2
                arah = 0
                angle = 0.0
                hand_states = ("-", "-")

                if tracking:
                    # urutkan tangan dari kiri ke kanan layar
                    info = []
                    for lm in landmarks:
                        wrist = lm.landmark[0]
                        info.append((wrist.x, wrist.y, is_hand_open(lm, w, h)))
                    info.sort(key=lambda t: t[0])
                    (lx, ly, l_open), (rx, ry, r_open) = info
                    hand_states = ("BUKA" if l_open else "KEPAL",
                                   "BUKA" if r_open else "KEPAL")

                    want_brake = (l_open and r_open) if BRAKE_MODE == "keduanya" \
                        else (l_open or r_open)
                    open_counter = open_counter + 1 if want_brake else 0
                    braking = open_counter >= BRAKE_HOLD_FRAMES

                    raw_angle = math.degrees(math.atan2((ry - ly) * h, (rx - lx) * w))
                    smooth_angle = SMOOTHING * smooth_angle + (1 - SMOOTHING) * raw_angle
                    angle = smooth_angle

                    turn_right = angle > DEAD_ZONE_DEG
                    turn_left = angle < -DEAD_ZONE_DEG
                    arah = 1 if turn_right else -1 if turn_left else 0

                    set_key(KEY_RIGHT, turn_right)
                    set_key(KEY_LEFT, turn_left)
                    set_key(KEY_BRAKE, braking)
                    set_key(KEY_GAS, AUTO_GAS and not braking)

                    # garis setir bercahaya di kamera
                    p1 = (int(lx * w), int(ly * h))
                    p2 = (int(rx * w), int(ry * h))
                    cv2.line(frame, p1, p2, (120, 90, 0), 11, LINE)
                    cv2.line(frame, p1, p2, C_ACCENT, 4, LINE)
                    for p in (p1, p2):
                        cv2.circle(frame, p, 10, C_ACCENT, -1, LINE)
                        cv2.circle(frame, p, 4, (255, 255, 255), -1, LINE)

                    draw_hand_label(frame, p1[0], p1[1], l_open)
                    draw_hand_label(frame, p2[0], p2[1], r_open)
                else:
                    smooth_angle = 0.0
                    open_counter = 0
                    release_all()

                # animasi halus untuk setir dan pedal
                target_angle = max(-90.0, min(90.0, angle)) if tracking else 0.0
                display_angle = lerp(display_angle, target_angle, 0.35)
                gas_level = lerp(gas_level, 1.0 if pressed.get(KEY_GAS) else 0.0, 0.35)
                brake_level = lerp(brake_level, 1.0 if pressed.get(KEY_BRAKE) else 0.0, 0.35)

                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps = lerp(fps, 1.0 / dt, 0.1)

                panel = build_panel(display_angle, tracking, arah, gas_level, brake_level,
                                    n_tangan, angle, hand_states, fps, cam_index)
                cv2.imshow(WINDOW_NAME, np.vstack([frame, panel]))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("n"):
                    release_all()
                    cap.release()
                    for step in range(1, MAX_CAMERAS + 1):
                        nxt = (cam_index + step) % MAX_CAMERAS
                        new_cap = open_camera(nxt)
                        if new_cap.isOpened():
                            cap = new_cap
                            cam_index = nxt
                            print(f"Pindah ke kamera indeks {cam_index}")
                            break
                        new_cap.release()
                    else:
                        cap = open_camera(cam_index)
        finally:
            release_all()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
