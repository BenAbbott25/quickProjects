#!/usr/bin/env python3
"""
Retro 3D solar-system N-body simulation.

Initial state from JPL Horizons (astroquery); gravity via REBOUND; display via VisPy.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bodies & masses (solar masses). Horizons does not return masses reliably.
# Values match major-body GM data commonly used with JPL / REBOUND.
# ---------------------------------------------------------------------------

BODIES: List[Tuple[str, str]] = [
    ("Sun", "10"),
    ("Mercury", "199"),
    ("Venus", "299"),
    ("Earth", "399"),
    ("Mars", "499"),
    ("Jupiter", "599"),
    ("Saturn", "699"),
    ("Uranus", "799"),
    ("Neptune", "899"),
]

# Solar masses (planet barycenters where applicable)
MASSES: Dict[str, float] = {
    "Sun": 1.0,
    "Mercury": 1.6601141530541165e-07,
    "Venus": 2.4478382877847715e-06,
    "Earth": 3.003489614915064e-06,  # Earth (id 399), not EMB
    "Mars": 3.227156037732997e-07,
    "Jupiter": 9.547919384243266e-04,
    "Saturn": 2.858856703621877e-04,
    "Uranus": 4.3662497883828915e-05,
    "Neptune": 5.151383772628674e-05,
}

EPOCH = "2024-01-01"
CACHE_PATH = Path(__file__).resolve().parent / ".ss_init.npz"
TRAIL_LEN = 2000
GREEN = (0.2, 1.0, 0.35, 1.0)
GREEN_DIM = (0.15, 0.75, 0.25, 0.55)

# Mean equatorial radii (km). Used only for display scaling.
RADIUS_KM: Dict[str, float] = {
    "Sun": 695700.0,
    "Mercury": 2439.7,
    "Venus": 6051.8,
    "Earth": 6371.0,
    "Mars": 3389.5,
    "Jupiter": 69911.0,
    "Saturn": 58232.0,
    "Uranus": 25362.0,
    "Neptune": 24622.0,
}
KM_PER_AU = 149_597_870.7
RADIUS_AU: Dict[str, float] = {name: km / KM_PER_AU for name, km in RADIUS_KM.items()}

# Stylized screen-space marker sizes at planet-scaling = 0.
STYLE_SIZE = 8.0
STYLE_SUN_SIZE = 14.0

ZOOM_MIN = 2.0
ZOOM_MAX = 200.0
ZOOM_FACTOR = 1.12
ZOOM_REF_DISTANCE = 40.0


def fetch_horizons_state(epoch: str = EPOCH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Query JPL Horizons for barycentric state vectors (AU, AU/day → AU/yr)."""
    from astropy.time import Time
    from astroquery.jplhorizons import Horizons

    # Horizons API expects a Julian date (float), not an ISO string.
    epoch_jd = float(Time(epoch).jd)

    names: List[str] = []
    masses = np.zeros(len(BODIES), dtype=np.float64)
    pos = np.zeros((len(BODIES), 3), dtype=np.float64)
    vel = np.zeros((len(BODIES), 3), dtype=np.float64)

    for i, (name, body_id) in enumerate(BODIES):
        print(f"Fetching {name} ({body_id}) @ {epoch} (JD {epoch_jd}) ...")
        obj = Horizons(id=body_id, location="@ssb", epochs=epoch_jd)
        tab = obj.vectors()
        names.append(name)
        masses[i] = MASSES[name]
        pos[i] = [float(tab["x"][0]), float(tab["y"][0]), float(tab["z"][0])]
        # Horizons vectors are AU/day; REBOUND units below use AU/yr
        vel[i] = [
            float(tab["vx"][0]) * 365.25,
            float(tab["vy"][0]) * 365.25,
            float(tab["vz"][0]) * 365.25,
        ]

    return masses, pos, vel, names


def load_or_fetch(
    refresh: bool = False, epoch: str = EPOCH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    if CACHE_PATH.exists() and not refresh:
        data = np.load(CACHE_PATH, allow_pickle=True)
        names = [str(n) for n in data["names"].tolist()]
        cached_epoch = str(np.asarray(data["epoch"]).reshape(-1)[0]) if "epoch" in data.files else epoch
        print(f"Loaded cached initial conditions from {CACHE_PATH}")
        return data["masses"], data["pos"], data["vel"], names, cached_epoch

    masses, pos, vel, names = fetch_horizons_state(epoch=epoch)
    np.savez(
        CACHE_PATH,
        masses=masses,
        pos=pos,
        vel=vel,
        names=np.array(names, dtype=object),
        epoch=np.array(epoch),
    )
    print(f"Cached initial conditions to {CACHE_PATH}")
    return masses, pos, vel, names, epoch


def build_simulation(
    masses: np.ndarray, pos: np.ndarray, vel: np.ndarray
) -> "rebound.Simulation":
    import rebound

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    for i in range(len(masses)):
        sim.add(
            m=float(masses[i]),
            x=float(pos[i, 0]),
            y=float(pos[i, 1]),
            z=float(pos[i, 2]),
            vx=float(vel[i, 0]),
            vy=float(vel[i, 1]),
            vz=float(vel[i, 2]),
        )
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = 0.005
    return sim


def particle_positions(sim) -> np.ndarray:
    n = sim.N
    out = np.empty((n, 3), dtype=np.float32)
    for i, p in enumerate(sim.particles):
        out[i] = (p.x, p.y, p.z)
    return out


# Discrete logarithmic-ish speed ladder (stored as years/s for the integrator).
DAYS_PER_YEAR = 365.25
MONTHS_PER_YEAR = 12.0


def _days_per_sec(n: float) -> float:
    return n / DAYS_PER_YEAR


def _months_per_sec(n: float) -> float:
    return n / MONTHS_PER_YEAR


# < 1 month: 1,2,4,8,16 days/s
# then months/s up to 1 year, then years/s on a log-ish scale
SPEED_STEPS_YR: List[float] = [
    _days_per_sec(1),
    _days_per_sec(2),
    _days_per_sec(4),
    _days_per_sec(8),
    _days_per_sec(16),
    _months_per_sec(1),
    _months_per_sec(2),
    _months_per_sec(3),
    _months_per_sec(4),
    _months_per_sec(6),
    _months_per_sec(9),
    _months_per_sec(12),  # 1 yr/s
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    50.0,
]
SPEED_DEFAULT_INDEX = 9  # 6 months/s (~0.5 yr/s)
SPEED_MIN = SPEED_STEPS_YR[0]
SPEED_MAX = SPEED_STEPS_YR[-1]
SPEED_DEFAULT = SPEED_STEPS_YR[SPEED_DEFAULT_INDEX]


def _format_speed(years_per_sec: float) -> str:
    days = years_per_sec * DAYS_PER_YEAR
    months = years_per_sec * MONTHS_PER_YEAR
    if years_per_sec < (1.0 - 1e-9) / MONTHS_PER_YEAR:
        # Strictly below 1 month/s → days
        n = int(round(days))
        return f"{n} day{'s' if n != 1 else ''}/s"
    if years_per_sec < (1.0 - 1e-9):
        n = int(round(months))
        return f"{n} month{'s' if n != 1 else ''}/s"
    if abs(years_per_sec - round(years_per_sec)) < 1e-6:
        n = int(round(years_per_sec))
        return f"{n} yr/s"
    return f"{years_per_sec:g} yr/s"


def _format_sim_datetime(epoch: str, years_elapsed: float) -> str:
    from astropy.time import Time
    import astropy.units as u

    t = Time(epoch) + years_elapsed * u.yr
    # ISO-like calendar stamp in UTC
    return t.utc.iso


def _stylized_sizes_px(names: List[str]) -> np.ndarray:
    return np.array(
        [STYLE_SUN_SIZE if name == "Sun" else STYLE_SIZE for name in names],
        dtype=np.float32,
    )


def _true_diameters_au(names: List[str]) -> np.ndarray:
    return np.array([2.0 * RADIUS_AU[name] for name in names], dtype=np.float32)


def _px_per_au(camera, canvas_size: Tuple[float, float]) -> float:
    """Approximate screen pixels per AU for a perspective turntable camera."""
    width, height = float(canvas_size[0]), float(canvas_size[1])
    view_px = max(min(width, height), 1.0)
    dist = max(float(camera.distance), 1e-6)
    fov = np.radians(float(camera.fov))
    world_span = 2.0 * dist * np.tan(0.5 * fov)
    return view_px / max(world_span, 1e-6)


def _marker_sizes_px(
    names: List[str],
    scale: float,
    camera,
    canvas_size: Tuple[float, float],
) -> np.ndarray:
    """
    scale=0: former fixed screen-space dots.
    scale=1: true AU diameters projected to pixels (shrinks when zooming out).
    """
    scale = float(np.clip(scale, 0.0, 1.0))
    stylized = _stylized_sizes_px(names)
    accurate = _true_diameters_au(names) * _px_per_au(camera, canvas_size)
    sizes = (1.0 - scale) * stylized + scale * accurate
    return np.maximum(sizes.astype(np.float32), 1e-3)


def run_viewer(sim, names: List[str], epoch: str = EPOCH) -> None:
    # Prefer PyQt6 on macOS/Apple Silicon. If VisPy fails to open a window,
    # ensure PyQt6 + PyOpenGL are installed (pygame projection is the Plan B).
    import os

    os.environ.setdefault("VISPY_BACKEND", "pyqt6")

    # QApplication must exist before VisPy creates any QWidget backends.
    # Keep a strong reference for the lifetime of this function.
    from PyQt6 import QtCore, QtWidgets

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication(sys.argv)

    from vispy import app, scene
    from vispy.scene import visuals

    app.use_app("pyqt6")

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Solar System — REBOUND")
    window.resize(1100, 860)

    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    canvas = scene.SceneCanvas(
        keys="interactive",
        show=False,
        bgcolor="black",
        size=(1100, 800),
    )
    layout.addWidget(canvas.native, stretch=1)

    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=45,
        distance=40,
        elevation=25,
        azimuth=30,
        up="+z",
    )

    # Retro-ish control strip
    hud = QtWidgets.QWidget()
    hud.setStyleSheet(
        "background-color: #050805; color: #33ff66; font-family: Menlo, Monaco, monospace;"
    )
    hud_outer = QtWidgets.QVBoxLayout(hud)
    hud_outer.setContentsMargins(12, 8, 12, 8)
    hud_outer.setSpacing(6)

    row1 = QtWidgets.QHBoxLayout()
    row1.setSpacing(12)
    row2 = QtWidgets.QHBoxLayout()
    row2.setSpacing(12)

    datetime_label = QtWidgets.QLabel()
    datetime_label.setMinimumWidth(280)
    datetime_label.setStyleSheet("font-size: 14px; font-weight: 600;")

    status_label = QtWidgets.QLabel("RUN")
    status_label.setMinimumWidth(70)

    _slider_style = """
        QSlider::groove:horizontal {
            height: 6px; background: #1a3320; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px; margin: -5px 0; background: #33ff66; border-radius: 7px;
        }
        QSlider::sub-page:horizontal { background: #226633; border-radius: 3px; }
        QSlider::tick-mark { background: #1a8840; }
    """

    speed_caption = QtWidgets.QLabel("Speed")
    speed_value = QtWidgets.QLabel()
    speed_value.setMinimumWidth(130)

    speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    speed_slider.setMinimum(0)
    speed_slider.setMaximum(len(SPEED_STEPS_YR) - 1)
    speed_slider.setPageStep(1)
    speed_slider.setSingleStep(1)
    speed_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
    speed_slider.setTickInterval(1)
    speed_slider.setValue(SPEED_DEFAULT_INDEX)
    speed_slider.setFixedWidth(220)
    speed_slider.setStyleSheet(_slider_style)

    scale_caption = QtWidgets.QLabel("Planet scale")
    scale_value = QtWidgets.QLabel("0.00")
    scale_value.setMinimumWidth(48)

    scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    scale_slider.setMinimum(0)
    scale_slider.setMaximum(100)
    scale_slider.setValue(0)
    scale_slider.setFixedWidth(160)
    scale_slider.setStyleSheet(_slider_style)

    help_label = QtWidgets.QLabel(
        "Space pause · ←→ speed · ↑↓ zoom · +/- speed · R trails · Esc quit"
    )
    help_label.setStyleSheet("color: #1a8840; font-size: 11px;")

    row1.addWidget(datetime_label)
    row1.addWidget(status_label)
    row1.addWidget(speed_caption)
    row1.addWidget(speed_slider)
    row1.addWidget(speed_value)
    row1.addStretch(1)

    row2.addWidget(scale_caption)
    row2.addWidget(scale_slider)
    row2.addWidget(scale_value)
    row2.addStretch(1)
    row2.addWidget(help_label)

    hud_outer.addLayout(row1)
    hud_outer.addLayout(row2)
    layout.addWidget(hud)

    n = sim.N
    sizes = _stylized_sizes_px(names)

    # Fixed pixel sizing; accuracy+zoom are handled by recomputing sizes each frame.
    markers = visuals.Markers(parent=view.scene, scaling="fixed")
    markers.set_data(
        particle_positions(sim),
        face_color=GREEN,
        edge_color=GREEN,
        size=sizes,
        edge_width=0,
    )

    trails: List[Deque[np.ndarray]] = [deque(maxlen=TRAIL_LEN) for _ in range(n)]
    trail_lines: List[visuals.Line] = []
    for i in range(n):
        line = visuals.Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=GREEN_DIM if i else (0.35, 1.0, 0.45, 0.35),
            width=1.0,
            parent=view.scene,
            method="gl",
        )
        trail_lines.append(line)

    state = {
        "paused": False,
        "years_per_sec": SPEED_DEFAULT,
        "speed_index": SPEED_DEFAULT_INDEX,
        "planet_scale": 0.0,
        "sim_t0": float(sim.t),
        "epoch": epoch,
    }

    def refresh_markers(pos: np.ndarray | None = None) -> None:
        if pos is None:
            pos = particle_positions(sim)
        nonlocal sizes
        sizes = _marker_sizes_px(
            names,
            state["planet_scale"],
            view.camera,
            canvas.size,
        )
        markers.set_data(pos, face_color=GREEN, edge_color=GREEN, size=sizes, edge_width=0)

    def apply_speed_index(index: int, *, sync_slider: bool = True) -> None:
        index = int(np.clip(index, 0, len(SPEED_STEPS_YR) - 1))
        years_per_sec = SPEED_STEPS_YR[index]
        state["speed_index"] = index
        state["years_per_sec"] = years_per_sec
        speed_value.setText(_format_speed(years_per_sec))
        if sync_slider:
            speed_slider.blockSignals(True)
            speed_slider.setValue(index)
            speed_slider.blockSignals(False)

    def nudge_speed(delta: int) -> None:
        apply_speed_index(state["speed_index"] + delta, sync_slider=True)

    def on_speed_slider(value: int) -> None:
        apply_speed_index(value, sync_slider=False)

    def apply_planet_scale(scale: float, *, sync_slider: bool = True) -> None:
        scale = float(np.clip(scale, 0.0, 1.0))
        state["planet_scale"] = scale
        scale_value.setText(f"{scale:.2f}")
        if sync_slider:
            scale_slider.blockSignals(True)
            scale_slider.setValue(int(round(scale * 100)))
            scale_slider.blockSignals(False)
        refresh_markers()

    def on_scale_slider(value: int) -> None:
        apply_planet_scale(value / 100.0, sync_slider=False)

    def nudge_zoom(zoom_in: bool) -> None:
        cam = view.camera
        distance = float(cam.distance)
        distance = distance / ZOOM_FACTOR if zoom_in else distance * ZOOM_FACTOR
        cam.distance = float(np.clip(distance, ZOOM_MIN, ZOOM_MAX))
        refresh_markers()

    speed_slider.valueChanged.connect(on_speed_slider)
    scale_slider.valueChanged.connect(on_scale_slider)
    apply_speed_index(SPEED_DEFAULT_INDEX)
    apply_planet_scale(0.0)

    def update_hud() -> None:
        years = sim.t - state["sim_t0"]
        datetime_label.setText(_format_sim_datetime(state["epoch"], years))
        status_label.setText("PAUSED" if state["paused"] else "RUN")
        window.setWindowTitle(
            f"Solar System — {_format_sim_datetime(state['epoch'], years)}  "
            f"({years:.2f} yr)  {_format_speed(state['years_per_sec'])}"
        )

    @canvas.events.key_press.connect
    def on_key(event):
        key = event.key
        if key == "Escape":
            window.close()
            app.quit()
        elif key == " ":
            state["paused"] = not state["paused"]
            update_hud()
        elif key in ("Right", "+", "="):
            nudge_speed(+1)
        elif key in ("Left", "-", "_"):
            nudge_speed(-1)
        elif key == "Up":
            nudge_zoom(True)
        elif key == "Down":
            nudge_zoom(False)
        elif key in ("r", "R"):
            for t in trails:
                t.clear()

    # Also catch arrow keys on the Qt window (canvas may not always get focus).
    class _KeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() == QtCore.QEvent.Type.KeyPress:
                key = event.key()
                if key == QtCore.Qt.Key.Key_Escape:
                    window.close()
                    return True
                if key == QtCore.Qt.Key.Key_Space:
                    state["paused"] = not state["paused"]
                    update_hud()
                    return True
                if key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
                    nudge_speed(+1)
                    return True
                if key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Minus):
                    nudge_speed(-1)
                    return True
                if key == QtCore.Qt.Key.Key_Up:
                    nudge_zoom(True)
                    return True
                if key == QtCore.Qt.Key.Key_Down:
                    nudge_zoom(False)
                    return True
                if key == QtCore.Qt.Key.Key_R:
                    for t in trails:
                        t.clear()
                    return True
            return super().eventFilter(obj, event)

    key_filter = _KeyFilter(window)
    window.installEventFilter(key_filter)
    canvas.native.installEventFilter(key_filter)
    speed_slider.installEventFilter(key_filter)
    scale_slider.installEventFilter(key_filter)

    timer = app.Timer(interval=1 / 60.0, start=True)

    @timer.connect
    def on_timer(event):
        if not state["paused"]:
            dt = float(event.dt) if event.dt else (1 / 60.0)
            sim.integrate(sim.t + state["years_per_sec"] * dt)

        pos = particle_positions(sim)
        refresh_markers(pos)

        for i in range(n):
            trails[i].append(pos[i].copy())
            if len(trails[i]) >= 2:
                trail_lines[i].set_data(pos=np.asarray(trails[i], dtype=np.float32))

        update_hud()

    update_hud()
    window.show()
    print(
        "Controls: Space pause | ←→ or +/- speed | ↑↓ zoom | planet-scale slider | "
        "R reset trails | Esc quit | mouse orbit"
    )
    # Keep a reference so the filter is not GC'd.
    window._key_filter = key_filter  # type: ignore[attr-defined]

    # Optional auto-quit for smoke tests: SOLARSYSTEM_QUIT_MS=500
    quit_ms = os.environ.get("SOLARSYSTEM_QUIT_MS")
    if quit_ms:
        def _auto_quit(event):
            window.close()
            app.quit()

        app.Timer(interval=max(int(quit_ms), 50) / 1000.0, iterations=1, start=True, connect=_auto_quit)

    app.run()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D solar system N-body (REBOUND + Horizons + VisPy)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch Horizons state (ignore cache)")
    parser.add_argument("--epoch", default=EPOCH, help=f"Horizons epoch (default {EPOCH})")
    parser.add_argument("--no-view", action="store_true", help="Build sim only (smoke / CI)")
    args = parser.parse_args(argv)

    masses, pos, vel, names, epoch = load_or_fetch(refresh=args.refresh, epoch=args.epoch)
    sim = build_simulation(masses, pos, vel)
    print(f"Simulation ready: {sim.N} bodies, integrator={sim.integrator}, dt={sim.dt} yr")
    print(f"Epoch: {epoch}")
    for i, name in enumerate(names):
        p = sim.particles[i]
        print(f"  {name:8s}  m={p.m:.6e}  r=({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f})")

    if args.no_view:
        # Advance a short stretch to verify integration
        sim.integrate(sim.t + 0.1)
        print(f"Integrated to t={sim.t:.4f} yr (no-view smoke OK)")
        print(f"Date: {_format_sim_datetime(epoch, sim.t)}")
        return 0

    run_viewer(sim, names, epoch=epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
