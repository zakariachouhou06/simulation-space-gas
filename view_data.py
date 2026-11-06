import inspect
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from simulation import Simulation

SPEED = 1              # recorded frames advanced per tick (raise to play faster)
HALF = 2.0 ** 30           # box half-width in sim units

# particle size comes from the radius default in simulation.py
_defaults = {k: v.default
             for k, v in inspect.signature(Simulation.__init__).parameters.items()
             if v.default is not inspect.Parameter.empty}
PARTICLE_SIZE = float(_defaults["radius"])

import json
meta = json.load(open("meta.json"))
n_part, dt = meta["n"], meta["dt"]
raw = np.memmap("posn=500c1=0R=100000rest=0.8.bin", dtype=np.int32, mode="r")
frames = len(raw) // (n_part * 3)
pos = raw[:frames * n_part * 3].reshape(frames, n_part, 3)
t = np.arange(frames, dtype=np.float64) * dt
e = np.loadtxt("energyn=500c1=0R=100000rest=0.8.csv", delimiter=",", skiprows=1, ndmin=2)
energy_t, energy = e[:, 0], e[:, 1:]


def fmt_t(x):
    return f"{x:.3e}" if x >= 1e4 else f"{x:.6f}"


app = pg.mkQApp()

# ---------------- main window: 3D view + controls ----------------
win = QtWidgets.QWidget()
win.setWindowTitle("particles (from run.npz)")
win.resize(1280, 760)
layout = QtWidgets.QVBoxLayout(win)

view = gl.GLViewWidget()
layout.addWidget(view, 1)

row = QtWidgets.QHBoxLayout()
btn = QtWidgets.QPushButton("pause")
slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider.setRange(0, frames - 1)
row.addWidget(btn)
row.addWidget(slider, 1)
layout.addLayout(row)

view.opts["center"] = pg.Vector(0, 0, 0)
view.setCameraPosition(distance=HALF * 3.0)

# ---------------- box, axes, labels ----------------
segments, colors = [], []


def add_segment(p0, p1, col):
    segments.extend([p0, p1])
    colors.extend([col, col])


names = "XYZ"
rgbs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
for idx, rgb in enumerate(rgbs):
    pe, ne = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    pe[idx], ne[idx] = HALF, -HALF
    add_segment((0, 0, 0), tuple(pe), (*rgb, 1.0))
    add_segment((0, 0, 0), tuple(ne), (*(0.3 * c for c in rgb), 1.0))

corners = [(sx * HALF, sy * HALF, sz * HALF) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
gray = (0.5, 0.5, 0.5, 1)
for i, a in enumerate(corners):
    for j, b in enumerate(corners):
        if j > i and sum(x != y for x, y in zip(a, b)) == 1:
            add_segment(a, b, gray)

view.addItem(gl.GLLinePlotItem(
    pos=np.array(segments, dtype=np.float32),
    color=np.array(colors, dtype=np.float32),
    width=1, antialias=False, mode="lines",
))

for idx, rgb in enumerate(rgbs):
    p, n = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    p[idx], n[idx] = HALF, -HALF
    col = QtGui.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    view.addItem(gl.GLTextItem(pos=np.array(p), text=f"+{names[idx]}", color=col))
    view.addItem(gl.GLTextItem(pos=np.array(n), text=f"-{names[idx]}", color=col))

# ---------------- particles ----------------
scatter = gl.GLScatterPlotItem(
    pos=pos[0].astype(np.float32),
    size=PARTICLE_SIZE,
    color=(0, 1, 1, 1),
    pxMode=False,
)
view.addItem(scatter)

# ---------------- time label in the corner of the 3D view ----------------
time_label = QtWidgets.QLabel(view)
time_label.setStyleSheet("color: #ddd; background: transparent; font: 16px monospace;")
time_label.move(12, 8)
time_label.resize(500, 24)
time_label.show()

# ---------------- energy plot (separate window) ----------------
energy_win = pg.PlotWidget(title="Energy")
energy_win.resize(700, 400)
energy_win.setLabel("bottom", "simulation time", units="s")
energy_win.setLabel("left", "energy")
energy_win.showGrid(x=True, y=True)
energy_win.addLegend()
pens = ["y", "c", "m", "g", "r", "w"]
for i in range(energy.shape[1]):
    label = "E" if energy.shape[1] == 1 else f"E{i}"
    energy_win.plot(energy_t, energy[:, i], pen=pens[i % len(pens)], name=label)
marker = pg.InfiniteLine(pos=t[0], angle=90, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine))
energy_win.addItem(marker)
energy_win.show()

# ---------------- playback ----------------
frame = 0
playing = True


def set_frame(k):
    global frame
    frame = int(k)
    scatter.setData(pos=pos[frame].astype(np.float32))
    time_label.setText(f"t = {fmt_t(t[frame])} s    (frame {frame}/{frames - 1})")
    marker.setValue(t[frame])
    slider.blockSignals(True)
    slider.setValue(frame)
    slider.blockSignals(False)


def toggle():
    global playing
    playing = not playing
    btn.setText("pause" if playing else "play")


def tick():
    if playing:
        set_frame((frame + SPEED) % frames)


slider.valueChanged.connect(set_frame)
btn.clicked.connect(toggle)

timer = QtCore.QTimer()
timer.timeout.connect(tick)
timer.start(16)

set_frame(0)
win.show()
pg.exec()