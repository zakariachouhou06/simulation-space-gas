

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from simulation import Simulation


def get_pos(simulation):
    return np.asarray(simulation.next_frame(), dtype=np.float32).reshape(-1, 3)


simulation = Simulation()

app = pg.mkQApp()
view = gl.GLViewWidget()
view.setWindowTitle("particles")
view.resize(1280, 720)
view.show()

int_min = -2147483648
int_max = 2147483647
int_minhalf = int_min >> 1
int_maxhalf = int_max >> 1
box_width = float(int_maxhalf - int_minhalf)
h = box_width / 2.0

view.opts["center"] = pg.Vector(0, 0, 0)
view.setCameraPosition(distance=box_width * 1.5)

segments = []
colors = []


def add_segment(p0, p1, col):
    segments.extend([p0, p1])
    colors.extend([col, col])


# Axes: positive half bright, negative half dim
axes = [
    (0, (1.0, 0.0, 0.0)),  # X red
    (1, (0.0, 1.0, 0.0)),  # Y green
    (2, (0.0, 0.0, 1.0)),  # Z blue
]
names = "XYZ"
for idx, rgb in axes:
    pos_end = [0.0, 0.0, 0.0]
    neg_end = [0.0, 0.0, 0.0]
    pos_end[idx] = h
    neg_end[idx] = -h
    bright = (*rgb, 1.0)
    dim = (*(0.3 * c for c in rgb), 1.0)
    add_segment((0, 0, 0), tuple(pos_end), bright)
    add_segment((0, 0, 0), tuple(neg_end), dim)

# Box outline
corners = [(sx * h, sy * h, sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
gray = (0.5, 0.5, 0.5, 1)
for i, a in enumerate(corners):
    for j, b in enumerate(corners):
        if j > i and sum(x != y for x, y in zip(a, b)) == 1:
            add_segment(a, b, gray)

view.addItem(gl.GLLinePlotItem(
    pos=np.array(segments, dtype=np.float32),
    color=np.array(colors, dtype=np.float32),
    width=1,
    antialias=False,
    mode="lines",
))

# Text labels at the axis ends
for idx, rgb in axes:
    p = [0.0, 0.0, 0.0]
    p[idx] = h
    n = [0.0, 0.0, 0.0]
    n[idx] = -h
    col = QtGui.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    view.addItem(gl.GLTextItem(pos=np.array(p), text=f"+{names[idx]}", color=col))
    view.addItem(gl.GLTextItem(pos=np.array(n), text=f"-{names[idx]}", color=col))

pos = get_pos(simulation)
particle_size = float(simulation.radius)

scatter = gl.GLScatterPlotItem(
    pos=pos,
    size=particle_size,
    color=(0, 1, 1, 1),
    pxMode=False,
)
view.addItem(scatter)

# ---------------- Energy plot (separate window) ----------------
MAX_POINTS = 5000      # history length
PLOT_EVERY = 5         # redraw the plot every N frames (plotting is slower than sim)

energy_win = pg.PlotWidget(title="Energy")
energy_win.resize(700, 400)
energy_win.setLabel("bottom", "simulation time", units="s")
energy_win.setLabel("left", "energy")
energy_win.showGrid(x=True, y=True)
energy_win.addLegend()
energy_win.show()

t_hist = []
e_hist = []      # list of lists, one per energy component
curves = []
frame = 0


def record_energy():
    e = np.atleast_1d(np.asarray(simulation.energy(), dtype=np.float64)).ravel()
    if not curves:
        pens = ["y", "c", "m", "g", "r", "w"]
        for i in range(len(e)):
            label = "E" if len(e) == 1 else f"E{i}"
            curves.append(energy_win.plot(pen=pens[i % len(pens)], name=label))
            e_hist.append([])
    t_hist.append(frame * simulation.dt)
    for i, v in enumerate(e):
        e_hist[i].append(v)
    if len(t_hist) > MAX_POINTS:
        del t_hist[0]
        for lst in e_hist:
            del lst[0]


def update():
    global frame
    scatter.setData(pos=get_pos(simulation))
    frame += 1
    if frame % PLOT_EVERY == 0:
        record_energy()
        for c, lst in zip(curves, e_hist):
            c.setData(t_hist, lst)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

app.aboutToQuit.connect(simulation.close)
pg.exec()