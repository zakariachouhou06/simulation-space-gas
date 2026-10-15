import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from simulation import Simulation


def get_pos(simulation):
    return np.asarray(simulation.next_frame(), dtype=np.float32).reshape(-1, 3)


simulation = Simulation()

app = pg.mkQApp()
view = gl.GLViewWidget()
view.setWindowTitle("particles")
view.resize(1280, 720)
view.show()

# Calculate box boundaries using the half-limits width (int_maxhalf - int_minhalf)
int_min = -2147483648
int_max = 2147483647
int_minhalf = int_min >> 1
int_maxhalf = int_max >> 1
box_width = float(int_maxhalf - int_minhalf)

# Explicitly set the camera center/focus point to the origin (0, 0, 0)
view.opts["center"] = pg.Vector(0, 0, 0)
view.setCameraPosition(distance=box_width * 0.75)

pos = get_pos(simulation)

# pxMode=False enables real-life perspective depth (closer = bigger, further = smaller)
# Size is set to the particle diameter (2 * radius)
particle_size = float(simulation.radius) * 2.0

scatter = gl.GLScatterPlotItem(
    pos=pos,
    size=particle_size,
    color=(0, 1, 1, 1),
    pxMode=False
)
view.addItem(scatter)


def update():
    scatter.setData(pos=get_pos(simulation))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

app.aboutToQuit.connect(simulation.close)
pg.exec()
