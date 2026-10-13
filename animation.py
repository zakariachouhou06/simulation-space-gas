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

# Fit camera to the first frame
pos = get_pos(simulation)
center = (pos.max(axis=0) + pos.min(axis=0)) / 2
extent = float((pos.max(axis=0) - pos.min(axis=0)).max()) or 1.0
view.opts["center"] = pg.Vector(*center)
view.setCameraPosition(distance=extent * 2.0)

scatter = gl.GLScatterPlotItem(pos=pos, size=4, color=(0, 1, 1, 1), pxMode=True)
view.addItem(scatter)


def update():
    scatter.setData(pos=get_pos(simulation))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

app.aboutToQuit.connect(simulation.close)
pg.exec()
