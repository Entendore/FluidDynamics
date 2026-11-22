import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout,
    QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont


# ---------------------------------------------------------------------------
# Fluid Solver – Jos Stam "Stable Fluids" (SIGGRAPH 1999)
# ---------------------------------------------------------------------------
class FluidSolver:
    """2-D incompressible fluid solver on a staggered grid.

    Pipeline per time-step (velocity then density):
        1. Add external forces / sources
        2. Diffuse  (implicit Jacobi)
        3. Advect   (semi-Lagrangian back-trace)
        4. Project  (pressure solve → divergence-free)

    Boundary conditions:
        b=0  scalar  → Neumann (copy)
        b=1  u-vel   → negate at vertical   walls
        b=2  v-vel   → negate at horizontal walls
    """

    def __init__(self, N: int, diffusion=0.0001, viscosity=0.0001, dt=0.1):
        self.N = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        self.iterations = 20          # Jacobi iterations
        self.decay = 0.0             # density decay per step

        s = (N + 2, N + 2)
        self.u      = np.zeros(s)    # x-velocity
        self.v      = np.zeros(s)    # y-velocity
        self.u_prev = np.zeros(s)
        self.v_prev = np.zeros(s)
        self.dens      = np.zeros(s)
        self.dens_prev = np.zeros(s)

    # -- helpers ----------------------------------------------------------
    def reset(self):
        for a in (self.u, self.v, self.u_prev, self.v_prev,
                  self.dens, self.dens_prev):
            a[:] = 0

    def _set_bnd(self, b, x):
        N = self.N
        x[0,    1:N+1] = -x[1, 1:N+1] if b == 1 else x[1, 1:N+1]
        x[N+1,  1:N+1] = -x[N, 1:N+1] if b == 1 else x[N, 1:N+1]
        x[1:N+1, 0   ] = -x[1:N+1, 1] if b == 2 else x[1:N+1, 1]
        x[1:N+1, N+1 ] = -x[1:N+1, N] if b == 2 else x[1:N+1, N]
        x[0,   0  ] = 0.5 * (x[1, 0  ] + x[0,   1])
        x[0,   N+1] = 0.5 * (x[1, N+1] + x[0,   N])
        x[N+1, 0  ] = 0.5 * (x[N, 0  ] + x[N+1, 1])
        x[N+1, N+1] = 0.5 * (x[N, N+1] + x[N+1, N])

    # -- core steps -------------------------------------------------------
    def _diffuse(self, b, x, x0, diff, dt):
        N = self.N
        a = dt * diff * N * N
        if a < 1e-12:
            x[:] = x0
            return
        c = 1.0 + 4.0 * a
        for _ in range(self.iterations):
            x[1:N+1, 1:N+1] = (
                x0[1:N+1, 1:N+1]
                + a * (x[0:N, 1:N+1] + x[2:N+2, 1:N+1]
                     + x[1:N+1, 0:N] + x[1:N+1, 2:N+2])
            ) / c
            self._set_bnd(b, x)

    def _advect(self, b, d, d0, u, v, dt):
        N = self.N
        dt0 = dt * N
        i = np.arange(1, N + 1)
        j = np.arange(1, N + 1)
        ii, jj = np.meshgrid(i, j, indexing='ij')

        x = ii - dt0 * u[1:N+1, 1:N+1]
        y = jj - dt0 * v[1:N+1, 1:N+1]
        x = np.clip(x, 0.5, N + 0.5)
        y = np.clip(y, 0.5, N + 0.5)

        i0 = np.floor(x).astype(int);  i1 = i0 + 1
        j0 = np.floor(y).astype(int);  j1 = j0 + 1
        s1 = x - i0;  s0 = 1.0 - s1
        t1 = y - j0;  t0 = 1.0 - t1

        d[1:N+1, 1:N+1] = (
            s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
          + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
        )
        self._set_bnd(b, d)

    def _project(self, u, v, p, div):
        N = self.N
        h = 1.0 / N
        div[1:N+1, 1:N+1] = -0.5 * h * (
            u[2:N+2, 1:N+1] - u[0:N, 1:N+1]
          + v[1:N+1, 2:N+2] - v[1:N+1, 0:N]
        )
        p[:] = 0.0
        self._set_bnd(0, div)
        self._set_bnd(0, p)

        for _ in range(self.iterations):
            p[1:N+1, 1:N+1] = (
                div[1:N+1, 1:N+1]
              + p[0:N, 1:N+1] + p[2:N+2, 1:N+1]
              + p[1:N+1, 0:N] + p[1:N+1, 2:N+2]
            ) / 4.0
            self._set_bnd(0, p)

        u[1:N+1, 1:N+1] -= 0.5 * (p[2:N+2, 1:N+1] - p[0:N, 1:N+1]) / h
        v[1:N+1, 1:N+1] -= 0.5 * (p[1:N+1, 2:N+2] - p[1:N+1, 0:N]) / h
        self._set_bnd(1, u)
        self._set_bnd(2, v)

    # -- full step --------------------------------------------------------
    def step(self):
        # ---- velocity step ----
        self.u += self.dt * self.u_prev
        self.v += self.dt * self.v_prev

        tmp = self.u.copy()
        self._diffuse(1, self.u, tmp, self.visc, self.dt)
        tmp = self.v.copy()
        self._diffuse(2, self.v, tmp, self.visc, self.dt)
        self._project(self.u, self.v, self.u_prev, self.v_prev)

        u0 = self.u.copy();  v0 = self.v.copy()
        self._advect(1, self.u, u0, u0, v0, self.dt)
        self._advect(2, self.v, v0, u0, v0, self.dt)
        self._project(self.u, self.v, self.u_prev, self.v_prev)

        # ---- density step ----
        self.dens += self.dt * self.dens_prev

        tmp = self.dens.copy()
        self._diffuse(0, self.dens, tmp, self.diff, self.dt)
        tmp = self.dens.copy()
        self._advect(0, self.dens, tmp, self.u, self.v, self.dt)

        if self.decay > 0:
            self.dens *= (1.0 - self.decay * self.dt)
        self.dens = np.clip(self.dens, 0, 10)

        # clear sources for next frame
        self.u_prev[:] = 0;  self.v_prev[:] = 0;  self.dens_prev[:] = 0

    # -- presets ----------------------------------------------------------
    def add_vortex(self, cx, cy, strength, radius=5):
        N = self.N
        i = np.arange(1, N + 1);  j = np.arange(1, N + 1)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        dx = (ii - cx).astype(float);  dy = (jj - cy).astype(float)
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        fac = strength * np.exp(-r * r / (2.0 * radius * radius))
        self.u_prev[1:N+1, 1:N+1] += -dy / r * fac
        self.v_prev[1:N+1, 1:N+1] +=  dx / r * fac
        self.dens_prev[1:N+1, 1:N+1] += np.abs(fac) * 0.3

    def add_jet(self, y_pos, strength=5.0):
        N = self.N
        half = max(1, N // 25)
        js = max(1, y_pos - half);  je = min(N + 1, y_pos + half + 1)
        self.u_prev[1:4, js:je] += strength
        self.dens_prev[1:4, js:je] += strength * 8


# ---------------------------------------------------------------------------
# Fluid Canvas – renders the simulation and captures mouse interaction
# ---------------------------------------------------------------------------
class FluidCanvas(QWidget):
    def __init__(self, solver: FluidSolver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # interaction settings (modified from main window sliders)
        self.color_mode = 0            # 0 density 1 velocity 2 vorticity 3 direction
        self.force_mult = 5.0
        self.dens_amount = 100.0
        self.brush_size = 3
        self.show_vectors = False

        self._pressed = False
        self._last_pos = None
        self._img_bytes = None         # prevent GC while QImage holds data

    # -- mouse interaction ------------------------------------------------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._pressed = True
            self._last_pos = ev.position()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._pressed = False
            self._last_pos = None

    def mouseMoveEvent(self, ev):
        if not self._pressed or self._last_pos is None:
            return
        pos = ev.position()
        N = self.solver.N
        w, h = self.width(), self.height()

        gx = int(pos.x() / w * N) + 1
        gy = int(pos.y() / h * N) + 1
        dx = (pos.x() - self._last_pos.x()) / w * N
        dy = (pos.y() - self._last_pos.y()) / h * N

        bs = self.brush_size
        for di in range(-bs, bs + 1):
            for dj in range(-bs, bs + 1):
                ix, iy = gx + di, gy + dj
                if 1 <= ix <= N and 1 <= iy <= N:
                    d = np.sqrt(di * di + dj * dj)
                    if d <= bs:
                        f = max(0.0, 1.0 - d / (bs + 1))
                        self.solver.u_prev[ix, iy] += dx * self.force_mult * f
                        self.solver.v_prev[ix, iy] += dy * self.force_mult * f
                        self.solver.dens_prev[ix, iy] += self.dens_amount * f
        self._last_pos = pos

    # -- rendering --------------------------------------------------------
    def _build_pixmap(self) -> QPixmap:
        N = self.solver.N
        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return QPixmap()

        # --- compute colour channels (row = y, col = x) via .T -----------
        if self.color_mode == 0:                       # density – fire palette
            d = self.solver.dens[1:N+1, 1:N+1].T
            t = np.clip(d, 0, 1)
            r = np.clip(t * 3.0, 0, 1)
            g = np.clip(t * 3.0 - 1.0, 0, 1)
            b = np.clip(t * 3.0 - 2.0, 0, 1) * 0.8
            br, bg, bb = 0.02, 0.01, 0.06

        elif self.color_mode == 1:                     # velocity magnitude
            u = self.solver.u[1:N+1, 1:N+1].T
            v = self.solver.v[1:N+1, 1:N+1].T
            t = np.clip(np.sqrt(u * u + v * v) * 3, 0, 1)
            r = t
            g = t * 0.35
            b = np.clip(0.35 - t * 0.35, 0, 0.35)
            br, bg, bb = 0.04, 0.01, 0.08

        elif self.color_mode == 2:                     # vorticity (curl)
            u = self.solver.u[1:N+1, 1:N+1].T
            v = self.solver.v[1:N+1, 1:N+1].T
            vort = np.zeros_like(u)
            if N > 2:
                vort[1:-1, 1:-1] = (
                    (v[2:, 1:-1] - v[:-2, 1:-1])
                  - (u[1:-1, 2:] - u[1:-1, :-2])
                ) * 0.5
            vm = max(np.abs(vort).max(), 1e-4)
            vn = vort / vm
            r = np.clip(vn * 2.5, 0, 1)
            g = np.full_like(vn, 0.06)
            b = np.clip(-vn * 2.5, 0, 1)
            br, bg, bb = 0.04, 0.02, 0.05

        else:                                          # velocity direction
            u = self.solver.u[1:N+1, 1:N+1].T
            v = self.solver.v[1:N+1, 1:N+1].T
            t = np.clip(np.sqrt(u * u + v * v) * 3, 0, 1)
            angle = np.arctan2(v, u)
            r = (np.sin(angle) * 0.5 + 0.5) * t
            g = (np.sin(angle + 2.094) * 0.5 + 0.5) * t
            b = (np.sin(angle + 4.189) * 0.5 + 0.5) * t
            br, bg, bb = 0.02, 0.02, 0.03

        r = np.clip(r + br, 0, 1)
        g = np.clip(g + bg, 0, 1)
        b = np.clip(b + bb, 0, 1)

        r8 = (r * 255).astype(np.uint32)
        g8 = (g * 255).astype(np.uint32)
        b8 = (b * 255).astype(np.uint32)

        pixels = 0xFF000000 | (r8 << 16) | (g8 << 8) | b8   # 0xAARRGGBB
        self._img_bytes = pixels.tobytes()
        img = QImage(self._img_bytes, N, N, N * 4, QImage.Format_RGB32)
        pm = QPixmap.fromImage(img.scaled(w, h, Qt.IgnoreAspectRatio,
                                           Qt.SmoothTransformation))

        # --- optional velocity vectors ------------------------------------
        if self.show_vectors:
            p = QPainter(pm)
            pen = QPen(QColor(255, 255, 255, 110), 1)
            p.setPen(pen)
            step = max(N // 28, 2)
            sx, sy = w / N, h / N
            for ci in range(1, N + 1, step):
                for cj in range(1, N + 1, step):
                    ux = self.solver.u[ci, cj]
                    vy = self.solver.v[ci, cj]
                    mag = np.sqrt(ux * ux + vy * vy)
                    if mag > 0.005:
                        ox = (ci - 0.5) * sx;  oy = (cj - 0.5) * sy
                        ln = min(mag * sx * 4, sx * step * 0.85)
                        p.drawLine(QPointF(ox, oy),
                                   QPointF(ox + ux / mag * ln,
                                           oy + vy / mag * ln))
            p.end()
        return pm

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        pm = self._build_pixmap()
        if not pm.isNull():
            p.drawPixmap(0, 0, pm)
        # hint overlay when canvas is empty
        if self.solver.dens.max() < 0.01 and not self._pressed:
            p.setPen(QColor(200, 200, 220, 160))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Click and drag to create fluid flow\n"
                       "Try the presets on the right panel →")
        p.end()


# ---------------------------------------------------------------------------
# Main Window – controls, presets, and educational panel
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluid Dynamics Learning Simulator")
        self.setMinimumSize(1080, 720)

        self._grid_size = 128
        self._solver = FluidSolver(self._grid_size)
        self._paused = False
        self._frame_count = 0

        # ---- central layout ----
        central = QWidget();  self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(12);  root.setContentsMargins(12, 12, 12, 12)

        # canvas
        self._canvas = FluidCanvas(self._solver)
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame{border:1px solid #45475a; border-radius:8px;}")
        fl = QVBoxLayout(frame);  fl.setContentsMargins(2, 2, 2, 2)
        fl.addWidget(self._canvas)
        root.addWidget(frame, stretch=3)

        # right panel
        right = QVBoxLayout();  right.setSpacing(8)
        root.addLayout(right, stretch=1)

        title = QLabel("🌊  Fluid Dynamics Lab")
        title.setStyleSheet("font-size:20px; font-weight:bold; padding:4px 0;")
        right.addWidget(title)

        # -- simulation parameters --
        grp = QGroupBox("⚙  Simulation");  lay = QGridLayout();  lay.setSpacing(5)
        grp.setLayout(lay)
        self.sl_visc, self.lb_visc = self._slider(lay, 0, "Viscosity", 0, 200, 10, self._set_visc)
        self.sl_diff, self.lb_diff = self._slider(lay, 1, "Diffusion", 0, 200, 10, self._set_diff)
        self.sl_dt,   self.lb_dt   = self._slider(lay, 2, "Time step", 1,  50, 10, self._set_dt)
        self.sl_decay,self.lb_decay= self._slider(lay, 3, "Decay",     0,  80,  5, self._set_decay)
        right.addWidget(grp)

        # -- interaction --
        grp2 = QGroupBox("🖌  Interaction");  lay2 = QGridLayout();  lay2.setSpacing(5)
        grp2.setLayout(lay2)
        self.sl_force,self.lb_force = self._slider(lay2, 0, "Force",    1, 150, 50, self._set_force)
        self.sl_dens, self.lb_dens  = self._slider(lay2, 1, "Density",  1, 300,100, self._set_dens)
        self.sl_brush,self.lb_brush = self._slider(lay2, 2, "Brush",    1,  15,  3, self._set_brush)
        right.addWidget(grp2)

        # -- visualisation --
        grp3 = QGroupBox("👁  Visualisation");  lay3 = QVBoxLayout();  grp3.setLayout(lay3)
        self._cmb_color = QComboBox()
        self._cmb_color.addItems(["🔥 Density (fire)", "💨 Velocity magnitude",
                                   "🌀 Vorticity",       "🌈 Velocity direction"])
        self._cmb_color.currentIndexChanged.connect(lambda i: setattr(self._canvas, 'color_mode', i))
        lay3.addWidget(self._cmb_color)
        self._btn_vec = QPushButton("Velocity vectors: OFF")
        self._btn_vec.setCheckable(True)
        self._btn_vec.clicked.connect(self._toggle_vec)
        lay3.addWidget(self._btn_vec)
        right.addWidget(grp3)

        # -- presets --
        grp4 = QGroupBox("🎯  Presets");  lay4 = QVBoxLayout();  grp4.setLayout(lay4)
        for label, fn in [("🌀  Double Vortex", self._pre_vortex),
                          ("➡️  Horizontal Jet",  self._pre_jet),
                          ("💫  Random Splashes", self._pre_random),
                          ("🌊  Turbulence Field",self._pre_turb)]:
            b = QPushButton(label);  b.clicked.connect(fn);  lay4.addWidget(b)
        right.addWidget(grp4)

        # -- actions --
        grp5 = QGroupBox("🎮  Controls");  lay5 = QVBoxLayout();  grp5.setLayout(lay5)
        row = QHBoxLayout()
        self._btn_pause = QPushButton("⏸  Pause");  self._btn_pause.setCheckable(True)
        self._btn_pause.clicked.connect(self._toggle_pause);  row.addWidget(self._btn_pause)
        b_reset = QPushButton("🔄  Reset");  b_reset.clicked.connect(self._reset);  row.addWidget(b_reset)
        lay5.addLayout(row)
        row2 = QHBoxLayout();  row2.addWidget(QLabel("Grid:"))
        self._cmb_grid = QComboBox();  self._cmb_grid.addItems(["64","128","256"])
        self._cmb_grid.setCurrentIndex(1)
        self._cmb_grid.currentIndexChanged.connect(self._change_grid)
        row2.addWidget(self._cmb_grid);  lay5.addLayout(row2)
        right.addWidget(grp5)

        # -- learn --
        grp6 = QGroupBox("📖  Learn – Navier-Stokes");  lay6 = QVBoxLayout();  grp6.setLayout(lay6)
        info = QLabel(
            "<small>"
            "<b>∂<b>v</b>/∂t + (<b>v</b>·∇)<b>v</b> = −∇p/ρ + ν∇²<b>v</b> + <b>f</b></b><br><br>"
            "This is the <b>incompressible Navier-Stokes</b> equation – "
            "the fundamental PDE governing fluid motion.<br><br>"
            "<b>Viscosity (ν)</b> – fluid's internal friction; "
            "high ν → honey, low ν → water.<br>"
            "<b>Diffusion</b> – molecular spreading of dye/density.<br>"
            "<b>Advection</b> – the velocity field carries (transports) "
            "quantities along flow lines.<br>"
            "<b>Pressure projection</b> – enforces ∇·<b>v</b>=0 "
            "(incompressibility).<br>"
            "<b>Vorticity ω = ∇×<b>v</b></b> – measures local rotation; "
            "red = CW, blue = CCW.<br><br>"
            "<b>Algorithm (Stable Fluids, Stam 1999):</b><br>"
            "1️⃣ Add forces → 2️⃣ Diffuse (implicit) → "
            "3️⃣ Project → 4️⃣ Advect (semi-Lagrangian) → 5️⃣ Project<br><br>"
            "The method is <b>unconditionally stable</b>: no CFL restriction "
            "on Δt, so it never blows up!"
            "</small>"
        )
        info.setWordWrap(True)
        lay6.addWidget(info)
        right.addWidget(grp6)

        self._lb_fps = QLabel("FPS: –")
        self._lb_fps.setStyleSheet("color:#777; font-size:11px;")
        right.addWidget(self._lb_fps)
        right.addStretch()

        # ---- timers ----
        self._timer = QTimer();  self._timer.timeout.connect(self._tick);  self._timer.start(16)
        self._fps_timer = QTimer();  self._fps_timer.timeout.connect(self._fps);  self._fps_timer.start(1000)

        # ---- style ----
        self.setStyleSheet(self._stylesheet())

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _slider(layout, row, label, lo, hi, default, callback):
        layout.addWidget(QLabel(label + ":"), row, 0)
        sl = QSlider(Qt.Horizontal);  sl.setRange(lo, hi);  sl.setValue(default)
        layout.addWidget(sl, row, 1)
        lb = QLabel("");  lb.setMinimumWidth(58)
        layout.addWidget(lb, row, 2)
        sl.valueChanged.connect(callback)
        callback(default)
        return sl, lb

    # -- slider callbacks -------------------------------------------------
    def _set_visc(self, v):
        val = v * 1e-5;  self._solver.visc = val;  self.lb_visc.setText(f"{val:.5f}")

    def _set_diff(self, v):
        val = v * 1e-5;  self._solver.diff = val;  self.lb_diff.setText(f"{val:.5f}")

    def _set_dt(self, v):
        val = v * 0.01;  self._solver.dt = val;  self.lb_dt.setText(f"{val:.2f}")

    def _set_decay(self, v):
        val = v * 0.02;  self._solver.decay = val;  self.lb_decay.setText(f"{val:.2f}")

    def _set_force(self, v):
        val = v * 0.1;  self._canvas.force_mult = val;  self.lb_force.setText(f"{val:.1f}")

    def _set_dens(self, v):
        self._canvas.dens_amount = float(v);  self.lb_dens.setText(str(v))

    def _set_brush(self, v):
        self._canvas.brush_size = v;  self.lb_brush.setText(str(v))

    def _toggle_vec(self, on):
        self._canvas.show_vectors = on
        self._btn_vec.setText(f"Velocity vectors: {'ON' if on else 'OFF'}")

    def _toggle_pause(self, on):
        self._paused = on
        self._btn_pause.setText("▶  Resume" if on else "⏸  Pause")

    def _reset(self):
        self._solver.reset()

    def _change_grid(self, idx):
        sizes = [64, 128, 256]
        n = sizes[idx]
        self._solver = FluidSolver(n, self._solver.diff, self._solver.visc, self._solver.dt)
        self._solver.decay = float(self.sl_decay.value()) * 0.02
        self._canvas.solver = self._solver

    # -- presets ----------------------------------------------------------
    def _pre_vortex(self):
        N = self._solver.N
        r = max(N // 15, 3)
        self._solver.add_vortex(N // 3,     N // 2,  10, r)
        self._solver.add_vortex(2 * N // 3, N // 2, -10, r)

    def _pre_jet(self):
        self._solver.add_jet(self._solver.N // 2, 6.0)

    def _pre_random(self):
        N = self._solver.N
        for _ in range(6):
            x = np.random.randint(N // 5, 4 * N // 5)
            y = np.random.randint(N // 5, 4 * N // 5)
            s = np.random.uniform(-8, 8)
            self._solver.add_vortex(x, y, s, max(N // 20, 3))

    def _pre_turb(self):
        N = self._solver.N
        i = np.arange(1, N+1); j = np.arange(1, N+1)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        cx, cy = N / 2, N / 2
        dx = (ii - cx).astype(float);  dy = (jj - cy).astype(float)
        r = np.sqrt(dx*dx + dy*dy) + 1e-6
        scale = 6.0
        self._solver.u_prev[1:N+1, 1:N+1] += scale * np.sin(2*np.pi*jj/N) * np.exp(-r*r/(N*N*0.15))
        self._solver.v_prev[1:N+1, 1:N+1] += scale * np.cos(2*np.pi*ii/N) * np.exp(-r*r/(N*N*0.15))
        self._solver.dens_prev[1:N+1, 1:N+1] += 2.0 * np.exp(-r*r/(N*N*0.1))

    # -- timer callbacks --------------------------------------------------
    def _tick(self):
        if not self._paused:
            self._solver.step()
        self._canvas.update()
        self._frame_count += 1

    def _fps(self):
        self._lb_fps.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    # -- stylesheet -------------------------------------------------------
    @staticmethod
    def _stylesheet():
        return """
        QMainWindow, QWidget { background:#1e1e2e; color:#cdd6f4; }
        QGroupBox {
            border:1px solid #45475a; border-radius:6px;
            margin-top:14px; padding:14px 8px 8px 8px;
            font-weight:bold; font-size:12px;
        }
        QGroupBox::title {
            subcontrol-origin:margin; left:12px; padding:0 6px;
        }
        QPushButton {
            background:#313244; border:1px solid #45475a;
            border-radius:5px; padding:6px 12px; color:#cdd6f4;
            font-size:12px;
        }
        QPushButton:hover  { background:#45475a; }
        QPushButton:pressed{ background:#585b70; }
        QPushButton:checked{ background:#585b70; border-color:#89b4fa; }
        QSlider::groove:horizontal {
            height:5px; background:#313244; border-radius:2px;
        }
        QSlider::handle:horizontal {
            background:#89b4fa; width:14px; height:14px;
            margin:-5px 0; border-radius:7px;
        }
        QSlider::handle:horizontal:hover { background:#b4befe; }
        QComboBox {
            background:#313244; border:1px solid #45475a;
            border-radius:5px; padding:5px 8px; color:#cdd6f4;
        }
        QComboBox QAbstractItemView {
            background:#313244; color:#cdd6f4;
            selection-background-color:#585b70;
        }
        QComboBox::drop-down { border:none; }
        QLabel { background:transparent; font-size:12px; }
        """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())