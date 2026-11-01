"""Ideal (noise-free) overhead depth camera: render -> detect ball -> back-project."""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco


class Pinhole:
    """Camera geometry only (no renderer): safe to use in worker processes."""
    def __init__(self, height=3.0, fovy=60.0, width=640, rows=480, radius=0.03):
        self.W, self.R, self.radius = width, rows, radius
        self.cx, self.cy = (width - 1) / 2, (rows - 1) / 2
        self.set_geometry(height, fovy)

    def set_geometry(self, height, fovy):
        self.H, self.fovy = height, fovy
        self.f = (self.R / 2) / np.tan(np.radians(fovy) / 2)    # focal length, pixels

    def back_project(self, u, v, z_surface):
        """Pixel + surface depth -> world (x, y) of the ball centre.
        The pixel ray through the centroid meets the sphere r before the centre, so along the optical axis the centre
        is r*cos(theta) deeper than the surface point (theta = ray angle).  Adding the full r (the old rule, exact only
        straight below the camera) biased the position outward by up to 3 mm near the walls (audit B2, 2026-11-01)."""
        a, b = (u - self.cx) / self.f, (v - self.cy) / self.f
        zc = z_surface + self.radius / np.sqrt(1.0 + a * a + b * b)
        return np.array([a * zc, -b * zc])                    # image rows grow downward

    def back_project_plane(self, u, v, dz=0.0):
        """Pixel -> world (x, y) by intersecting the ray with the plane of the ball centre (z = radius); no depth used.
        dz: error of the assumed camera-to-plane distance (m), for calibration-error tests (audit B1)."""
        zc = self.H - self.radius + dz
        return np.array([(u - self.cx) * zc / self.f, -(v - self.cy) * zc / self.f])


class DepthCamera(Pinhole):
    def __init__(self, model, height=3.0, fovy=60.0, width=640, rows=480, name="overhead"):
        super().__init__(height, fovy, width, rows, model.geom("ball_geom").size[0])
        self.model, self.name = model, name
        self.renderer = mujoco.Renderer(model, rows, width)     # create ONCE and reuse
        self.configure(height, fovy)

    def configure(self, height, fovy):
        """Set camera height (m) and vertical field of view (deg); renderer is reused."""
        cam = self.model.cam(self.name)
        cam.pos[:] = [0.0, 0.0, height]            # look straight down from this height
        cam.fovy[:] = fovy
        self.set_geometry(height, fovy)

    def close(self):
        self.renderer.close()

    def render(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        rgb = self.renderer.render().astype(int)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().copy()       # metres
        return rgb, depth

    def detect(self, rgb, min_pixels=3):
        """Colour mask (green dominates) -> centroid (u, v) in pixels, or None."""
        g = (rgb[..., 1] > rgb[..., 0] + 40) & (rgb[..., 1] > rgb[..., 2] + 40)
        if g.sum() < min_pixels:
            return None
        v, u = np.nonzero(g)
        return u.mean(), v.mean()

    def measure_parts(self, data):
        """Ideal detection: (u, v, z_surface) or None if the ball is not seen."""
        rgb, depth = self.render(data)
        uv = self.detect(rgb)
        if uv is None:
            return None
        return uv[0], uv[1], depth[int(round(uv[1])), int(round(uv[0]))]

    def measure(self, data):
        """Ideal measurement: world (x, y) of the ball, or None if not seen."""
        parts = self.measure_parts(data)
        return None if parts is None else self.back_project(*parts)
