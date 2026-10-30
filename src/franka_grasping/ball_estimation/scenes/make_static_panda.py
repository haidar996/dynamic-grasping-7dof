"""Write panda_static.xml: the Panda frozen at Q_HOME with every joint removed (rigid scenery)."""
import sys
from pathlib import Path
import numpy as np
import mujoco

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from shared.franka_common import Q_HOME

SRC, DST = here / "mjx_panda_local.xml", here / "panda_static.xml"


def axis_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    return q


def main():
    spec = mujoco.MjSpec.from_file(str(SRC))
    # the wrist LED (link6_12) is pure green like the ball and fools the colour detector
    spec.material("green").rgba = [0.901961, 0.921569, 0.929412, 1]
    pose = dict(zip([f"joint{i}" for i in range(1, 8)], Q_HOME))
    pose.update(finger_joint1=0.0, finger_joint2=0.0)
    for a in list(spec.actuators):
        spec.delete(a)
    for e in list(spec.equalities):
        spec.delete(e)
    for j in list(spec.joints):
        assert np.allclose(j.pos, 0), "joint offset not supported"
        b = j.parent
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, np.asarray(b.quat, float), axis_quat(j.axis, pose[j.name]))
            b.quat = out
        else:
            b.pos = np.asarray(b.pos) + pose[j.name] * np.asarray(j.axis)
        spec.delete(j)
    spec.compile()
    DST.write_text(spec.to_xml())
    print("wrote", DST)


if __name__ == "__main__":
    main()
