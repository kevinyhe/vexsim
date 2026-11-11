"""Stop a run the moment MuJoCo says it has gone unstable or run out of memory.

MuJoCo only logs these and carries on.  An unstable world is reset to its
start unless the model turns that off, and contacts and constraints that do
not fit in the arena are dropped."""
from __future__ import annotations

import mujoco

W = mujoco.mjtWarning


class MujocoUnstable(RuntimeError):
    """MuJoCo found a NaN, an infinity or a huge value in the world's state."""


class MujocoOutOfMemory(RuntimeError):
    """The world's contacts or constraints did not fit in MuJoCo's arena."""


UNSTABLE = (W.mjWARN_INERTIA, W.mjWARN_BADQPOS, W.mjWARN_BADQVEL, W.mjWARN_BADQACC, W.mjWARN_BADCTRL)
OUT_OF_MEMORY = (W.mjWARN_CONTACTFULL, W.mjWARN_CNSTRFULL)


def _joint(model: mujoco.MjModel, joint: int) -> str:
    name = model.joint(joint).name
    body = f"body '{model.body(int(model.jnt_bodyid[joint])).name}'"
    return f"joint '{name}' of {body}" if name else body


def _joint_at_qpos(model: mujoco.MjModel, address: int) -> int:
    return max((j for j in range(model.njnt) if model.jnt_qposadr[j] <= address), key=lambda j: model.jnt_qposadr[j])


def _where(model: mujoco.MjModel, warning: int, info: int) -> str:
    if warning == W.mjWARN_BADQPOS:
        return _joint(model, _joint_at_qpos(model, info))
    if warning in (W.mjWARN_BADQVEL, W.mjWARN_BADQACC, W.mjWARN_INERTIA):
        return _joint(model, int(model.dof_jntid[info]))
    if warning == W.mjWARN_BADCTRL:
        return f"actuator '{model.actuator(info).name}'"
    return f"{model.narena / 1e6:.0f} MB arena"


WATCHED = [int(warning) for warning in UNSTABLE + OUT_OF_MEMORY]


def raise_on_warnings(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if not data.warning.number[WATCHED].any():
        return
    for warnings, error in ((UNSTABLE, MujocoUnstable), (OUT_OF_MEMORY, MujocoOutOfMemory)):
        for warning in warnings:
            stat = data.warning[warning]
            if stat.number:
                raise error(f"{mujoco.mju_warningText(warning, stat.lastinfo)} t={data.time:.4f} s, "
                            f"{_where(model, warning, stat.lastinfo)}")
