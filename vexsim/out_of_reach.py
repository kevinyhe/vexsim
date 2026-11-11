"""Resting pieces out of reach of everything that moves, left out of MuJoCo's
collision checks.

Every step MuJoCo works out a bounding box for each shape of each body that
can collide, and a Cup is 307 shapes.  A piece that is asleep only touches
something awake or something moved by hand (a Loader), and only once the two
come within their shapes' bounding spheres and contact margins of each other.
MuJoCo can wake a sleeping island part way through a step and check
collisions again, so a piece that sleeps with one that could be woken this
step counts as moving, and so does anything near such a piece.  A sleeping
piece out of reach of all of these makes no contact this step whether MuJoCo
checks it or not, so leaving it out changes nothing.

Before each step MuJoCo's kinematics works out where everything is for that
step, as the step itself then does again from the same joint positions, and
the choice is made again once anything moving has moved REACH_SLACK."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import mujoco
import numpy as np

REACH_SLACK = 0.05
SIX = np.arange(6)
SEVEN = np.arange(7)
THREE = np.arange(3)


def _reach(model: mujoco.MjModel, bodies: np.ndarray) -> np.ndarray:
    """How far each body's shapes, with their contact margins, reach from
    its origin.  Each of `bodies` must have a bounding sphere round each shape."""
    unbounded = np.isin(model.geom_bodyid, bodies) & (model.geom_rbound <= 0.0)
    if np.any(unbounded):
        raise ValueError(f"body {model.geom_bodyid[unbounded][0]} moves and has a shape without a bounding sphere, "
                         "such as a plane")
    reach = np.zeros(model.nbody)
    each = (np.sqrt(np.sum(model.geom_pos ** 2, axis=1)) + model.geom_rbound + model.geom_margin
            + model.geom_gap)
    np.maximum.at(reach, model.geom_bodyid, each)
    return reach


def _within(centres: np.ndarray, reach: np.ndarray, others: np.ndarray, other_reach: np.ndarray) -> np.ndarray:
    """Which of `centres` come within REACH_SLACK of touching any of `others`."""
    if len(centres) == 0 or len(others) == 0:
        return np.zeros(len(centres), dtype=bool)
    apart = np.sum((centres[:, None, :] - others[None, :, :]) ** 2, axis=2)
    touching = reach[:, None] + other_reach[None, :] + REACH_SLACK
    return np.any(apart <= touching ** 2, axis=1)


def _sleep_cycles(links: np.ndarray) -> np.ndarray:
    """The smallest tree of each sleeping tree's cycle (MuJoCo's
    `tree_asleep` chains the trees that sleep together), -1 for an awake tree."""
    cycle = np.full(len(links), -1)
    for start in np.nonzero(links >= 0)[0].tolist():
        if cycle[start] >= 0:
            continue
        members = [start]
        tree = int(links[start])
        while tree != start:
            members.append(tree)
            tree = int(links[tree])
        cycle[members] = min(members)
    return cycle


class OutOfReach:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, piece_bodies: Sequence[int],
                 wanted_bits: Dict[int, Tuple[int, int]]):
        """`piece_bodies` are every piece's body, in play or not, and
        `wanted_bits` the (contype, conaffinity) of those in play."""
        self.model, self.data = model, data
        self.wanted_bits = wanted_bits
        pieces = set(piece_bodies)
        collides = np.zeros(model.nbody, dtype=bool)
        collides[model.geom_bodyid[(model.geom_contype != 0) | (model.geom_conaffinity != 0)]] = True
        self.moving_bodies = np.array([b for b in range(model.nbody) if model.body_treeid[b] >= 0
                                       and collides[b] and b not in pieces], dtype=int)
        self.mocap_bodies = np.nonzero(model.body_mocapid >= 0)[0]
        self.reach = _reach(model, np.concatenate([self.moving_bodies, self.mocap_bodies, sorted(pieces)]))
        self.mocap_ids = model.body_mocapid[self.mocap_bodies]
        self.mover_reach = np.concatenate([self.reach[self.moving_bodies], self.reach[self.mocap_bodies]])
        self.in_play = np.zeros(0, dtype=int)
        self.left_out = np.zeros(0, dtype=int)
        self.chosen = False

    def before_step(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        if (not self.chosen or len(self.wanted_bits) != len(self.in_play) or d.ntree_awake != self.trees_awake
                or d.tree_asleep[self.sleeping_trees].tobytes() != self.sleeping_links
                or d.mocap_pos.tobytes() != self.mocap_at):
            self._choose()
            return
        moved = d.xpos[self.moving_bodies] - self.moving_at
        if np.einsum("ij,ij->i", moved, moved).max(initial=0.0) > REACH_SLACK ** 2:
            self._choose()
            return
        if len(self.awake_at):
            moved = d.qpos[self.awake_qpos] - self.awake_at
            if np.einsum("ij,ij->i", moved, moved).max() > REACH_SLACK ** 2:
                self._choose()
                return
        if len(self.left_out) and (d.qvel[self.left_out_dofs].any() or d.qfrc_applied[self.left_out_dofs].any()
                                   or d.xfrc_applied[self.left_out].any()
                                   or d.qpos[self.left_out_qpos].tobytes() != self.left_out_at):
            self._choose()

    def _choose(self) -> None:
        m, d = self.model, self.data
        bodies = np.array(sorted(self.wanted_bits), dtype=int)
        self.in_play = bodies
        links = np.maximum(d.tree_asleep, -1)
        trees = m.body_treeid[bodies]
        dofs = m.jnt_dofadr[m.body_jntadr[bodies]][:, None] + SIX
        qpos = m.jnt_qposadr[m.body_jntadr[bodies]][:, None] + THREE
        disturbed = (np.any(d.qvel[dofs] != 0.0, axis=1) | np.any(d.qfrc_applied[dofs] != 0.0, axis=1)
                     | np.any(d.xfrc_applied[bodies] != 0.0, axis=1))
        resting = (links[trees] >= 0) & ~disturbed
        centres = d.qpos[qpos]
        reach = self.reach[bodies]
        movers = np.vstack([d.xpos[self.moving_bodies], d.mocap_pos[self.mocap_ids], centres[~resting]])
        mover_reach = np.concatenate([self.mover_reach, reach[~resting]])
        near = np.zeros(len(bodies), dtype=bool)
        near[resting] = _within(centres[resting], reach[resting], movers, mover_reach)
        cycle = _sleep_cycles(links)[trees]
        waking = resting & np.isin(cycle, cycle[near | (disturbed & (links[trees] >= 0))])
        quiet = resting & ~waking
        beside = np.zeros(len(bodies), dtype=bool)
        beside[quiet] = _within(centres[quiet], reach[quiet], centres[waking], reach[waking])
        left_out = quiet & ~beside
        contype = np.array([self.wanted_bits[b][0] for b in bodies.tolist()], dtype=m.body_contype.dtype)
        conaffinity = np.array([self.wanted_bits[b][1] for b in bodies.tolist()], dtype=m.body_conaffinity.dtype)
        m.body_contype[bodies] = np.where(left_out, 0, contype)
        m.body_conaffinity[bodies] = np.where(left_out, 0, conaffinity)
        self.trees_awake = d.ntree_awake
        self.sleeping_trees = np.nonzero(links >= 0)[0]
        self.sleeping_links = d.tree_asleep[self.sleeping_trees].tobytes()
        self.mocap_at = d.mocap_pos.tobytes()
        self.moving_at = d.xpos[self.moving_bodies]
        self.awake_qpos = qpos[~resting]
        self.awake_at = centres[~resting]
        self.left_out = bodies[left_out]
        self.left_out_dofs = dofs[left_out].ravel()
        self.left_out_qpos = (m.jnt_qposadr[m.body_jntadr[self.left_out]][:, None] + SEVEN).ravel()
        self.left_out_at = d.qpos[self.left_out_qpos].tobytes()
        self.chosen = True
