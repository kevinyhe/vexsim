import unittest

import mujoco

from vexsim import cascade, pieces, presets
from vexsim.mujoco_checks import MujocoOutOfMemory, MujocoUnstable, raise_on_warnings
from vexsim.override import OverrideWorld
from vexsim.sim import SimConfig, Simulator
from vexsim.units import INCH
import vexsim.override_field as of


def overlapping_boxes(memory: str) -> mujoco.MjModel:
    boxes = "".join(f'<body pos="{0.03 * (i % 5)} {0.03 * (i // 5 % 5)} {0.05 + 0.03 * (i // 25)}">'
                    f'<freejoint/><geom type="box" size=".02 .02 .02"/></body>' for i in range(100))
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><size memory="{memory}"/><worldbody><geom type="plane" size="1 1 0.1"/>{boxes}</worldbody></mujoco>')


class TestMujocoChecks(unittest.TestCase):
    def test_contacts_that_do_not_fit_in_the_arena_stop_the_run(self):
        model = overlapping_boxes("2M")
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        with self.assertRaisesRegex(MujocoOutOfMemory, r"Too many contacts.*2 MB arena"):
            raise_on_warnings(model, data)

    def test_the_same_contacts_in_a_big_enough_arena_run_on(self):
        model = overlapping_boxes("20M")
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        raise_on_warnings(model, data)

    def test_a_field_of_pieces_that_blows_up_stops_naming_the_piece_at_the_time_it_happened(self):
        world = pieces.Pieces(goals=(), loaders=(), pins=1, cups=1)
        cup = world.add_cup((0.0, 0.0, 0.1), of.UP)
        for _ in range(4):
            world.step()
        world.data.qvel[world.qvel_at[cup.body]] = 1e11
        with self.assertRaisesRegex(MujocoUnstable, r"QVEL .* t=0\.0013 s, body 'cup0'"):
            world.step()

    def test_a_robot_world_that_blows_up_stops_naming_the_robot(self):
        sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
        sim.set_pose(-40.0 * INCH, -12.0 * INCH, 0.0)
        world = OverrideWorld(sim, objects=False, robot=cascade.CascadeRobot())
        world.step(0.001)
        world.data.qvel[world.drive.free_v] = 1e11
        with self.assertRaisesRegex(MujocoUnstable, r"QVEL .* body 'robot'"):
            world.step(0.001)


if __name__ == "__main__":
    unittest.main()
