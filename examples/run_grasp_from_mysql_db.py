import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from one.grasp.mysql_lookup import (
    add_common_task_args,
    add_db_args,
    connect,
    db_config_from_args,
    lookup_solution,
    lookup_solution_by_key,
    make_gripper,
    make_robot,
)
from one import ouc, oum, osso, ossop, ovw


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the bunny grasp task from a MySQL grasp lookup table.'
    )
    add_db_args(parser)
    add_common_task_args(parser)
    parser.add_argument('--object-file', default='bunny.stl')
    parser.add_argument('--object-pos', type=float, nargs=3,
                        default=(-0.50, 0.00, 0.00))
    parser.add_argument('--object-yaw-deg', type=float, default=0.0)
    parser.add_argument('--pose-key', default=None,
                        help=('Lookup an exact generated pose key instead '
                              'of object-pos/yaw.'))
    parser.add_argument('--home-steps', type=int, default=80)
    parser.add_argument('--grasp-steps', type=int, default=120)
    parser.add_argument('--hold-steps', type=int, default=60)
    parser.add_argument('--headless', action='store_true')
    return parser


def interpolate_qs(q0, q1, steps):
    steps = max(int(steps), 1)
    for i in range(steps):
        t = (i + 1) / steps
        yield (1.0 - t) * q0 + t * q1


def apply_pose_from_row(bunny, row):
    pos = np.array(
        [row['object_px'], row['object_py'], row['object_pz']],
        dtype=np.float32,
    )
    quat = np.array(
        [row['object_qx'], row['object_qy'], row['object_qz'], row['object_qw']],
        dtype=np.float32,
    )
    bunny.set_rotmat_pos(oum.rotmat_from_quat(quat), pos)


def run_headless(robot, gripper, goal_qs, jaw_width):
    for qs in interpolate_qs(robot.qs.copy(), goal_qs, 120):
        robot.fk(qs=qs)
    gripper.set_jaw_width(jaw_width)
    print('headless grasp pose applied')


def run_viewer(args, robot, gripper, bunny, row):
    base = ovw.World(
        cam_pos=(2, 2, 1.5),
        cam_lookat_pos=(0, 0, 0.75),
        toggle_auto_cam_orbit=False,
    )
    ossop.frame().attach_to(base.scene)
    robot.attach_to(base.scene)
    gripper.attach_to(base.scene)
    bunny.attach_to(base.scene)
    ossop.plane(pos=(0, 0, 0.01)).attach_to(base.scene)

    goal_qs = row['joint_qs']
    jaw_width = float(row['jaw_width'])
    start_qs = robot.qs.copy()
    timeline = []
    timeline.extend(interpolate_qs(start_qs, goal_qs, args.grasp_steps))
    timeline.extend([goal_qs.copy()] * max(args.hold_steps, 0))
    timeline.extend(interpolate_qs(goal_qs, start_qs, args.home_steps))
    state = {'idx': 0, 'closed': False}

    def tick(dt):
        idx = state['idx']
        if idx >= len(timeline):
            return
        robot.fk(qs=timeline[idx])
        if idx >= args.grasp_steps and not state['closed']:
            gripper.set_jaw_width(jaw_width)
            state['closed'] = True
        state['idx'] += 1

    base.schedule_interval(tick, interval=0.03)
    base.run()


def main():
    args = build_parser().parse_args()
    config = db_config_from_args(args)
    conn = connect(config)
    if args.pose_key:
        row = lookup_solution_by_key(
            conn, config.table,
            args.robot, args.gripper, args.object_name,
            args.pose_key,
        )
    else:
        obj_pos = np.asarray(args.object_pos, dtype=np.float32)
        obj_rot = oum.rotmat_from_axangle(
            ouc.StandardAxis.Z, np.deg2rad(args.object_yaw_deg)
        )
        row = lookup_solution(
            conn, config.table,
            args.robot, args.gripper, args.object_name,
            obj_pos, obj_rot,
            pos_res=args.pos_res,
            quat_res=args.quat_res,
        )
    conn.close()
    if row is None:
        raise RuntimeError(
            'No exact database grasp was found for this quantized object pose. '
            'Generate the pose first with examples/build_grasp_mysql_db.py.'
        )

    robot = make_robot(args.robot)
    gripper = make_gripper(args.gripper)
    robot.engage(gripper)
    bunny = osso.SceneObject.from_file(
        args.object_file, collision_type=ouc.CollisionType.MESH, is_free=True
    )
    apply_pose_from_row(bunny, row)

    print(f'loaded pose_key={row["pose_key"]} score={row["score"]:.4f}')
    if args.headless:
        run_headless(robot, gripper, row['joint_qs'], float(row['jaw_width']))
    else:
        run_viewer(args, robot, gripper, bunny, row)


if __name__ == '__main__':
    main()
