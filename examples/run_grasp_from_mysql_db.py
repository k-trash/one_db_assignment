import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyglet.window.key as key

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The runner lives under examples/, so direct execution needs the repository
# root in sys.path before importing the local one package.
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
from one.grasp.mysql_lookup import (
    add_common_task_args,
    add_db_args,
    connect,
    db_config_from_args,
    make_gripper,
    make_robot,
    pose_key,
)
from one import ouc, oum, osso, ossop, ovw


def build_parser():
    """Create CLI options for DB-backed grasp playback."""
    parser = argparse.ArgumentParser(
        description='Run the rs007l bunny grasp task from robot_manipulation_db.'
    )
    add_db_args(parser)
    parser.set_defaults(db_name='robot_manipulation_db')
    add_common_task_args(parser)
    parser.add_argument('--object-file', default='bunny.stl')
    parser.add_argument('--pose-key', default=None)
    parser.add_argument('--grid-dx', type=float, default=0.0)
    parser.add_argument('--grid-dy', type=float, default=0.0)
    parser.add_argument('--grid-yaw-deg', type=float, default=0.0)
    parser.add_argument('--grid-center-x', type=float, default=None)
    parser.add_argument('--grid-center-y', type=float, default=None)
    parser.add_argument('--z-offset', type=float, default=0.0)
    parser.add_argument('--grid-step', type=float, default=0.01)
    parser.add_argument('--yaw-step-deg', type=float, default=10.0)
    parser.add_argument('--pose-decimals', type=int, default=3)
    parser.add_argument('--hold-steps', type=int, default=60)
    parser.add_argument('--headless', action='store_true')
    return parser


def apply_robot_grid_defaults(args):
    """Fill robot-specific grid defaults when the user did not specify them."""
    if args.grid_center_x is None:
        args.grid_center_x = 0.25 if args.robot == 'cvr038' else -0.5
    if args.grid_center_y is None:
        args.grid_center_y = 0.0


def json_value(value):
    """Decode MySQL JSON values returned as strings or native objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def pose_from_json(pose):
    """Convert a stored JSON pose to rotmat/pos."""
    pos_obj = pose['position']
    quat_obj = pose['quaternion']
    pos = np.array([
        pos_obj['x'],
        pos_obj['y'],
        pos_obj['z'],
    ], dtype=np.float32)
    quat = np.array([
        quat_obj['x'],
        quat_obj['y'],
        quat_obj['z'],
        quat_obj['w'],
    ], dtype=np.float32)
    return oum.rotmat_from_quat(quat), pos


def compute_test_initial_pose(object_file, args):
    """Recreate test_rs007l_grasp_motion.py's initial bunny pose."""
    bunny = osso.SceneObject.from_file(
        object_file, collision_type=ouc.CollisionType.MESH, is_free=True
    )
    geom = bunny.collisions[0].geom
    geom_hull = ogf.convex_hull(geom)
    facets = ogs.segment_surface(geom_hull)
    stable_poses = ogp.compute_stable_poses(
        geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=10.0
    )
    if not stable_poses:
        raise RuntimeError('No stable object poses were generated.')
    pos_local, rotmat, _, _, _ = stable_poses[0]
    pos = pos_local + np.array([
        args.grid_center_x + args.grid_dx,
        args.grid_center_y + args.grid_dy,
        args.z_offset,
    ], dtype=np.float32)
    rz = oum.rotmat_from_axangle(
        ouc.StandardAxis.Z, np.deg2rad(args.grid_yaw_deg)
    )
    return rz @ rotmat, pos


def lookup_row_by_pose_key(conn, args, key_value):
    """Fetch one planned task by the pose_key stored in GRASP JSON."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        '''
        SELECT
            t.task_id,
            g.grasp_id,
            p.placement_id,
            m.motion_id,
            g.quality_score,
            g.grasp_pose,
            g.gripper_config,
            p.placement_pose,
            m.start_joint_config,
            m.goal_joint_config,
            m.trajectory
        FROM TASK t
        JOIN ROBOT r ON t.robot_id = r.robot_id
        JOIN OBJECTS o ON t.object_id = o.object_id
        JOIN GRASP g ON t.grasp_id = g.grasp_id
        JOIN PLACEMENT p ON t.placement_id = p.placement_id
        JOIN ARM_MOTION m ON t.motion_id = m.motion_id
        WHERE r.robot_name = %s
          AND o.object_name = %s
          AND JSON_UNQUOTE(JSON_EXTRACT(g.grasp_pose, '$.pose_key')) = %s
        ORDER BY g.quality_score DESC, t.task_id ASC
        LIMIT 1
        ''',
        (args.robot, args.object_name, key_value),
    )
    row = cur.fetchone()
    cur.close()
    return normalize_db_row(row)


def lookup_row_by_grid(conn, args, dx, dy, yaw_deg):
    """Fetch one planned task by the grid metadata stored in GRASP JSON."""
    dx = round(float(dx), args.pose_decimals)
    dy = round(float(dy), args.pose_decimals)
    yaw_deg = float(yaw_deg) % 360.0
    cur = conn.cursor(dictionary=True)
    cur.execute(
        '''
        SELECT
            t.task_id,
            g.grasp_id,
            p.placement_id,
            m.motion_id,
            g.quality_score,
            g.grasp_pose,
            g.gripper_config,
            p.placement_pose,
            m.start_joint_config,
            m.goal_joint_config,
            m.trajectory
        FROM TASK t
        JOIN ROBOT r ON t.robot_id = r.robot_id
        JOIN OBJECTS o ON t.object_id = o.object_id
        JOIN GRASP g ON t.grasp_id = g.grasp_id
        JOIN PLACEMENT p ON t.placement_id = p.placement_id
        JOIN ARM_MOTION m ON t.motion_id = m.motion_id
        WHERE r.robot_name = %s
          AND o.object_name = %s
          AND CAST(JSON_EXTRACT(g.grasp_pose, '$.grid.dx') AS DECIMAL(10,3)) = %s
          AND CAST(JSON_EXTRACT(g.grasp_pose, '$.grid.dy') AS DECIMAL(10,3)) = %s
          AND CAST(JSON_EXTRACT(g.grasp_pose, '$.grid.yaw_deg') AS DECIMAL(10,3)) = %s
        ORDER BY g.quality_score DESC, t.task_id ASC
        LIMIT 1
        ''',
        (args.robot, args.object_name, dx, dy, yaw_deg),
    )
    row = cur.fetchone()
    cur.close()
    return normalize_db_row(row)


def normalize_db_row(row):
    """Decode JSON columns and expose numpy arrays used by playback."""
    if row is None:
        return None
    row['grasp_pose'] = json_value(row['grasp_pose'])
    row['gripper_config'] = json_value(row['gripper_config'])
    row['placement_pose'] = json_value(row['placement_pose'])
    row['start_joint_config'] = np.asarray(
        json_value(row['start_joint_config']), dtype=np.float32
    )
    row['goal_joint_config'] = np.asarray(
        json_value(row['goal_joint_config']), dtype=np.float32
    )
    row['trajectory'] = [
        np.asarray(qs, dtype=np.float32)
        for qs in json_value(row['trajectory'])
    ]
    row['pose_key'] = row['grasp_pose']['pose_key']
    return row


def apply_row_to_scene(bunny, gripper, row):
    """Apply object pose and gripper opening from one DB row."""
    rotmat, pos = pose_from_json(row['placement_pose']['pose'])
    bunny.set_rotmat_pos(rotmat, pos)
    jaw_width = float(row['gripper_config']['jaw_width'])
    gripper.set_jaw_width(jaw_width)
    return jaw_width


def make_timeline(row, current_qs, hold_steps):
    """Build forward/hold/reverse playback from the stored trajectory."""
    trajectory = [np.asarray(qs, dtype=np.float32) for qs in row['trajectory']]
    if not trajectory:
        trajectory = [current_qs.copy(), row['goal_joint_config'].copy()]
    if np.linalg.norm(current_qs - trajectory[0]) > 1e-4:
        trajectory = [current_qs.copy()] + trajectory
    forward_len = len(trajectory)
    reverse = [qs.copy() for qs in reversed(trajectory[:-1])]
    timeline = trajectory + [trajectory[-1].copy()] * max(hold_steps, 0) + reverse
    return timeline, forward_len


def run_headless(robot, gripper, bunny, row):
    """Apply the stored row without launching the viewer."""
    jaw_width = apply_row_to_scene(bunny, gripper, row)
    for qs in row['trajectory']:
        robot.fk(qs=qs)
    gripper.set_jaw_width(jaw_width)
    print(
        f'headless task applied: task_id={row["task_id"]} '
        f'pose_key={row["pose_key"]}'
    )


def run_viewer(args, robot, gripper, bunny, initial_row):
    """Run a test_rs007l_grasp_motion-like viewer backed by DB rows."""
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

    conn = connect(db_config_from_args(args))
    initial_grid = initial_row['grasp_pose'].get('grid', {})
    state = {
        'grid_dx': float(initial_grid.get('dx', args.grid_dx)),
        'grid_dy': float(initial_grid.get('dy', args.grid_dy)),
        'grid_yaw_deg': float(
            initial_grid.get('yaw_deg', args.grid_yaw_deg)
        ) % 360.0,
        'row': initial_row,
        'timeline': [],
        'forward_len': 0,
        'idx': 0,
        'closed': False,
        'current_qs': robot.qs.copy(),
    }

    def load_row(row):
        jaw_width = apply_row_to_scene(bunny, gripper, row)
        state['row'] = row
        state['timeline'], state['forward_len'] = make_timeline(
            row, state['current_qs'], args.hold_steps
        )
        state['idx'] = 0
        state['closed'] = False
        print(
            f'loaded task_id={row["task_id"]} pose_key={row["pose_key"]} '
            f'score={float(row["quality_score"]):.3f} jaw={jaw_width:.3f}'
        )

    def reload_from_grid():
        row = lookup_row_by_grid(
            conn, args,
            state['grid_dx'],
            state['grid_dy'],
            state['grid_yaw_deg'],
        )
        if row is None:
            print(
                'no DB row for '
                f'dx={state["grid_dx"]:.3f} '
                f'dy={state["grid_dy"]:.3f} '
                f'yaw={state["grid_yaw_deg"]:.1f}'
            )
            return
        load_row(row)

    def move_bunny_once():
        moved = False
        if base.input_manager.is_key_pressed(key.W):
            state['grid_dy'] += args.grid_step
            moved = True
        if base.input_manager.is_key_pressed(key.S):
            state['grid_dy'] -= args.grid_step
            moved = True
        if base.input_manager.is_key_pressed(key.A):
            state['grid_dx'] -= args.grid_step
            moved = True
        if base.input_manager.is_key_pressed(key.D):
            state['grid_dx'] += args.grid_step
            moved = True
        if base.input_manager.is_key_pressed(key.Q):
            state['grid_yaw_deg'] += args.yaw_step_deg
            moved = True
        if base.input_manager.is_key_pressed(key.E):
            state['grid_yaw_deg'] -= args.yaw_step_deg
            moved = True
        if moved:
            state['grid_dx'] = round(state['grid_dx'], args.pose_decimals)
            state['grid_dy'] = round(state['grid_dy'], args.pose_decimals)
            state['grid_yaw_deg'] %= 360.0
            reload_from_grid()

    def tick(dt):
        move_bunny_once()
        timeline = state['timeline']
        if not timeline:
            return
        idx = min(state['idx'], len(timeline) - 1)
        qs = timeline[idx]
        robot.fk(qs=qs)
        state['current_qs'] = qs.copy()
        if idx >= state['forward_len'] and not state['closed']:
            gripper.set_jaw_width(
                float(state['row']['gripper_config']['jaw_width'])
            )
            state['closed'] = True
        if state['idx'] < len(timeline) - 1:
            state['idx'] += 1

    load_row(initial_row)
    base.schedule_interval(tick, interval=0.03)
    try:
        base.run()
    finally:
        conn.close()


def main():
    """Fetch one stored DB task and play its saved collision-free trajectory."""
    args = build_parser().parse_args()
    apply_robot_grid_defaults(args)
    config = db_config_from_args(args)
    conn = connect(config)
    if args.pose_key:
        row = lookup_row_by_pose_key(conn, args, args.pose_key)
    else:
        row = lookup_row_by_grid(
            conn, args, args.grid_dx, args.grid_dy, args.grid_yaw_deg
        )
        if row is None:
            obj_rot, obj_pos = compute_test_initial_pose(args.object_file, args)
            key_value = pose_key(
                np.round(obj_pos, decimals=args.pose_decimals),
                obj_rot,
                args.pos_res,
                args.quat_res,
            )
            row = lookup_row_by_pose_key(conn, args, key_value)
    conn.close()
    if row is None:
        raise RuntimeError(
            'No database task was found. Generate it first with '
            'examples/build_grasp_mysql_db.py.'
        )

    robot = make_robot(args.robot)
    gripper = make_gripper(args.gripper)
    robot.engage(gripper)
    bunny = osso.SceneObject.from_file(
        args.object_file, collision_type=ouc.CollisionType.MESH, is_free=True
    )

    if args.headless:
        run_headless(robot, gripper, bunny, row)
    else:
        run_viewer(args, robot, gripper, bunny, row)


if __name__ == '__main__':
    main()
