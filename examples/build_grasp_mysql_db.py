import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The generator is placed under examples/, so we add the repository root above
# before importing the local one package when the file is executed directly.
import one.collider.mj_collider as ocm
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.prm as ompp
import one.scene.scene as oss
from one.grasp.antipodal import antipodal
from one.grasp.mysql_lookup import (
    add_common_task_args,
    add_db_args,
    connect,
    db_config_from_args,
    make_gripper,
    make_robot,
    pose_key,
    positive_int,
    tf_to_pos_quat,
)
from one import ouc, oum, osso, ossop


def build_parser():
    """Create CLI options for the offline database generation job."""
    parser = argparse.ArgumentParser(
        description='Generate grasp/placement rows for robot_manipulation_db.'
    )
    add_db_args(parser)
    parser.set_defaults(db_name='robot_manipulation_db')
    add_common_task_args(parser)
    parser.add_argument('--samples', type=positive_int, default=None,
                        help='Deprecated for grid generation; ignored.')
    parser.add_argument('--commit-every', type=positive_int, default=25)
    parser.add_argument('--object-file', default='bunny.stl')
    parser.add_argument('--grid-center-x', type=float, default=None)
    parser.add_argument('--grid-center-y', type=float, default=None)
    parser.add_argument('--grid-range', type=float, default=0.1)
    parser.add_argument('--grid-step', type=float, default=0.01)
    parser.add_argument('--yaw-step-deg', type=float, default=10.0)
    parser.add_argument('--z-offset', type=float, default=0.0)
    parser.add_argument('--pose-decimals', type=int, default=3)
    parser.add_argument('--antipodal-density', type=float, default=0.01)
    parser.add_argument('--normal-tol-deg', type=float, default=20.0)
    parser.add_argument('--roll-step-deg', type=float, default=30.0)
    parser.add_argument('--max-grasps', type=positive_int, default=120)
    parser.add_argument('--sim-steps', type=int, default=0)
    parser.add_argument('--keys-file', default='generated_grasp_pose_keys.csv')
    parser.add_argument('--print-every', type=positive_int, default=25)
    parser.add_argument('--debug-failures', action='store_true')
    parser.add_argument('--include-ground-collision', action='store_true')
    parser.add_argument('--support-surface', default='Table')
    parser.add_argument('--disable-motion-planning', action='store_true')
    parser.add_argument('--planner-k', type=positive_int, default=15)
    parser.add_argument('--planner-samples', type=positive_int, default=120)
    parser.add_argument('--planner-max-sample-tries', type=positive_int,
                        default=3000)
    parser.add_argument('--cd-step-deg', type=float, default=1.0)
    parser.add_argument('--workers', type=positive_int, default=1)
    parser.add_argument('--chunk-size', type=positive_int, default=16)
    parser.add_argument('--create-schema', action='store_true')
    return parser


def apply_robot_grid_defaults(args):
    """Fill robot-specific grid defaults when the user did not specify them."""
    if args.grid_center_x is None:
        args.grid_center_x = 0.25 if args.robot == 'cvr038' else -0.5
    if args.grid_center_y is None:
        args.grid_center_y = 0.0


def create_robot_manipulation_schema(config):
    """Create the stationary manipulation schema used by this example."""
    conn = connect(config, with_database=False)
    cur = conn.cursor()
    cur.execute(f'CREATE DATABASE IF NOT EXISTS `{config.database}`')
    cur.execute(f'USE `{config.database}`')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ROBOT (
            robot_id INT AUTO_INCREMENT PRIMARY KEY,
            robot_name VARCHAR(100) NOT NULL,
            manipulator_type VARCHAR(100),
            dof INT NOT NULL,
            gripper_type VARCHAR(100),
            payload_kg DECIMAL(6,2),
            description TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS OBJECTS (
            object_id INT AUTO_INCREMENT PRIMARY KEY,
            object_name VARCHAR(100) NOT NULL,
            category VARCHAR(100),
            shape VARCHAR(100),
            weight_kg DECIMAL(6,2),
            mesh_model VARCHAR(255),
            description TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS GRASP (
            grasp_id INT AUTO_INCREMENT PRIMARY KEY,
            object_id INT NOT NULL,
            robot_id INT NOT NULL,
            grasp_pose JSON,
            approach_vector JSON,
            gripper_config JSON,
            quality_score DECIMAL(6,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_grasp_object
                FOREIGN KEY (object_id) REFERENCES OBJECTS(object_id)
                ON DELETE CASCADE,
            CONSTRAINT fk_grasp_robot
                FOREIGN KEY (robot_id) REFERENCES ROBOT(robot_id)
                ON DELETE CASCADE
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS PLACEMENT (
            placement_id INT AUTO_INCREMENT PRIMARY KEY,
            object_id INT NOT NULL,
            placement_pose JSON,
            support_surface VARCHAR(100),
            stability_score DECIMAL(6,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_placement_object
                FOREIGN KEY (object_id) REFERENCES OBJECTS(object_id)
                ON DELETE CASCADE
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ARM_MOTION (
            motion_id INT AUTO_INCREMENT PRIMARY KEY,
            robot_id INT NOT NULL,
            start_joint_config JSON,
            goal_joint_config JSON,
            trajectory JSON,
            trajectory_time DECIMAL(8,3),
            collision_free BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_motion_robot
                FOREIGN KEY (robot_id) REFERENCES ROBOT(robot_id)
                ON DELETE CASCADE
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS TASK (
            task_id INT AUTO_INCREMENT PRIMARY KEY,
            robot_id INT NOT NULL,
            object_id INT NOT NULL,
            grasp_id INT NOT NULL,
            placement_id INT NOT NULL,
            motion_id INT NOT NULL,
            task_type VARCHAR(100),
            status VARCHAR(50) DEFAULT 'planned',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_task_robot
                FOREIGN KEY (robot_id) REFERENCES ROBOT(robot_id),
            CONSTRAINT fk_task_object
                FOREIGN KEY (object_id) REFERENCES OBJECTS(object_id),
            CONSTRAINT fk_task_grasp
                FOREIGN KEY (grasp_id) REFERENCES GRASP(grasp_id),
            CONSTRAINT fk_task_placement
                FOREIGN KEY (placement_id) REFERENCES PLACEMENT(placement_id),
            CONSTRAINT fk_task_motion
                FOREIGN KEY (motion_id) REFERENCES ARM_MOTION(motion_id)
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()


def compute_stable_poses(bunny):
    """Compute physically plausible resting poses for the bunny mesh."""
    geom = bunny.collisions[0].geom
    geom_hull = ogf.convex_hull(geom)
    facets = ogs.segment_surface(geom_hull)
    return ogp.compute_stable_poses(
        geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=10.0
    )


def grid_values(radius, step):
    """Return inclusive grid offsets from -radius to +radius."""
    if radius < 0.0:
        raise ValueError('grid range must be non-negative')
    if step <= 0.0:
        raise ValueError('grid step must be positive')
    n_steps = int(round((2.0 * radius) / step))
    return np.linspace(-radius, radius, n_steps + 1, dtype=np.float32)


def iter_object_pose_grid(stable_poses, args):
    """Yield the exhaustive pose grid around test_rs007l_grasp_motion's pose."""
    pos_local, rot_local, seg_id, stability_score, _ = stable_poses[0]
    center_offset = np.array([
        args.grid_center_x,
        args.grid_center_y,
        args.z_offset,
    ], dtype=np.float32)
    center_pos = pos_local + center_offset
    xy_offsets = grid_values(args.grid_range, args.grid_step)
    yaw_values = np.arange(0.0, 360.0, args.yaw_step_deg, dtype=np.float32)
    for dx in xy_offsets:
        for dy in xy_offsets:
            pos = center_pos + np.array([dx, dy, 0.0], dtype=np.float32)
            for yaw_deg in yaw_values:
                rz = oum.rotmat_from_axangle(
                    ouc.StandardAxis.Z, np.deg2rad(float(yaw_deg))
                )
                # Match the interactive example: rotate the object about its
                # current world position rather than recomputing a placement.
                yield (
                    pos.copy(),
                    rz @ rot_local,
                    int(seg_id),
                    float(stability_score),
                    float(dx),
                    float(dy),
                    float(yaw_deg),
                )


def make_scene(robot, gripper, bunny, include_ground_collision=False):
    """Build the minimal collision scene used by the MuJoCo validator."""
    scene = oss.Scene()
    scene.add(robot)
    scene.add(gripper)
    scene.add(bunny)
    # Ground contact is expected when the bunny is in a stable pose.  Because
    # MJCollider currently reports any MuJoCo contact, including the ground here
    # would reject valid grasps just because the object rests on the table.
    if include_ground_collision:
        scene.add(ossop.plane(pos=(0, 0, 0.01)))
    return scene


def build_collision_context(robot, gripper, bunny, include_ground_collision=False):
    """Compile a MuJoCo collider once and reuse it for all sampled poses."""
    mjc = ocm.MJCollider()
    mjc.scene = make_scene(
        robot, gripper, bunny,
        include_ground_collision=include_ground_collision,
    )
    mjc.actors = [robot]
    mjc.compile(margin=0.0)
    return mjc


def validate_solution(mjc, gripper, qs, jaw_width, sim_steps):
    """Reject IK results that collide in MuJoCo at the pre-grasp state."""
    jaw_qs = (jaw_width * 0.5, jaw_width * 0.5)
    # Keep the gripper opening synchronized with the candidate jaw width before
    # checking the manipulator state.  The robot state is pushed by is_collided.
    mjc.set_mecba_qpos(gripper, jaw_qs)
    if mjc.is_collided(qs):
        return False
    mjc._mjenv.runtime.exit_cd()
    # Optional dynamic stepping is off by default.  It is left here for users
    # who want a slightly stricter MuJoCo sanity check after static collision.
    for _ in range(max(sim_steps, 0)):
        mjc._mjenv.runtime.step()
    mjc.set_mecba_qpos(gripper, jaw_qs)
    return not mjc.is_collided(qs)


def build_planning_context(mjc, gripper, args):
    """Create a reusable planning context for arm trajectory validation."""
    return omppc.PlanningContext(
        collider=mjc,
        aux_mecbas={gripper: gripper.qs.copy()},
        cd_step_size=np.deg2rad(args.cd_step_deg),
    )


def densify_segment(pln_ctx, start_qs, goal_qs):
    """Return joint waypoints at the collision-check resolution."""
    start_qs = np.asarray(start_qs, dtype=np.float32)
    goal_qs = np.asarray(goal_qs, dtype=np.float32)
    dist = pln_ctx.distance(start_qs, goal_qs)
    if dist == 0.0:
        return [start_qs.copy()]
    n_steps = int(np.ceil(dist / pln_ctx.cd_step_size))
    return [
        np.asarray(pln_ctx.interpolate(start_qs, goal_qs, i / n_steps),
                   dtype=np.float32)
        for i in range(n_steps + 1)
    ]


def densify_path(pln_ctx, path):
    """Densify all path segments without duplicating shared waypoints."""
    dense = []
    for i, (q0, q1) in enumerate(zip(path[:-1], path[1:])):
        segment = densify_segment(pln_ctx, q0, q1)
        if i > 0:
            segment = segment[1:]
        dense.extend(segment)
    if not dense and path:
        dense.append(np.asarray(path[0], dtype=np.float32))
    return dense


def plan_collision_free_trajectory(pln_ctx, gripper, start_qs, goal_qs,
                                   jaw_width, args):
    """Return a collision-free joint path to goal_qs, or None if none is found."""
    jaw_qs = (jaw_width * 0.5, jaw_width * 0.5)
    pln_ctx.set_aux_mecbas(gripper, qs=jaw_qs)
    if not pln_ctx.is_state_valid(start_qs):
        return None
    if not pln_ctx.is_state_valid(goal_qs):
        return None
    if args.disable_motion_planning:
        if pln_ctx.is_motion_valid(start_qs, goal_qs):
            return densify_segment(pln_ctx, start_qs, goal_qs)
        return None

    # Prefer the cheap straight-line joint path when it is already safe.
    if pln_ctx.is_motion_valid(start_qs, goal_qs):
        return densify_segment(pln_ctx, start_qs, goal_qs)

    planner = ompp.LazyPRMPlanner(
        pln_ctx=pln_ctx,
        k=args.planner_k,
        n_samples=args.planner_samples,
        max_sample_tries=args.planner_max_sample_tries,
    )
    path = planner.solve(start=start_qs, goal=goal_qs)
    if not path:
        return None
    return densify_path(pln_ctx, path)


def _json_array(values):
    """Convert numpy values to a JSON-friendly list of Python floats."""
    return [float(v) for v in np.asarray(values).reshape(-1)]


def _rounded_array(values, decimals):
    """Return values rounded for compact database storage."""
    return np.round(np.asarray(values, dtype=np.float32), decimals=decimals)


def _round_float(value, decimals):
    return float(np.round(float(value), decimals=decimals))


def _pose_json(pos, quat, decimals):
    """Return a compact position/quaternion pose object."""
    return {
        'position': {
            'x': _round_float(pos[0], decimals),
            'y': _round_float(pos[1], decimals),
            'z': _round_float(pos[2], decimals),
        },
        'quaternion': {
            'x': float(quat[0]),
            'y': float(quat[1]),
            'z': float(quat[2]),
            'w': float(quat[3]),
        },
    }


def _approach_vector(pre_grasp_tf, grasp_tf):
    """Compute the normalized world-space approach vector from pre to grasp."""
    vec = grasp_tf[:3, 3] - pre_grasp_tf[:3, 3]
    length = np.linalg.norm(vec)
    if length <= 0.0:
        return [0.0, 0.0, 0.0]
    return _json_array(vec / length)


def row_from_solution(args, obj_pos, obj_rot, grasp_tf, pre_grasp_tf,
                      qs, jaw_width, score, stable_pose_id, stability_score,
                      start_qs, trajectory, grid_meta):
    """Build JSON payloads matching robot_manipulation_db tables."""
    obj_pos_stored = _rounded_array(obj_pos, args.pose_decimals)
    grasp_tf_stored = grasp_tf.copy()
    pre_grasp_tf_stored = pre_grasp_tf.copy()
    grasp_tf_stored[:3, 3] = _rounded_array(
        grasp_tf_stored[:3, 3], args.pose_decimals
    )
    pre_grasp_tf_stored[:3, 3] = _rounded_array(
        pre_grasp_tf_stored[:3, 3], args.pose_decimals
    )
    obj_quat = tf_to_pos_quat(oum.tf_from_rotmat_pos(obj_rot, obj_pos))[1]
    grasp_pos, grasp_quat = tf_to_pos_quat(grasp_tf)
    pre_pos, pre_quat = tf_to_pos_quat(pre_grasp_tf)
    grasp_pos = grasp_tf_stored[:3, 3]
    pre_pos = pre_grasp_tf_stored[:3, 3]
    key = pose_key(obj_pos_stored, obj_rot, args.pos_res, args.quat_res)
    goal_qs = _json_array(qs)
    start_qs = _json_array(start_qs)
    trajectory = [_json_array(state) for state in trajectory]
    grid_meta = {
        'dx': _round_float(grid_meta['dx'], args.pose_decimals),
        'dy': _round_float(grid_meta['dy'], args.pose_decimals),
        'yaw_deg': float(grid_meta['yaw_deg']),
    }
    return {
        'pose_key': key,
        'placement_pose': {
            'pose_key': key,
            'stable_pose_id': int(stable_pose_id),
            'grid': grid_meta,
            'pose': _pose_json(obj_pos_stored, obj_quat, args.pose_decimals),
        },
        'grasp_pose': {
            'pose_key': key,
            'stable_pose_id': int(stable_pose_id),
            'grid': grid_meta,
            'object_pose': _pose_json(obj_pos_stored, obj_quat, args.pose_decimals),
            'tcp_grasp_pose': _pose_json(grasp_pos, grasp_quat, args.pose_decimals),
            'tcp_pre_grasp_pose': _pose_json(pre_pos, pre_quat, args.pose_decimals),
            'pre_grasp_joint_config': goal_qs,
            'sim_steps': int(args.sim_steps),
        },
        'arm_motion': {
            'pose_key': key,
            'start_joint_config': start_qs,
            'goal_joint_config': goal_qs,
            'trajectory': trajectory,
            'trajectory_time': None,
            'collision_free': True,
            'note': 'Trajectory was collision-checked during DB generation.',
        },
        'approach_vector': _approach_vector(pre_grasp_tf, grasp_tf),
        'gripper_config': {
            'gripper_name': args.gripper,
            'jaw_width': float(jaw_width),
            'jaw_qs': [float(jaw_width * 0.5), float(jaw_width * 0.5)],
        },
        'score': float(score),
        'stability_score': float(stability_score),
    }


def ensure_robot(conn, args, robot):
    """Return an existing or newly inserted ROBOT id."""
    cur = conn.cursor()
    cur.execute(
        'SELECT robot_id FROM ROBOT WHERE robot_name=%s LIMIT 1',
        (args.robot,),
    )
    row = cur.fetchone()
    if row is not None:
        cur.close()
        return int(row[0])
    cur.execute(
        '''INSERT INTO ROBOT
           (robot_name, manipulator_type, dof, gripper_type, description)
           VALUES (%s, %s, %s, %s, %s)''',
        (
            args.robot,
            'Fixed-base manipulator',
            int(robot.ndof),
            args.gripper,
            'Inserted by examples/build_grasp_mysql_db.py',
        ),
    )
    robot_id = int(cur.lastrowid)
    cur.close()
    return robot_id


def ensure_object(conn, args):
    """Return an existing or newly inserted OBJECTS id."""
    cur = conn.cursor()
    cur.execute(
        'SELECT object_id FROM OBJECTS WHERE object_name=%s LIMIT 1',
        (args.object_name,),
    )
    row = cur.fetchone()
    if row is not None:
        cur.close()
        return int(row[0])
    cur.execute(
        '''INSERT INTO OBJECTS
           (object_name, category, shape, mesh_model, description)
           VALUES (%s, %s, %s, %s, %s)''',
        (
            args.object_name,
            'Graspable object',
            'Mesh',
            args.object_file,
            'Inserted by examples/build_grasp_mysql_db.py',
        ),
    )
    object_id = int(cur.lastrowid)
    cur.close()
    return object_id


def insert_placement(conn, object_id, support_surface, row):
    """Insert one sampled object placement."""
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO PLACEMENT
           (object_id, placement_pose, support_surface, stability_score)
           VALUES (%s, %s, %s, %s)''',
        (
            object_id,
            json.dumps(row['placement_pose']),
            support_surface,
            row['stability_score'],
        ),
    )
    placement_id = int(cur.lastrowid)
    cur.close()
    return placement_id


def insert_grasp(conn, object_id, robot_id, placement_id, row):
    """Insert one feasible grasp tied to a sampled placement by JSON metadata."""
    grasp_pose = dict(row['grasp_pose'])
    grasp_pose['placement_id'] = int(placement_id)
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO GRASP
           (object_id, robot_id, grasp_pose, approach_vector,
            gripper_config, quality_score)
           VALUES (%s, %s, %s, %s, %s, %s)''',
        (
            object_id,
            robot_id,
            json.dumps(grasp_pose),
            json.dumps(row['approach_vector']),
            json.dumps(row['gripper_config']),
            row['score'],
        ),
    )
    grasp_id = int(cur.lastrowid)
    cur.close()
    return grasp_id


def insert_arm_motion(conn, robot_id, row):
    """Insert the stored pre-grasp joint target as a minimal motion record."""
    motion = row['arm_motion']
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO ARM_MOTION
           (robot_id, start_joint_config, goal_joint_config, trajectory,
            trajectory_time, collision_free)
           VALUES (%s, %s, %s, %s, %s, %s)''',
        (
            robot_id,
            json.dumps(motion['start_joint_config']),
            json.dumps(motion['goal_joint_config']),
            json.dumps(motion['trajectory']),
            motion['trajectory_time'],
            motion['collision_free'],
        ),
    )
    motion_id = int(cur.lastrowid)
    cur.close()
    return motion_id


def insert_task(conn, robot_id, object_id, grasp_id, placement_id, motion_id):
    """Insert a planned pick task linking placement, grasp, and motion rows."""
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO TASK
           (robot_id, object_id, grasp_id, placement_id, motion_id,
            task_type, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s)''',
        (
            robot_id,
            object_id,
            grasp_id,
            placement_id,
            motion_id,
            'pick',
            'planned',
        ),
    )
    task_id = int(cur.lastrowid)
    cur.close()
    return task_id


def insert_generated_row(conn, robot_id, object_id, args, row):
    """Insert one generated row across placement, grasp, motion, and task."""
    placement_id = insert_placement(
        conn, object_id, args.support_surface, row
    )
    grasp_id = insert_grasp(conn, object_id, robot_id, placement_id, row)
    motion_id = insert_arm_motion(conn, robot_id, row)
    insert_task(
        conn, robot_id, object_id, grasp_id, placement_id, motion_id
    )


def evaluate_pose_item(pose_item, args, robot, gripper, bunny, grasps,
                       mjc, pln_ctx, start_qs):
    """Evaluate one object pose and return a DB row or failure counters."""
    (obj_pos, obj_rot, stable_pose_id, stability_score,
     grid_dx, grid_dy, grid_yaw_deg) = pose_item
    bunny.set_rotmat_pos(obj_rot, obj_pos)
    tf_bunny = oum.tf_from_rotmat_pos(obj_rot, obj_pos)
    chosen = None
    sample_ik_fail = 0
    sample_collision_fail = 0
    sample_motion_fail = 0
    for grasp_tf_local, pre_tf_local, jaw_width, score in grasps:
        # The database stores a pre-grasp joint state, so online execution can
        # skip both grasp generation and inverse kinematics.  It also stores a
        # collision-checked arm path from the configured start.
        pre_tf = tf_bunny @ pre_tf_local
        qs = robot.ik_tcp_nearest(pre_tf[:3, :3], pre_tf[:3, 3])
        if qs is None:
            sample_ik_fail += 1
            continue
        if not validate_solution(mjc, gripper, qs, jaw_width, args.sim_steps):
            sample_collision_fail += 1
            continue
        trajectory = plan_collision_free_trajectory(
            pln_ctx, gripper, start_qs, qs, jaw_width, args
        )
        if trajectory is None:
            sample_motion_fail += 1
            continue
        chosen = (
            tf_bunny @ grasp_tf_local, pre_tf, qs, jaw_width, score,
            trajectory,
        )
        break
    if chosen is None:
        return {
            'row': None,
            'ik_fail': sample_ik_fail,
            'collision_fail': sample_collision_fail,
            'motion_fail': sample_motion_fail,
            'obj_pos': _json_array(obj_pos),
        }
    row = row_from_solution(
        args, obj_pos, obj_rot, chosen[0], chosen[1],
        chosen[2], chosen[3], chosen[4], stable_pose_id,
        stability_score, start_qs, chosen[5],
        {
            'dx': grid_dx,
            'dy': grid_dy,
            'yaw_deg': grid_yaw_deg,
        },
    )
    return {
        'row': row,
        'ik_fail': 0,
        'collision_fail': 0,
        'motion_fail': 0,
        'obj_pos': _json_array(obj_pos),
    }


_WORKER_CONTEXT = {}


def init_pose_worker(args_dict, grasps):
    """Initialize one process-local planning scene for parallel pose evaluation."""
    args = SimpleNamespace(**args_dict)
    robot = make_robot(args.robot)
    gripper = make_gripper(args.gripper)
    robot.engage(gripper)
    start_qs = robot.qs.copy()
    bunny = osso.SceneObject.from_file(
        args.object_file, collision_type=ouc.CollisionType.MESH, is_free=True
    )
    mjc = build_collision_context(
        robot, gripper, bunny,
        include_ground_collision=args.include_ground_collision,
    )
    pln_ctx = build_planning_context(mjc, gripper, args)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update({
        'args': args,
        'robot': robot,
        'gripper': gripper,
        'bunny': bunny,
        'grasps': grasps,
        'mjc': mjc,
        'pln_ctx': pln_ctx,
        'start_qs': start_qs,
    })


def evaluate_pose_chunk_worker(pose_chunk):
    """Evaluate a chunk in a process worker."""
    ctx = _WORKER_CONTEXT
    return [
        evaluate_pose_item(
            pose_item,
            ctx['args'],
            ctx['robot'],
            ctx['gripper'],
            ctx['bunny'],
            ctx['grasps'],
            ctx['mjc'],
            ctx['pln_ctx'],
            ctx['start_qs'],
        )
        for pose_item in pose_chunk
    ]


def iter_chunks(items, chunk_size):
    """Yield fixed-size chunks from a list."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def append_key_row(path, row):
    """Append a compact replay record so pose keys are not lost in stdout."""
    exists = Path(path).exists()
    obj_pose = row['placement_pose']['pose']
    obj_pos = obj_pose['position']
    obj_quat = obj_pose['quaternion']
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                'pose_key', 'score', 'object_px', 'object_py', 'object_pz',
                'object_qx', 'object_qy', 'object_qz', 'object_qw',
            ),
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            'pose_key': row['pose_key'],
            'score': f'{row["score"]:.6f}',
            'object_px': f'{obj_pos["x"]:.6f}',
            'object_py': f'{obj_pos["y"]:.6f}',
            'object_pz': f'{obj_pos["z"]:.6f}',
            'object_qx': f'{obj_quat["x"]:.6f}',
            'object_qy': f'{obj_quat["y"]:.6f}',
            'object_qz': f'{obj_quat["z"]:.6f}',
            'object_qw': f'{obj_quat["w"]:.6f}',
        })


def handle_evaluation_result(conn, robot_id, object_id, args, result, counts,
                             recent_keys, processed):
    """Insert one successful evaluation or account for a failed pose."""
    row = result['row']
    if row is None:
        counts['failed'] += 1
        counts['fail_ik'] += result['ik_fail']
        counts['fail_collision'] += result['collision_fail']
        counts['fail_motion'] += result['motion_fail']
        if args.debug_failures:
            obj_pos = result['obj_pos']
            print(
                f'failed sample={processed} '
                f'ik_fail={result["ik_fail"]} '
                f'collision_fail={result["collision_fail"]} '
                f'motion_fail={result["motion_fail"]} '
                f'obj_pos=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})'
            )
        return

    insert_generated_row(conn, robot_id, object_id, args, row)
    append_key_row(args.keys_file, row)
    counts['inserted'] += 1
    recent_keys.append(row['pose_key'])
    if len(recent_keys) > 5:
        recent_keys.pop(0)
    if counts['inserted'] % args.print_every == 0:
        print(
            f'inserted={counts["inserted"]} failed={counts["failed"]} '
            f'latest_pose_key={row["pose_key"]}'
        )
    if counts['inserted'] % args.commit_every == 0:
        conn.commit()
        print(
            f'committed={counts["inserted"]} failed={counts["failed"]} '
            f'processed={processed} '
            f'ik_fail={counts["fail_ik"]} '
            f'collision_fail={counts["fail_collision"]} '
            f'motion_fail={counts["fail_motion"]}'
        )


def main():
    """Generate object placements and feasible grasps into MySQL."""
    args = build_parser().parse_args()
    apply_robot_grid_defaults(args)
    config = db_config_from_args(args)
    if args.create_schema:
        create_robot_manipulation_schema(config)

    # The online runner must use the same robot/gripper names to retrieve rows.
    robot = make_robot(args.robot)
    gripper = make_gripper(args.gripper)
    robot.engage(gripper)
    start_qs = robot.qs.copy()
    bunny = osso.SceneObject.from_file(
        args.object_file, collision_type=ouc.CollisionType.MESH, is_free=True
    )
    grasps = antipodal(
        gripper=gripper,
        target_sobj=bunny,
        density=args.antipodal_density,
        normal_tol_deg=args.normal_tol_deg,
        roll_step_deg=args.roll_step_deg,
        max_grasps=args.max_grasps,
    )
    # Grasp candidates are generated once in the object's local frame.  Each
    # sampled object pose only needs a transform into the world frame.
    stable_poses = compute_stable_poses(bunny)
    if not grasps:
        raise RuntimeError('No antipodal grasp candidates were generated.')
    if not stable_poses:
        raise RuntimeError('No stable object poses were generated.')
    pose_grid = list(iter_object_pose_grid(stable_poses, args))
    total_samples = len(pose_grid)
    print(
        f'pose grid: total={total_samples} '
        f'range=+/-{args.grid_range:.3f}m '
        f'xy_step={args.grid_step:.3f}m '
        f'yaw_step={args.yaw_step_deg:.1f}deg'
    )

    if args.workers == 1:
        mjc = build_collision_context(
            robot, gripper, bunny,
            include_ground_collision=args.include_ground_collision,
        )
        pln_ctx = build_planning_context(mjc, gripper, args)
    else:
        mjc = None
        pln_ctx = None
        print(
            f'parallel evaluation: workers={args.workers} '
            f'chunk_size={args.chunk_size}'
        )
    conn = connect(config)
    robot_id = ensure_robot(conn, args, robot)
    object_id = ensure_object(conn, args)
    conn.commit()
    counts = {
        'inserted': 0,
        'failed': 0,
        'fail_ik': 0,
        'fail_collision': 0,
        'fail_motion': 0,
    }
    recent_keys = []
    processed = 0
    if args.workers == 1:
        for pose_item in pose_grid:
            processed += 1
            result = evaluate_pose_item(
                pose_item, args, robot, gripper, bunny, grasps,
                mjc, pln_ctx, start_qs
            )
            handle_evaluation_result(
                conn, robot_id, object_id, args, result, counts,
                recent_keys, processed
            )
    else:
        args_dict = vars(args).copy()
        chunks = list(iter_chunks(pose_grid, args.chunk_size))
        with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_pose_worker,
                initargs=(args_dict, grasps)) as executor:
            for results in executor.map(evaluate_pose_chunk_worker, chunks):
                for result in results:
                    processed += 1
                    handle_evaluation_result(
                        conn, robot_id, object_id, args, result, counts,
                        recent_keys, processed
                    )
    conn.commit()
    conn.close()
    print(
        f'done: inserted={counts["inserted"]} '
        f'failed={counts["failed"]} samples={total_samples} '
        f'ik_fail={counts["fail_ik"]} '
        f'collision_fail={counts["fail_collision"]} '
        f'motion_fail={counts["fail_motion"]}'
    )
    print(f'pose keys saved to: {args.keys_file}')
    if recent_keys:
        print('recent pose keys:')
        for key in recent_keys:
            print(f'  {key}')


if __name__ == '__main__':
    main()
