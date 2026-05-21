import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import one.collider.mj_collider as ocm
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
import one.scene.scene as oss
from one.grasp.antipodal import antipodal
from one.grasp.mysql_lookup import (
    add_common_task_args,
    add_db_args,
    create_schema,
    db_config_from_args,
    insert_solution,
    make_gripper,
    make_robot,
    pose_key,
    positive_int,
    tf_to_pos_quat,
)
from one import ouc, oum, osso, ossop


def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate a MySQL object-pose to grasp-pose lookup table.'
    )
    add_db_args(parser)
    add_common_task_args(parser)
    parser.add_argument('--samples', type=positive_int, default=200)
    parser.add_argument('--commit-every', type=positive_int, default=25)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--object-file', default='bunny.stl')
    parser.add_argument('--x-min', type=float, default=-0.65)
    parser.add_argument('--x-max', type=float, default=-0.35)
    parser.add_argument('--y-min', type=float, default=-0.20)
    parser.add_argument('--y-max', type=float, default=0.20)
    parser.add_argument('--z-offset', type=float, default=0.0)
    parser.add_argument('--antipodal-density', type=float, default=0.01)
    parser.add_argument('--normal-tol-deg', type=float, default=20.0)
    parser.add_argument('--roll-step-deg', type=float, default=30.0)
    parser.add_argument('--max-grasps', type=positive_int, default=120)
    parser.add_argument('--sim-steps', type=int, default=0)
    parser.add_argument('--create-schema', action='store_true')
    return parser


def compute_stable_poses(bunny):
    geom = bunny.collisions[0].geom
    geom_hull = ogf.convex_hull(geom)
    facets = ogs.segment_surface(geom_hull)
    return ogp.compute_stable_poses(
        geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=10.0
    )


def sample_object_pose(rng, stable_poses, args):
    stable_idx = int(rng.integers(0, len(stable_poses)))
    pos_local, rot_local, seg_id, _, _ = stable_poses[stable_idx]
    yaw = rng.uniform(-np.pi, np.pi)
    rz = oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw)
    table_pos = np.array([
        rng.uniform(args.x_min, args.x_max),
        rng.uniform(args.y_min, args.y_max),
        args.z_offset,
    ], dtype=np.float32)
    return table_pos + rz @ pos_local, rz @ rot_local, int(seg_id)


def make_scene(robot, gripper, bunny):
    scene = oss.Scene()
    scene.add(robot)
    scene.add(gripper)
    scene.add(bunny)
    scene.add(ossop.plane(pos=(0, 0, 0.01)))
    return scene


def build_collision_context(robot, gripper, bunny):
    mjc = ocm.MJCollider()
    mjc.scene = make_scene(robot, gripper, bunny)
    mjc.actors = [robot]
    mjc.compile(margin=0.0)
    return mjc


def validate_solution(mjc, gripper, qs, jaw_width, sim_steps):
    jaw_qs = (jaw_width * 0.5, jaw_width * 0.5)
    mjc.set_mecba_qpos(gripper, jaw_qs)
    if mjc.is_collided(qs):
        return False
    mjc._mjenv.runtime.exit_cd()
    for _ in range(max(sim_steps, 0)):
        mjc._mjenv.runtime.step()
    mjc.set_mecba_qpos(gripper, jaw_qs)
    return not mjc.is_collided(qs)


def row_from_solution(args, obj_pos, obj_rot, grasp_tf, pre_grasp_tf,
                      qs, jaw_width, score, stable_pose_id):
    obj_quat = tf_to_pos_quat(oum.tf_from_rotmat_pos(obj_rot, obj_pos))[1]
    grasp_pos, grasp_quat = tf_to_pos_quat(grasp_tf)
    pre_pos, pre_quat = tf_to_pos_quat(pre_grasp_tf)
    key = pose_key(obj_pos, obj_rot, args.pos_res, args.quat_res)
    return {
        'robot_name': args.robot,
        'gripper_name': args.gripper,
        'object_name': args.object_name,
        'pose_key': key,
        'object_px': float(obj_pos[0]),
        'object_py': float(obj_pos[1]),
        'object_pz': float(obj_pos[2]),
        'object_qx': float(obj_quat[0]),
        'object_qy': float(obj_quat[1]),
        'object_qz': float(obj_quat[2]),
        'object_qw': float(obj_quat[3]),
        'grasp_px': float(grasp_pos[0]),
        'grasp_py': float(grasp_pos[1]),
        'grasp_pz': float(grasp_pos[2]),
        'grasp_qx': float(grasp_quat[0]),
        'grasp_qy': float(grasp_quat[1]),
        'grasp_qz': float(grasp_quat[2]),
        'grasp_qw': float(grasp_quat[3]),
        'pre_grasp_px': float(pre_pos[0]),
        'pre_grasp_py': float(pre_pos[1]),
        'pre_grasp_pz': float(pre_pos[2]),
        'pre_grasp_qx': float(pre_quat[0]),
        'pre_grasp_qy': float(pre_quat[1]),
        'pre_grasp_qz': float(pre_quat[2]),
        'pre_grasp_qw': float(pre_quat[3]),
        'joint_qs_json': json.dumps([float(v) for v in qs]),
        'jaw_width': float(jaw_width),
        'score': float(score),
        'stable_pose_id': int(stable_pose_id),
        'sim_steps': int(args.sim_steps),
    }


def main():
    args = build_parser().parse_args()
    config = db_config_from_args(args)
    if args.create_schema:
        create_schema(config)

    from one.grasp.mysql_lookup import connect

    rng = np.random.default_rng(args.seed)
    robot = make_robot(args.robot)
    gripper = make_gripper(args.gripper)
    robot.engage(gripper)
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
    stable_poses = compute_stable_poses(bunny)
    if not grasps:
        raise RuntimeError('No antipodal grasp candidates were generated.')
    if not stable_poses:
        raise RuntimeError('No stable object poses were generated.')

    mjc = build_collision_context(robot, gripper, bunny)
    conn = connect(config)
    inserted = 0
    failed = 0
    for i in range(args.samples):
        obj_pos, obj_rot, stable_pose_id = sample_object_pose(
            rng, stable_poses, args
        )
        bunny.set_rotmat_pos(obj_rot, obj_pos)
        tf_bunny = oum.tf_from_rotmat_pos(obj_rot, obj_pos)
        chosen = None
        for grasp_tf_local, pre_tf_local, jaw_width, score in grasps:
            pre_tf = tf_bunny @ pre_tf_local
            qs = robot.ik_tcp_nearest(pre_tf[:3, :3], pre_tf[:3, 3])
            if qs is None:
                continue
            if not validate_solution(
                    mjc, gripper, qs, jaw_width, args.sim_steps):
                continue
            chosen = (tf_bunny @ grasp_tf_local, pre_tf, qs, jaw_width, score)
            break
        if chosen is None:
            failed += 1
            continue
        row = row_from_solution(
            args, obj_pos, obj_rot, chosen[0], chosen[1],
            chosen[2], chosen[3], chosen[4], stable_pose_id
        )
        insert_solution(conn, config.table, row)
        inserted += 1
        print(f'inserted pose_key={row["pose_key"]} score={row["score"]:.4f}')
        if inserted % args.commit_every == 0:
            conn.commit()
            print(f'committed={inserted} failed={failed} processed={i + 1}')
    conn.commit()
    conn.close()
    print(
        f'done: inserted_or_updated={inserted} '
        f'failed={failed} samples={args.samples}'
    )


if __name__ == '__main__':
    main()
