import numpy as np

import one.collider.mj_collider as ocm
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
import pyglet.window.key as key
from one.grasp.antipodal import antipodal
from one import denso_cvr038, ompp, omppc, or_2fg7
from one import ouc, oum, osso, ossop, ovw


base = ovw.World(
    cam_pos=(1.2, 1.0, 0.9),
    cam_lookat_pos=(0.18, 0.0, 0.28),
    toggle_auto_cam_orbit=False,
)
ossop.frame().attach_to(base.scene)

robot = denso_cvr038.CVR038()
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# CVR038 has no calibrated OR2FG7 flange/collision setup in this repository.
# Keep the gripper in the visual scene and TCP model, but do not include it in
# MuJoCo collision checks; otherwise the mounted gripper can be reported as a
# persistent collision and reject every IK solution.
CHECK_GRIPPER_COLLISION = False

# load bunny
bunny = osso.SceneObject.from_file(
    'bunny.stl', collision_type=ouc.CollisionType.MESH
)
bunny.rgb = (0.8, 0.7, 0.6)
bunny.attach_to(base.scene)

# create ground plane
ground = ossop.plane(pos=(0, 0, 0.01))
ground.attach_to(base.scene)

# setup mj collider
mjc = ocm.MJCollider()
mjc.append(robot)
if CHECK_GRIPPER_COLLISION:
    mjc.append(gripper)
mjc.append(bunny)
# Do not add the ground to the collision checker here.  The bunny is expected
# to rest on the table, and MJCollider currently treats that contact as a
# collision even when the arm itself is clear.
mjc.actors = [robot]
mjc.compile(margin=0.0)

pln_ctx = omppc.PlanningContext(collider=mjc)
planner = ompp.LazyPRMPlanner(pln_ctx=pln_ctx)

# --- compute antipodal grasps on bunny at origin ---
print('Computing antipodal grasps on bunny...')
grasps = antipodal(
    gripper=gripper,
    target_sobj=bunny,
    density=0.005,
    normal_tol_deg=25,
    roll_step_deg=15,
    max_grasps=300,
)
print(f'Found {len(grasps)} collision-free grasps')

# --- compute stable pose (convex hull) ---
geom = bunny.collisions[0].geom
geom_hull = ogf.convex_hull(geom)
facets = ogs.segment_surface(geom_hull)

stable_poses = ogp.compute_stable_poses(
    geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=10.0
)

print(f'Found {len(stable_poses)} stable poses')
if not stable_poses:
    print('No stable poses found, abort.')
    base.run()

# Same idea as the RS007L example, but centered in CVR038's reachable area.
OBJECT_OFFSET = np.array([0.25, 0.0, 0.0], dtype=np.float32)


def solve_prepose_for_tf(tf_bunny):
    """Find one IK/collision-valid pre-grasp for the current object pose."""
    fail_ik = 0
    fail_collision = 0
    for pose, pre_pose, jaw_width, score in grasps:
        pre_pose_world = tf_bunny @ pre_pose
        pre_rot = pre_pose_world[:3, :3]
        pre_pos = pre_pose_world[:3, 3]
        qs_list = robot.ik_tcp(
            tgt_rotmat=pre_rot,
            tgt_pos=pre_pos,
            max_solutions=16,
        )
        if not qs_list:
            fail_ik += 1
            continue
        jaw_qs = (jaw_width / 2, jaw_width / 2)
        if CHECK_GRIPPER_COLLISION:
            mjc.set_mecba_qpos(gripper, jaw_qs)
            pln_ctx.set_aux_mecbas(gripper, qs=jaw_qs)
        for qs in qs_list:
            if not pln_ctx.is_state_valid(qs):
                fail_collision += 1
                continue
            return qs, jaw_qs, fail_ik, fail_collision
    return None, None, fail_ik, fail_collision


def select_initial_pose():
    """Select a stable pose/yaw that is actually reachable by CVR038."""
    total_ik_fail = 0
    total_collision_fail = 0
    yaw_values = np.arange(0.0, 360.0, 15.0, dtype=np.float32)
    for stable_idx, (pos_local, rot_local, seg_id, ratio, _) in enumerate(stable_poses):
        obj_pos = pos_local + OBJECT_OFFSET
        for yaw_deg in yaw_values:
            rz = oum.rotmat_from_axangle(
                ouc.StandardAxis.Z, np.deg2rad(float(yaw_deg))
            )
            obj_rot = rz @ rot_local
            tf_obj = oum.tf_from_rotmat_pos(obj_rot, obj_pos)
            bunny.set_rotmat_pos(obj_rot, obj_pos)
            qs, jaw_qs, fail_ik, fail_collision = solve_prepose_for_tf(tf_obj)
            total_ik_fail += fail_ik
            total_collision_fail += fail_collision
            if qs is None:
                continue
            print(
                f'Selected stable pose: idx={stable_idx} seg={seg_id} '
                f'ratio={ratio:.6f} yaw={yaw_deg:.1f}'
            )
            return tf_obj, qs, jaw_qs, total_ik_fail, total_collision_fail
    return None, None, None, total_ik_fail, total_collision_fail


tf_bunny, goal_qs, aux_qs, fail_ik, fail_collision = select_initial_pose()

# dump all pre-grasp candidates for standalone IK diagnosis
if tf_bunny is not None:
    pre_pose_pos_list = []
    pre_pose_rot_list = []
    jaw_width_list = []
    for pose, pre_pose, jaw_width, score in grasps:
        gl_pre_pose = tf_bunny @ pre_pose
        pre_pose_pos_list.append(gl_pre_pose[:3, 3].astype(np.float32))
        pre_pose_rot_list.append(gl_pre_pose[:3, :3].astype(np.float32))
        jaw_width_list.append(np.float32(jaw_width))
    if len(pre_pose_pos_list) > 0:
        np.savez(
            'cvr038_grasp_candidates.npz',
            pre_pos=np.asarray(pre_pose_pos_list, dtype=np.float32),
            pre_rot=np.asarray(pre_pose_rot_list, dtype=np.float32),
            jaw_width=np.asarray(jaw_width_list, dtype=np.float32),
        )
        print(
            f'Saved {len(pre_pose_pos_list)} candidates '
            f'to cvr038_grasp_candidates.npz'
        )


if goal_qs is None:
    if grasps and tf_bunny is not None:
        pose, pre_pose, jaw_width, score = grasps[0]
        gl_pre_pose = tf_bunny @ pre_pose
        ghost = gripper.clone()
        ghost.grip_at(gl_pre_pose[:3, 3], gl_pre_pose[:3, :3], jaw_width)
        ghost.rgb = (1.0, 0.0, 0.0)
        ghost.alpha = 0.3
        ghost.attach_to(base.scene)
    print(
        'No valid pre-pose IK found. '
        f'ik_fail={fail_ik} collision_fail={fail_collision}'
    )
    base.run()

# --- plan start -> pre and loop ---
start_qs = robot.qs.copy()
path = None
state = start_qs.copy()
cursor = 0
current_target = goal_qs
move_step = 0.01
need_replan = True
drawn_nodes = {}


def move_bunny_once():
    global need_replan, tf_bunny

    moved = False
    pos = np.array(bunny.pos, dtype=np.float32)

    if base.input_manager.is_key_pressed(key.W):
        pos[1] += move_step
        moved = True
    if base.input_manager.is_key_pressed(key.S):
        pos[1] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.A):
        pos[0] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.D):
        pos[0] += move_step
        moved = True
    rot_step = np.deg2rad(10.0)
    if base.input_manager.is_key_pressed(key.Q):
        rz = oum.rotmat_from_euler(0, 0, rot_step)
        bunny.rotmat = rz @ bunny.rotmat
        moved = True
    if base.input_manager.is_key_pressed(key.E):
        rz = oum.rotmat_from_euler(0, 0, -rot_step)
        bunny.rotmat = rz @ bunny.rotmat
        moved = True

    if moved:
        bunny.pos = pos
        tf_bunny[:] = oum.tf_from_rotmat_pos(bunny.rotmat, bunny.pos)
        need_replan = True


def clear_drawn():
    for obj in drawn_nodes.values():
        obj.alpha = 0.0


def tick(dt):
    global path, state, current_target, cursor, goal_qs, aux_qs, need_replan

    move_bunny_once()

    if need_replan:
        goal_qs = None
        aux_qs = None
        clear_drawn()
        max_draw = 10
        count = 0
        for i, (pose, pre_pose, jaw_width, score) in enumerate(grasps):
            pre_pose_world = tf_bunny @ pre_pose
            pre_rot = pre_pose_world[:3, :3]
            pre_pos = pre_pose_world[:3, 3]
            qs_list = robot.ik_tcp(
                tgt_rotmat=pre_rot,
                tgt_pos=pre_pos,
                max_solutions=16,
            )
            if not qs_list:
                continue
            if CHECK_GRIPPER_COLLISION:
                pln_ctx.set_aux_mecbas(
                    gripper, qs=(jaw_width / 2, jaw_width / 2)
                )
            qs = None
            for candidate_qs in qs_list:
                if pln_ctx.is_state_valid(candidate_qs):
                    qs = candidate_qs
                    break
            if qs is None:
                continue
            if i in drawn_nodes:
                tmp = drawn_nodes[i]
            else:
                tmp = robot.clone()
                tmp.attach_to(base.scene)
                drawn_nodes[i] = tmp

            tmp.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp.fk(qs=qs)
            goal_qs = qs
            aux_qs = (jaw_width / 2, jaw_width / 2)

            count += 1
            if count >= max_draw:
                break

        if goal_qs is None:
            path = None
            return

        current_target = goal_qs
        path = None
        cursor = 0
        need_replan = False

    if path is None:
        if CHECK_GRIPPER_COLLISION:
            pln_ctx.set_aux_mecbas(gripper, aux_qs)
        path = planner.solve(start=state, goal=current_target)
        if not path:
            return
        gripper.fk(qs=aux_qs)

    next_idx = min(cursor + 1, len(path) - 1)
    state = path[next_idx]
    cursor = next_idx
    robot.fk(qs=state)

    if pln_ctx.states_equal(state, current_target, tol=5e-3):
        if np.allclose(current_target, goal_qs):
            current_target = start_qs
        else:
            current_target = goal_qs
        path = None
        cursor = 0


base.schedule_interval(tick, interval=0.05)
base.run()
