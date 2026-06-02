import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

import one.utils.math as oum
from one import denso_cvr038, khi_rs007l, or_2fg7, xarm_lite6


DEFAULT_TABLE = 'grasp_pose_lookup'


@dataclass(frozen=True)
class DBConfig:
    """Connection settings shared by the offline generator and online runner."""
    host: str
    port: int
    user: str
    password: str
    database: str
    table: str = DEFAULT_TABLE


def add_db_args(parser):
    """Add MySQL CLI options, with environment variables as convenient defaults."""
    parser.add_argument('--db-host', default=os.getenv('ONE_DB_HOST', '127.0.0.1'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('ONE_DB_PORT', '3306')))
    parser.add_argument('--db-user', default=os.getenv('ONE_DB_USER', 'root'))
    parser.add_argument('--db-password', default=os.getenv('ONE_DB_PASSWORD', ''))
    parser.add_argument('--db-name', default=os.getenv('ONE_DB_NAME', 'one_grasp'))
    parser.add_argument('--db-table', default=os.getenv('ONE_DB_TABLE', DEFAULT_TABLE))


def db_config_from_args(args):
    return DBConfig(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        table=args.db_table,
    )


def connect(config, with_database=True):
    """Open a MySQL connection and report a clear error if the driver is missing."""
    try:
        import mysql.connector
    except ImportError as exc:
        raise RuntimeError(
            'mysql-connector-python is required. Install with `pip install -e .` '
            'after updating dependencies, or `pip install mysql-connector-python`.'
        ) from exc
    kwargs = dict(host=config.host, port=config.port,
                  user=config.user, password=config.password)
    if with_database:
        kwargs['database'] = config.database
    return mysql.connector.connect(**kwargs)


def create_schema(config):
    """Create the lookup database/table used for one-to-one pose retrieval."""
    conn = connect(config, with_database=False)
    cur = conn.cursor()
    cur.execute(f'CREATE DATABASE IF NOT EXISTS `{config.database}`')
    cur.execute(f'USE `{config.database}`')
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS `{config.table}` (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            robot_name VARCHAR(64) NOT NULL,
            gripper_name VARCHAR(64) NOT NULL,
            object_name VARCHAR(128) NOT NULL,
            pose_key VARCHAR(160) NOT NULL,
            object_px DOUBLE NOT NULL,
            object_py DOUBLE NOT NULL,
            object_pz DOUBLE NOT NULL,
            object_qx DOUBLE NOT NULL,
            object_qy DOUBLE NOT NULL,
            object_qz DOUBLE NOT NULL,
            object_qw DOUBLE NOT NULL,
            grasp_px DOUBLE NOT NULL,
            grasp_py DOUBLE NOT NULL,
            grasp_pz DOUBLE NOT NULL,
            grasp_qx DOUBLE NOT NULL,
            grasp_qy DOUBLE NOT NULL,
            grasp_qz DOUBLE NOT NULL,
            grasp_qw DOUBLE NOT NULL,
            pre_grasp_px DOUBLE NOT NULL,
            pre_grasp_py DOUBLE NOT NULL,
            pre_grasp_pz DOUBLE NOT NULL,
            pre_grasp_qx DOUBLE NOT NULL,
            pre_grasp_qy DOUBLE NOT NULL,
            pre_grasp_qz DOUBLE NOT NULL,
            pre_grasp_qw DOUBLE NOT NULL,
            joint_qs_json JSON NOT NULL,
            jaw_width DOUBLE NOT NULL,
            score DOUBLE NOT NULL,
            stable_pose_id INT NOT NULL,
            sim_steps INT NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_grasp_pose
                (robot_name, gripper_name, object_name, pose_key),
            INDEX ix_lookup
                (robot_name, gripper_name, object_name, pose_key)
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()


def make_robot(name):
    """Build one of the manipulators supported by this repository."""
    if name == 'rs007l':
        return khi_rs007l.RS007L()
    if name == 'cvr038':
        return denso_cvr038.CVR038()
    if name == 'lite6':
        return xarm_lite6.Lite6()
    raise ValueError(f'Unsupported robot: {name}')


def make_gripper(name):
    """Build the gripper paired with the stored grasp records."""
    if name == 'or2fg7':
        return or_2fg7.OR2FG7()
    raise ValueError(f'Unsupported gripper: {name}')


def normalize_quat(quat):
    """Normalize quaternion sign so equivalent rotations map to one key."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError('zero quaternion')
    quat /= norm
    if quat[3] < 0:
        quat *= -1.0
    return quat


def pose_key(pos, rotmat, pos_res=0.005, quat_res=0.01):
    """Convert a continuous object pose into a deterministic quantized key.

    The database is intentionally an exact lookup table.  Position and
    quaternion components are quantized so repeated measurements that fall in
    the same bin retrieve the same grasp without running IK online.
    """
    pos_i = np.rint(np.asarray(pos, dtype=np.float32) / pos_res).astype(np.int64)
    quat = normalize_quat(oum.quat_from_rotmat(rotmat))
    quat_i = np.rint(quat / quat_res).astype(np.int64)
    vals = list(pos_i) + list(quat_i)
    return ':'.join(str(int(v)) for v in vals)


def tf_to_pos_quat(tf):
    """Split a homogeneous transform into position and normalized quaternion."""
    pos = np.asarray(tf[:3, 3], dtype=np.float32)
    quat = normalize_quat(oum.quat_from_rotmat(tf[:3, :3]))
    return pos, quat


def insert_solution(conn, table, row):
    """Insert one solved pose, replacing the old solution for the same key."""
    cols = (
        'robot_name', 'gripper_name', 'object_name', 'pose_key',
        'object_px', 'object_py', 'object_pz',
        'object_qx', 'object_qy', 'object_qz', 'object_qw',
        'grasp_px', 'grasp_py', 'grasp_pz',
        'grasp_qx', 'grasp_qy', 'grasp_qz', 'grasp_qw',
        'pre_grasp_px', 'pre_grasp_py', 'pre_grasp_pz',
        'pre_grasp_qx', 'pre_grasp_qy', 'pre_grasp_qz', 'pre_grasp_qw',
        'joint_qs_json', 'jaw_width', 'score', 'stable_pose_id', 'sim_steps'
    )
    placeholders = ', '.join(['%s'] * len(cols))
    key_cols = ('robot_name', 'gripper_name', 'object_name', 'pose_key')
    update_cols = [c for c in cols if c not in key_cols]
    update = ', '.join(f'{c}=VALUES({c})' for c in update_cols)
    sql = (
        f'INSERT INTO `{table}` ({", ".join(cols)}) VALUES ({placeholders}) '
        f'ON DUPLICATE KEY UPDATE {update}'
    )
    vals = [row[c] for c in cols]
    cur = conn.cursor()
    cur.execute(sql, vals)
    cur.close()


def lookup_solution(conn, table, robot_name, gripper_name, object_name,
                    pos, rotmat, pos_res=0.005, quat_res=0.01):
    """Lookup by a continuous pose after applying the same quantization."""
    key = pose_key(pos, rotmat, pos_res=pos_res, quat_res=quat_res)
    return lookup_solution_by_key(conn, table, robot_name, gripper_name,
                                  object_name, key)


def lookup_solution_by_key(conn, table, robot_name, gripper_name, object_name,
                           key):
    """Lookup a previously printed/generated pose key exactly."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        f'''SELECT * FROM `{table}`
            WHERE robot_name=%s AND gripper_name=%s AND object_name=%s
              AND pose_key=%s
            LIMIT 1''',
        (robot_name, gripper_name, object_name, key),
    )
    row = cur.fetchone()
    cur.close()
    if row is None:
        return None
    row['joint_qs'] = np.asarray(
        json.loads(row['joint_qs_json']), dtype=np.float32
    )
    row['pose_key'] = key
    return row


def add_common_task_args(parser):
    """Add robot/object options shared by both scripts."""
    parser.add_argument(
        '--robot',
        default='rs007l',
        choices=('rs007l', 'cvr038', 'lite6'),
    )
    parser.add_argument('--gripper', default='or2fg7', choices=('or2fg7',))
    parser.add_argument('--object-name', default='bunny')
    parser.add_argument('--pos-res', type=float, default=0.005)
    parser.add_argument('--quat-res', type=float, default=0.01)


def positive_int(value):
    """argparse validator for counts such as sample size and commit interval."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value
