import json
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..sim.sampling import sample_camera_pose
from .robot_tcp import RobotTcp
from .stream import Stream


def vel_transform(
    BvQ = np.zeros(3),
    BwQ = np.zeros(3),
    AvB = np.zeros(3),
    AwB = np.zeros(3),
    ABR = np.eye(3),
    BQ = np.zeros(3)
):
    AvQ = AvB + ABR @ BvQ + np.cross(AwB, ABR @ BQ)
    AwQ = AwB + ABR @ BwQ
    return AvQ, AwQ


class RealEnv(object):
    def __init__(self, args, resample=True, auto_reinit=False):
        self.robot_tcp = RobotTcp(args["jkrobot"]["ip"], args["jkrobot"]["port"])
        self.stream = Stream()
        self.extrinsic = args["extrinsic"]
        # self.target_scene_origin = np.array([-0.300138, -0.371354, 0.260147])
        self.target_scene_origin = np.array([-0.300138, -0.371354, 0.160147])  # under base frame
        self.target_pose_r = [0.1, 0.15]
        self.target_pose_phi = [70, 90] 
        self.target_pose_drz_max = 15
        self.target_pose_dry_max = 5
        self.target_pose_drx_max = 5

        self.initial_pose_r = [0.1, 0.15]
        self.initial_pose_phi = [80, 90]
        self.initial_pose_drz_max = 15
        self.initial_pose_dry_max = 5
        self.initial_pose_drx_max = 5

        self.resample = resample
        self.auto_reinit = auto_reinit

        self.target_base_cam_T = None
        self.initial_base_cam_T = None
        self.current_base_cam_T = None
        self.steps = 0
        self.tcp_cam_T = self.load_extrinsic() ## TODO
        self.dt = 0.05

    def load_extrinsic(self):
        cam2end = np.eye(4)
        R = self.extrinsic["R"]
        t = self.extrinsic["t"]
        cam2end[:3, :3] = R
        cam2end[:3, 3] = t
        return cam2end

    def sample_target_pose(self):
        base_cam_T = sample_camera_pose(
            r_min=self.target_pose_r[0],
            r_max=self.target_pose_r[1],
            phi_min=self.target_pose_phi[0],
            phi_max=self.target_pose_phi[1],
            drz_max=self.target_pose_drz_max,
            dry_max=self.target_pose_dry_max,
            drx_max=self.target_pose_drx_max
        )
        base_cam_T[:3, 3] += self.target_scene_origin
        return base_cam_T
    
    def sample_initial_pose(self):
        base_cam_T = sample_camera_pose(
            r_min=self.initial_pose_r[0],
            r_max=self.initial_pose_r[1],
            phi_min=self.initial_pose_phi[0],
            phi_max=self.initial_pose_phi[1],
            drz_max=self.initial_pose_drz_max,
            dry_max=self.initial_pose_dry_max,
            drx_max=self.initial_pose_drx_max
        )
        base_cam_T[:3, 3] += self.target_scene_origin
        return base_cam_T

    def generate_poses(self, num, seed=None):
        target_poses = np.zeros((num, 4, 4))
        initial_poses = np.zeros((num, 4, 4))
        if seed is not None:
            np.random.seed(seed)
        for i in range(num):
            target_poses[i] = self.sample_target_pose()
            initial_poses[i] = self.sample_initial_pose()
        return target_poses, initial_poses
    def flip_poses(self, poses):

        return poses[:, :, [0, 1, 3, 2]]
    def observation(self):
        bgr, depth = self.stream.get()
        dist_scale = np.median(depth)
        return bgr, depth, dist_scale

    def transform_vel(self, cam_vel_cam):
        cam_vel_cam = np.asarray(cam_vel_cam)
        cam_v_cam, cam_w_cam = cam_vel_cam[:3], cam_vel_cam[3:]

        # Q as cam, B as cam, A as base
        base_cam_R = self.get_current_base_cam_T()[:3, :3]
        base_v_cam, base_w_cam = vel_transform(
            BvQ=cam_v_cam,
            BwQ=cam_w_cam,
            ABR=base_cam_R,
            BQ=np.zeros(3)
        )

        # Q as tcp, B as cam, A as base
        cam_t_tcpORG = np.linalg.inv(self.tcp_cam_T)[:3, 3]
        base_v_tcp, base_w_tcp = vel_transform(
            AvB=base_v_cam,
            AwB=base_w_cam,
            ABR=base_cam_R,
            BQ=cam_t_tcpORG
        )

        base_vel_tcp = np.concatenate([base_v_tcp, base_w_tcp])
        return base_vel_tcp

    def action(self, cam_vel_cam):
        max_vel = 0.1
        max_acc = 0.4

        vel_norm = np.linalg.norm(cam_vel_cam)
        if vel_norm > max_vel:
            cam_vel_cam = cam_vel_cam / (vel_norm + 1e-7) * max_vel

        base_vel_tcp = self.transform_vel(cam_vel_cam)
        if np.any(np.abs(base_vel_tcp) > 1e-5):
            base_vel_tcp[3:] = R.from_rotvec(base_vel_tcp[3:]).as_euler('zyx', degrees = True)[[2, 1, 0]]
            self.robot_tcp.end_vel_jkrobot(base_vel_tcp.tolist())
        return base_vel_tcp

    def is_safe_pose(self):
        base_cam_T = self.get_current_base_cam_T()
        z_axis = base_cam_T[:3, 2]
        proj = z_axis @ np.array([0, 0, -1])
        if proj < 0.6: ## TODO
            print("[INFO] Camera almost parallel to the ground")
            return False

        box_size = 0.3
        tx, ty, tz = base_cam_T[:3, 3]
        ox, oy, oz = self.target_scene_origin

        if tx < ox - box_size / 2. or tx > ox + box_size / 2.:
            print("[INFO] tx (={:.2f}) should in [{:.2f}, {:.2f}]"
                  .format(tx, ox - box_size / 2., ox + box_size / 2.))
            return False

        if ty < oy - box_size / 2. or ty > oy + box_size / 2.:
            print("[INFO] ty (={:.2f}) should in [{:.2f}, {:.2f}]"
                  .format(ty, oy - box_size / 2., oy + box_size / 2.))
            return False

        min_z = min(*self.target_pose_r, *self.initial_pose_r) * 0.5 + self.target_scene_origin[-1]
        max_z = max(*self.target_pose_r, *self.initial_pose_r) * 1.2 + self.target_scene_origin[-1]
        if tz < min_z or tz > max_z:
            print("[INFO] tz (={:.2f}) should be in [{:.2f}, {:.2f}]"
                  .format(tz, min_z, max_z))
            return False

        return True

    def stop_action(self):
        self.robot_tcp.stop_jkrobot()

    def get_current_base_tcp_T(self):
        x, y, z, rx, ry, rz = self.robot_tcp.get_end_pos_jkrobot()
        rot_deg = np.array([rx, ry, rz])

        base_tcp_T = np.eye(4)
        base_tcp_T[:3, :3] = R.from_euler("zyx", rot_deg[[2, 1, 0]], degrees = True).as_matrix()
        base_tcp_T[:3, 3] = [x, y, z]
        return base_tcp_T

    def get_current_base_cam_T(self):
        base_cam_T = self.get_current_base_tcp_T() @ self.tcp_cam_T
        return base_cam_T

    def close_enough(self, xyz, rot):
        cur_pose = self.robot_tcp.get_end_pos_jkrobot()
        dist = np.linalg.norm(cur_pose[:3] - xyz)
        angle = np.linalg.norm(cur_pose[3:] - rot)
        return (angle < 3) and (dist < 3)

    def move_to(self, base_cam_T, sample = True):
        self.stop_action()
        # base_cam_T: camera pose under base frame
        base_tcp_T = base_cam_T @ np.linalg.inv(self.tcp_cam_T)

        xyz = base_tcp_T[:3, 3] ## TODO
        rot = R.from_matrix(base_tcp_T[:3, :3]).as_euler("zyx", degrees = True)[[2 ,1, 0]]
        if sample:
            rot[0] = -rot[0] if rot[0] < 0 else rot[0]
            rot[-1] = -rot[-1] if rot[-1] < 0 else rot[-1]
        pose = np.concatenate([xyz, rot]).tolist()
        self.robot_tcp.end_absmove_jkrobot(pose) ##
        self.robot_tcp.stop_jkrobot() ##

    def pbvs_move_to(self, base_cam_T):
        while True:
            cur_bcT = self.get_current_base_cam_T()
            if self.close_enough(cur_bcT, base_cam_T):
                break
            tcT = np.linalg.inv(base_cam_T) @ cur_bcT
            u = R.from_matrix(tcT[:3, :3]).as_rotvec()
            v = -tcT[:3, :3].T @ tcT[:3, 3]
            w = -u
            cam_vel_cam = np.concatenate([v, w])
            self.action(cam_vel_cam)
        self.stop_action()

    def recover_end_joint_pose(self):
        self.stop_action()
        while True:
            cur_joint_pos = self.robot_tcp.get_joint_pos_jkrobot()
            if np.abs(cur_joint_pos[-1]) > 0.1:
                tar_joint_pos = cur_joint_pos.copy()
                tar_joint_pos[-1] = 0
                self.robot_tcp.joints_move_jkrobot(tar_joint_pos)
            else:
                self.robot_tcp.stop_jkrobot()
                break

    def go(self, pose, z=0.2):
        self.recover_end_joint_pose()
        self.robot_tcp.end_absmove_jkrobot(pose)
        self.robot_tcp.stop_jkrobot()

        
if __name__ == "__main__":
    with open("../../pipeline.json", "r") as fp:
        args = json.load(fp)
        env = RealEnv(args, resample=True, auto_reinit=False)

    home = [-0.35, -0.372, 0.52, 179.96, -1.297, 148.94]
    env.go(pose=home)
    for _ in range(4):
        tar_bcT = env.sample_target_pose()
        env.move_to(tar_bcT)