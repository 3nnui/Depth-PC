import os
import cv2
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
from depth_pc.real.environment_jk import RealEnv
from depth_pc.utils.perception import CameraIntrinsic
from depth_pc.benchmark.benchmark import PickleRecord
from depth_pc.benchmark.stop_policy import PixelStopPolicy
from depth_pc.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from depth_pc.utils.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
matplotlib.use("TkAgg")

cmap = matplotlib.cm.get_cmap('Spectral_r')

def process_depth(depth_img):
    max_value = np.max(depth_img)
    min_value = np.min(depth_img)
    norm_depth = (depth_img - min_value)/(max_value - min_value)
    return (((1-norm_depth)* 255).astype(np.uint8))

manual_mode = False
total_rounds = 50


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

with open("pipeline.json", "r") as fp:
    args = json.load(fp)
    intrinsic = CameraIntrinsic.from_dict(args["intrinsic"])
    env = RealEnv(args, resample=True, auto_reinit=False)
    target_poses_seq, initial_poses_seq = env.generate_poses(total_rounds*2, seed=42)

pipeline = CorrespondenceBasedPipeline.from_file("pipeline_jk5.json")
result_folder = os.path.join(
    "experiment_results_dpt",
    "base2_test",
    "wait02_vel01_cross_attn_depthanything"
)
img_folder = result_folder
pkl_record = PickleRecord(result_folder)
stopper = PixelStopPolicy(waiting_time=0.2, conduct_thresh=0.1)

home = [-0.35, -0.372, 0.42, 179.96, -1.297, 148.94]
env.go(pose=home)
current_round = 0
iter_rounds = True

while iter_rounds:
    if current_round > total_rounds:
        print("[INFO] All test rounds complete, exit")
        break
    print("[INFO] ----------------------------------")
    print("[INFO] Start test round {}/{}".format(current_round, total_rounds))
    env.recover_end_joint_pose()
    tar_bcT = target_poses_seq[current_round-1]
    env.move_to(tar_bcT)
    tar_tcp_T = env.get_current_base_tcp_T()
    print("[INFO] moved to target pose")
    tar_img, tar_depth, dist_scale = env.observation()
    tar_depth = depth_model.infer_image(tar_img)
    tar_depth = process_depth(tar_depth)
    tar_img = np.ascontiguousarray(tar_img[:, :, [2, 1, 0]])
    pipeline.set_target(tar_img, tar_depth)
    ini_bcT = initial_poses_seq[current_round-1]
    env.move_to(ini_bcT)
    print("[INFO] moved to initial pose")
    actual_conduct_vel = np.zeros(6)
    stopper.reset()
    start_time = time.time()
    matches = []
    graphes = []
    depthes = []
    while True:
        cur_bcT = env.get_current_base_cam_T()
        cur_tcp_T = env.get_current_base_tcp_T()
        cur_img, cur_depth, _ = env.observation()
        cur_depth = depth_model.infer_image(cur_img)
        cur_depth = process_depth(cur_depth)
        cur_img = np.ascontiguousarray(cur_img[:, :, [2, 1, 0]])

        vel, data, timing, match, graph = pipeline.get_control_rate(cur_img, cur_depth)
        matches.append(match)
        graphes.append(graph)
        depthes.append(cv2.applyColorMap(cur_depth, cv2.COLORMAP_JET))
        need_stop = (
            stopper(data, time.time()) or 
            not env.is_safe_pose() or
            data is None or
            (time.time() - start_time > 30)
        )
        pose_error = np.linalg.norm(cur_tcp_T[:3,3]-tar_tcp_T[:3,3])
        pkl_record.append_traj(cur_bcT, vel, timing, stopper.cur_err, data)
        
        if need_stop:
            env.stop_action()
            servo_time= time.time() - start_time
            dT = np.linalg.inv(cur_bcT) @ tar_bcT
            du = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec())/np.pi*180
            dt = np.linalg.norm(dT[:3, 3]*1000)            
            du = np.linalg.norm(R.from_matrix(cur_tcp_T[:3,:3]).as_euler("zyx", degrees=True)-
                        R.from_matrix(tar_tcp_T[:3,:3]).as_euler("zyx", degrees=True))
            dt = pose_error*1000            
            if manual_mode:
                input("[INFO] Press Enter to start round {}: ".format(current_round + 1))
            break
        actual_conduct_vel = env.action(vel)
        # break


