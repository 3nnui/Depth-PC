import os
import cv2
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
from depth.real.environment_jk import RealEnv
from depth.utils.perception import CameraIntrinsic
from depth.benchmark.benchmark import PickleRecord
from depth.benchmark.stop_policy import PixelStopPolicy
from depth.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from depth.utils.depth_anything_v2.dpt import DepthAnythingV2
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
go_time = time.time()
current_round = 17
iter_rounds = True

while iter_rounds:
    # current_round += 1
    if current_round > total_rounds:
        print("[INFO] All test rounds complete, exit")
        break

    print("[INFO] ----------------------------------")
    print("[INFO] Start test round {}/{}".format(current_round, total_rounds))

    # move to another place
    env.recover_end_joint_pose()
    # tar_bcT = env.sample_target_pose()
    tar_bcT = target_poses_seq[current_round-1]
    env.move_to(tar_bcT)

    # tar_bcT = target_large
    # env.move_to(tar_bcT)

    tar_tcp_T = env.get_current_base_tcp_T()
    print("[INFO] moved to target pose")
    # time.sleep(1.0)
    tar_img, tar_depth, dist_scale = env.observation()
    dist_scale = min(max(0.2, dist_scale), 0.3)
    tar_depth = depth_model.infer_image(tar_img)
    tar_depth = process_depth(tar_depth)

    cv2.imwrite(f"{img_folder}/round_{current_round}_target.png", tar_img)
    cv2.imwrite(f"{img_folder}/round_{current_round}_target_Depth.png", tar_depth)
    tar_img = np.ascontiguousarray(tar_img[:, :, [2, 1, 0]])

    pipeline.set_target(tar_img, tar_depth, dist_scale)
    ini_bcT = initial_poses_seq[current_round-1]
    # ini_bcT = ini_large
    env.move_to(ini_bcT)
    print("[INFO] moved to initial pose")

    if manual_mode:
        input("[INFO] Press Enter to continue: ")

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
        start_dpt = time.time()
        cur_depth = depth_model.infer_image(cur_img)
        cur_depth = process_depth(cur_depth)
        end_dpt = time.time()
        cur_img = np.ascontiguousarray(cur_img[:, :, [2, 1, 0]])

        end_sam = time.time()
        vel, data, timing, match, graph = pipeline.get_control_rate(cur_img, cur_depth)
        matches.append(match)
        graphes.append(graph)
        depthes.append(cv2.applyColorMap(cur_depth, cv2.COLORMAP_JET))
        end_infer = time.time()
        need_stop = (
            stopper(data, time.time()) or 
            not env.is_safe_pose() or
            data is None or
            (time.time() - start_time > 30)
        )
        pose_error = np.linalg.norm(cur_tcp_T[:3,3]-tar_tcp_T[:3,3])
        # need_stop = True if pose_error < 0.01 else False
        print(f"[INFO] pose_error: {pose_error}, stopper: {stopper(data, time.time())}, need_stop: {need_stop}")
        print(f"[INFO] round{current_round} is safe : {env.is_safe_pose()}")
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

        print("[INFO] dist_scale at target = {:.3f}, pred_vel = {}"
              .format(dist_scale, np.round(vel, 2)))
        actual_conduct_vel = env.action(vel)
        # break


