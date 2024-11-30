import os
import numpy as np
from typing import Union
from .environment import BenchmarkEnvRender, BenchmarkEnvAffine
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .stop_policy import ErrorHoldingStopPolicy, PixelStopPolicy
from .benchmark import NpzRecord


def run_benchmark(
    env: Union[BenchmarkEnvRender, BenchmarkEnvAffine],
    pipeline: Union[CorrespondenceBasedPipeline, ImageBasedPipeline],
    stop_policy: ErrorHoldingStopPolicy,
    result_folder: str,
    record: bool = False,
    skip_saved: bool = True,
    est_dist=0.7,
):
    npz_recorder = NpzRecord(result_folder) if record else None
    if skip_saved:
        env.skip_indices(env.get_global_indices_of_saved(result_folder))
    
    requires_intrinsic = isinstance(pipeline, CorrespondenceBasedPipeline)
    need_rgb_channel_shuffle = pipeline.REQUIRE_IMAGE_FORMAT != env.RETURN_IMAGE_FORMAT
    print("Need to reorder RGB channel: {}".format(need_rgb_channel_shuffle))
    
    for i in range(len(env)):
        env.clear_debug_items()
        
        tar_img = env.init(i)
        if need_rgb_channel_shuffle:
            tar_img = np.ascontiguousarray(tar_img[:, :, ::-1])
        tPo_norm = est_dist

        if requires_intrinsic:
            pipeline.frontend.intrinsic = env.camera.intrinsic
        pipeline.set_target(tar_img, tPo_norm)
        stop_policy.reset()

        while True:
            cur_img = env.observation()
            if need_rgb_channel_shuffle:
                cur_img = np.ascontiguousarray(cur_img[:, :, ::-1])
            
            vel, data, timing = pipeline.get_control_rate(cur_img)
            need_stop = (
                stop_policy(data, env.steps*env.dt) or 
                env.exceeds_maximum_steps() or
                data is None
            )

            if record:
                npz_recorder.append_traj(
                    env.current_wcT, vel, timing, stop_policy.cur_err)

            if need_stop:
                break
            
            env.action(vel * 2)

        print("[INFO] Round: {}/{}".format(i+1, len(env)))
        print("[INFO] Steps: {}/{}".format(env.steps, env.max_steps))
        env.print_pose_err()
        print("---------------------------------------------------")

        if record:
            npz_recorder.finalize(
                env.prefer_result_fname(i),
                env.target_wcT, 
                env.initial_wcT
            )
