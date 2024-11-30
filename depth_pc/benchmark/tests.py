import os
from pathlib import Path

import numpy as np
import pybullet as p

from ..ablation import *
from ..reimpl import *
from .benchmark import run_benchmark
from .environment import BenchmarkEnvAffine, BenchmarkEnvRender
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline, VisOpt
from .stop_policy import PixelStopPolicy, SSIMStopPolicy


def get_benchmark_results_root(ckpt_path: str, category: str = None):
    if category is None or len(category) == 0:
        category = "default"
    
    here = os.path.dirname(__file__)
    if ckpt_path is None:
        root = os.path.join(here, "results", category)
    else:
        p = Path(os.path.abspath(ckpt_path))
        root = os.path.join(here, "results", category, p.parent.name)

    return root


def benchmark_scale_info():
    ckpt_paths = [
    ]
    category = "ablation_on_scale_info"

    for ckpt_path in ckpt_paths:
        np.random.seed(0)

        pipeline = CorrespondenceBasedPipeline(
            detector="AKAZE",
            ckpt_path=ckpt_path,
            intrinsic=None,
            # vis=VisOpt.MATCH,
            # vis=VisOpt.KP,
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=0.01
        )

        for scale in [1.0, 0.2, 5.0]:
            env = BenchmarkEnvRender(
                scale=scale,
                section="A"
            )

            result_folder = os.path.join(
                get_benchmark_results_root(ckpt_path, category),
                env.prefer_result_folder()
            )

            print("[INFO] Result folder: {}".format(result_folder))
            run_benchmark(
                env, pipeline, stop_policy, result_folder, 
                record=True,
                skip_saved=True
            )

            p.disconnect()


def benchmark_gvs_on_render_env(ckpt_paths, category):
    for ckpt_path in ckpt_paths:
        pipeline = CorrespondenceBasedPipeline(
            detector="AKAZE",
            ckpt_path=ckpt_path,
            intrinsic=None,
            vis=VisOpt.ALL
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=0.01
        )

        env = BenchmarkEnvRender(
            scale=1.0,
            section="M"
        )

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )
        
        p.disconnect()


def benchmark_data_generation():
    ckpt_paths = [
    ]
    category = "ablation_on_data_generation"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_network_structure():
    ckpt_paths = [
        "checkpoints/09_01_23_28_12/checkpoint_best.pth",

    ]
    category = "ablation_on_network_structure"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_ibvs():
    ckpt_paths = [
    ]
    category = "compare_with_ibvs"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_raft_ibvs():
    ckpt_path = ".cache/torch/hub/checkpoints/raft_ibvs/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    category = "compare_with_raft_ibvs"

    pipeline = ImageBasedPipeline(
        ckpt_path=ckpt_path,
        vis=VisOpt.ALL
    )
    env = BenchmarkEnvRender(
        scale=1.0,
        section="S"
    )
    stop_policy = SSIMStopPolicy(
        waiting_time=0.5, 
        conduct_thresh=0.01
    )

    result_folder = os.path.join(
        get_benchmark_results_root(ckpt_path, category),
        env.prefer_result_folder()
    )

    print("[INFO] Result folder: {}".format(result_folder))
    run_benchmark(
        env, pipeline, stop_policy, result_folder, 
        record=True,
        skip_saved=True
    )

    p.disconnect()


def benchmark_detector():
    ckpt_path = "checkpoints/06_18_20_29_52/checkpoint_best.pth"
    category = "test_different_detector"

    threshs = {
        "ORB": 0.005,
        "SIFT": 0.002,
        "BRISK": 0.002,
        "SuperGlue": 0.01
    }

    for detector in ["ORB", "SIFT", "BRISK", "SuperGlue"]:
        pipeline = CorrespondenceBasedPipeline(
            detector=detector,
            ckpt_path=ckpt_path,
            intrinsic=None,
            # vis=VisOpt.MATCH
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=threshs[detector]
        )

        env = BenchmarkEnvRender(
            scale=1.0,
            section="A"
        )

        result_folder = os.path.join(
            get_benchmark_results_root(None, category),
            detector,
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


def benchmark_seen_image():
    ckpt_paths = [
        "checkpoints/06_18_20_29_52/checkpoint_best.pth"
    ]
    category = "compare_with_image_vs_on_image0"

    for i, ckpt_path in enumerate(ckpt_paths):
        # if i == 0:
        if "graph" in ckpt_path:
            pipeline = CorrespondenceBasedPipeline(
                detector="AKAZE",
                ckpt_path=ckpt_path,
                intrinsic=None,
                vis=VisOpt.MATCH
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=0.8, 
                waiting_time=999999, 
                conduct_thresh=0.1
            )
        else:
            pipeline = ImageBasedPipeline(
                ckpt_path=ckpt_path,
                vis=VisOpt.ALL
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=1.0, 
                waiting_time=999999, 
                conduct_thresh=0.5
            )
        
        env = BenchmarkEnvAffine(
            scale=1.0,
            aug=False,
            one_image=BenchmarkEnvAffine.image0
        )
        env.max_steps = 400  # avoid too much waiting time

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


def benchmark_unseen_image():
    ckpt_paths = [
        "checkpoints/06_18_20_29_52/checkpoint_best.pth"
    ]
    category = "compare_with_image_vs_on_mscoco14_150"

    for i, ckpt_path in enumerate(ckpt_paths):
        if "graph" in ckpt_path:
            pipeline = CorrespondenceBasedPipeline(
                detector="AKAZE",
                ckpt_path=ckpt_path,
                intrinsic=None,
                vis=VisOpt.MATCH
            )
            stop_policy = SSIMStopPolicy(
                waiting_time=999999, 
                conduct_thresh=0.1
            )
        else:
            pipeline = ImageBasedPipeline(
                ckpt_path=ckpt_path,
                vis=VisOpt.ALL
            )
            stop_policy = SSIMStopPolicy(
                waiting_time=999999, 
                conduct_thresh=0.5
            )
        
        env = BenchmarkEnvAffine(
            scale=1.0,
            aug=False,
            one_image=False
        )
        env.max_steps = 400  # avoid waiting too much time

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


if __name__ == "__main__":
    benchmark_network_structure()
    # benchmark_ibvs()

    # benchmark_detector()

    # benchmark_seen_image()
    # benchmark_unseen_image()

    # benchmark_raft_ibvs()
