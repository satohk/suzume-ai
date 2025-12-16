# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import cv2
import shutil
import glob
from tqdm import tqdm
import json
import pickle
import argparse
import glob
import gc


def read_video(video_path:str) -> list:
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return 0
    else:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames


def apply_mask_and_crop(frame_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    frame_image: (H, W, rgb)
    mask: (1, H, W)
    """
    # マスクを (H, W) に変換
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask_hw = mask.squeeze(0)
    else:
        raise ValueError(f"mask.shape が (1,H,W) ではありません: {mask.shape}")

    # 出力画像をコピー
    result = frame_image.copy()
    # マスクが False の部分を白に
    result[~mask_hw] = [255, 255, 255]  # BGRで白

    # マスクの外接矩形に切り詰め
    ys, xs = np.where(mask_hw)
    if len(ys) ==0:
        return None
    y0, y1 = ys.min(), ys.max() + 1  # +1 は Python のスライス上限非含め
    x0, x1 = xs.min(), xs.max() + 1
    crop = result[y0:y1, x0:x1]

    return crop
    

def main():
    parser = argparse.ArgumentParser(description="Create dataset from video")
    parser.add_argument("video_dir", help="Path to input video")
    parser.add_argument("segmentation_results_dir", help="Directory to segmentation results pickle")
    parser.add_argument("images_dir", help="Directory to save results")
    parser.add_argument("--start_group_id", default=0, type=int)
    args = parser.parse_args()

    video_dir = args.video_dir
    seg_dir = args.segmentation_results_dir
    res_img_dir_parent = args.images_dir

    group_id = args.start_group_id
    for video_path in sorted(glob.glob(video_dir + "/*.mp4")):
        print(f"process {video_path}")

        # make res dir
        res_img_dir = f"{res_img_dir_parent}/{os.path.basename(video_path).split('.')[0]}"
        if os.path.isdir(res_img_dir):
            print(f"{res_img_dir} is already exist")
            continue
        os.makedirs(res_img_dir, exist_ok=True)

        frames = read_video(video_path)
        for segments_file in sorted(glob.glob(f"{seg_dir}/{os.path.basename(video_path)}_segments*")):
            with open(segments_file, "rb") as f:
                try:
                    segments = pickle.load(f)
                except:
                    print(f"{segments_file} cannot read")
                    continue
            for frame_id in segments:
                frame_image = frames[int(frame_id)]
                for obj_id in segments[frame_id]:
                    mask_applyed_image = apply_mask_and_crop(frame_image, segments[frame_id][obj_id])
                    if mask_applyed_image is None:
                        continue

                    res_img_name = f"group{group_id:08}_label{obj_id:08}_frame{frame_id}.jpg"
                    cv2.imwrite(os.path.join(res_img_dir, res_img_name), mask_applyed_image)
            group_id += 1
        def frames[:]
        del frames
        gc.collect()

if __name__ == "__main__":
    main()
