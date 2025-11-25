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


def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    SAM2の実行結果のセグメンテーションをマスクとして描画する。

    Args:
        mask (numpy.ndarray): 実行結果のセグメンテーション
        ax (matplotlib.axes._axes.Axes): matplotlibのAxis
        obj_id (int): オブジェクトID
        random_color (bool): マスクの色をランダムにするかどうか
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """
    指定した座標に星を描画する。
    labelsがPositiveの場合は緑、Negativeの場合は赤。

    Args:
        coords (numpy.ndarray): 指定した座標
        labels (numpy.ndarray): Positive or Negative
        ax (matplotlib.axes._axes.Axes): matplotlibのAxis
        marker_size (int, optional): マーカーのサイズ
    """
    print(type(coords))
    print(type(labels))
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """
    指定された矩形を描画する

    Args:
        box (numoy.ndarray): 矩形の座標情報（x_min, y_min, x_max, y_max）
        ax (matplotlib.axes._axes.Axes): matplotlibのAxis
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def read_video(video_path:str, frame_images_dir:str) -> int:
    if not os.path.exists(frame_images_dir):
        os.makedirs(frame_images_dir)

    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return 0
    else:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_name = f'{frame_index:08d}.jpg'
            path = os.path.join(frame_images_dir, frame_name)
            # Save the frame as a JPEG image
            cv2.imwrite(path, frame)
            del frame
            frame_index += 1
        cap.release()
        return frame_index


def copy_frames(frame_images_dir:str, cache_dir:str, start_index:int, end_index:int) -> bool:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for i in range(start_index, end_index):
        frame_name = f'{i:08d}.jpg'
        src_path = os.path.join(frame_images_dir, frame_name)
        dst_path = os.path.join(cache_dir, frame_name)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
    return True


def remove_frames_dir(dir_name:str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def read_frames(cache_dir:str) -> tuple[list, list]:
    frames = []
    frame_names = []

    for filename in sorted(glob.glob(os.path.join(cache_dir, '*.jpg'))):
        img = cv2.imread(filename)
        if img is not None:
            frames.append(img)
            frame_names.append(os.path.basename(filename).split(".")[0])
    return frames, frame_names


def detect_bird_boxes(frames:list) -> list:
    """Faster -R-CNNでの鳥抽出
    """
    # Determine the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Get the class ID for 'bird'
    bird_class_id = 16  # COCO_CLASSES.index('bird')

    # Perform detection on each frame
    bird_boxes = []

    # Set a confidence threshold for detections
    confidence_threshold = 0.8 # Adjust this value as needed

    with torch.no_grad(): # No need to calculate gradients for inference
        for i, frame in tqdm(enumerate(frames)):
            # Convert the frame from OpenCV BGR format to RGB and then to a PyTorch tensor
            # Permute the dimensions from (H, W, C) to (C, H, W) and normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.to(device)

            # Add a batch dimension
            frame_tensor = frame_tensor.unsqueeze(0)

            # Get predictions
            predictions = model(frame_tensor)

            # Filter predictions for 'bird' class with high confidence
            boxes = predictions[0]['boxes']
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']

            bird_detections = boxes[
                (labels == bird_class_id) & (scores > confidence_threshold)
            ]

            # Store the bounding boxes for the current frame
            bird_boxes.append(bird_detections.cpu().tolist())

    return bird_boxes


def save_bird_box_images(frames:list, bird_boxes:list, frame_names:list, bird_box_images_dir:str) -> bool:
    if not os.path.exists(bird_box_images_dir):
        os.makedirs(bird_box_images_dir)

    # セグメンテーション結果を保存
    plt.close("all")
    for frame, bird_box, frame_name in zip(frames, bird_boxes, frame_names):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {frame_name}")
        plt.axis('off')
        plt.tight_layout(pad=0)

        # cv2はデフォルトがBGRのため、RGBに変換してから出力する
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

        # マスクの描画
        for bbox in bird_box:
            show_box(bbox, plt.gca())

        # マスク済みの画像を出力する
        plt.savefig(os.path.join(bird_box_images_dir, frame_name + ".jpg"))
        plt.clf()
        plt.close()
        del image_rgb


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): [x_min, y_min, x_max, y_max]
        box2 (list): [x_min, y_min, x_max, y_max]

    Returns:
        float: The IoU value.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def is_overlapping(box, segments):
    iou_threshold = 0.5  # Define IoU threshold
    for obj_id, mask in segments.items():
        # Convert mask to bounding box for IoU calculation
        # Find the coordinates of the true values in the mask
        coords = np.where(mask)
        if coords[0].size > 0:
            y_min_mask, y_max_mask = np.min(coords[1]), np.max(coords[1])
            x_min_mask, x_max_mask = np.min(coords[2]), np.max(coords[2])
            mask_box = [x_min_mask, y_min_mask, x_max_mask, y_max_mask]

            iou = calculate_iou(box, mask_box)
            if iou > iou_threshold:
                return True
    return False


def out_obj2segment(out_obj_ids, out_mask_logits):
    current_segments = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    return current_segments


def track_object(predictor, frames_dir, frame_names, initial_frame_idx:int, center_x:int, center_y:int, obj_id:int):
    object_segments = {}

    # Initialize the inference state with the first frame
    inference_state = predictor.init_state(frames_dir)

    points = np.array([[center_x, center_y]], dtype=np.float32)
    labels = np.array([1], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=initial_frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )
    object_segments[frame_names[initial_frame_idx]] = out_obj2segment(out_obj_ids, out_mask_logits)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if len(out_obj_ids) == 0:
            break
        object_segments[frame_names[out_frame_idx]] = out_obj2segment(out_obj_ids, out_mask_logits)

    return object_segments


def merge_segments(object_segments, new_object_segments):
    for i, segments in new_object_segments.items():
        object_segments[i] |= segments
    return object_segments


def segment_frames(frames_dir, frame_names, detected_boxes, next_obj_id) -> dict:
    # Determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    print("SAM2 model loaded successfully.")

    object_segments = {n:{} for n in frame_names}
    for frame_idx, (detected_boxes_in_frame, frame_name) in enumerate(zip(detected_boxes, frame_names)):
        print(f"Processing frame {frame_name}")
        for box in detected_boxes_in_frame:
            if is_overlapping(box, object_segments[frame_name]):
                continue

            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            new_object_segments = track_object(predictor, frames_dir, frame_names, frame_idx, center_x, center_y, next_obj_id)
            object_segments = merge_segments(object_segments, new_object_segments)

            next_obj_id += 1
    return object_segments, next_obj_id


def main():
    parser = argparse.ArgumentParser(description="Create dataset from video")
    parser.add_argument("video_dir", help="Path to input video")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    parser.add_argument("--bulk-size", type=int, default=100, help="Number of frames to process in a batch")
    args = parser.parse_args()

    video_dir = args.video_dir
    results_dir = args.results_dir 
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    bulk_size = args.bulk_size
    remove_frames_dir("./temp")
    remove_frames_dir("./frame_images")

    for video_path in sorted(glob.glob(video_dir + "/*.mp4")):
        print(f"process {video_path}")
        num_frames = read_video(video_path, "./frame_images")
        for start_frame in range(0, num_frames, bulk_size):
            segments_file = f"{results_dir}/{os.path.basename(video_path)}_segments_{start_frame:08d}.pickle"
            if os.path.exists(segments_file):
                print(segments_file, "is already exists. skip.")
                continue

            copy_frames("./frame_images", "./temp", start_frame, start_frame+bulk_size)
            frames, frame_names = read_frames("./temp")
            bird_boxes = detect_bird_boxes(frames)
            object_segments, next_obj_id = segment_frames("./temp", frame_names, bird_boxes, start_frame*10)
            remove_frames_dir("./temp")

            # save results
            with open(segments_file, 'wb') as f:
                pickle.dump(object_segments, f)
        remove_frames_dir("./frame_images")


if __name__ == "__main__":
    main()
