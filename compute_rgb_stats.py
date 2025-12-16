
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
フォルダ以下の全 JPG/JPEG を対象に、RGB各チャネルの mean と std（0-1 スケール）を計算します。
- Exif の向きを補正
- Alpha は無視（RGB 3 チャネル化）
- 例外発生画像は警告してスキップ
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import random
import sys

def iter_image_paths(root: Path, exts=("jpg", "jpeg"), recursive=True, include_png=False):
    if include_png:
        exts = tuple(list(exts) + ["png"])
    if recursive:
        # 大文字拡張子も拾うため rglob して後段でフィルタ
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                yield p
    else:
        for p in root.glob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                yield p

def load_rgb_array(path: Path, resize=None):
    # 画像読み込み・向き補正・RGB 変換
    with Image.open(path) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            # Exif なし等は無視
            pass
        img = img.convert("RGB")
        if resize is not None:
            # DL 前処理に合わせたい場合：例) (224, 224)、(256,256)→CenterCrop など
            img = img.resize(resize, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # 0-1 スケール
        # arr.shape: (H, W, 3)
        return arr

def compute_mean_std(root_dir: str, recursive=True, include_png=False, resize=None, sample_ratio=1.0, seed=42):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {root_dir}")

    paths = list(iter_image_paths(root, recursive=recursive, include_png=include_png))
    if len(paths) == 0:
        raise RuntimeError("対象画像が見つかりませんでした。拡張子とフォルダをご確認ください。")

    # サンプリング（高速化用）
    if sample_ratio < 1.0:
        random.seed(seed)
        k = max(1, int(len(paths) * sample_ratio))
        paths = random.sample(paths, k)

    # 進捗バー（tqdm があれば使用）
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Processing images")
    except Exception:
        iterator = paths

    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    skipped = 0

    for p in iterator:
        try:
            arr = load_rgb_array(p, resize=resize)
        except Exception as e:
            skipped += 1
            print(f"[WARN] 読み込み失敗: {p} ({e})", file=sys.stderr)
            continue

        # (H, W, 3) → チャンネル毎の合計・二乗和
        H, W, _ = arr.shape
        # sum over H and W
        ch_sum += arr.sum(axis=(0, 1), dtype=np.float64)
        ch_sq_sum += (arr ** 2).sum(axis=(0, 1), dtype=np.float64)
        pixel_count += H * W

    if pixel_count == 0:
        raise RuntimeError("有効なピクセルがありません（すべて失敗）。")

    mean = ch_sum / pixel_count
    var = ch_sq_sum / pixel_count - mean ** 2
    # 数値誤差で負になる可能性に備えクリップ
    var = np.clip(var, 0, None)
    std = np.sqrt(var)

    return {
        "count_images": len(paths),
        "skipped_images": skipped,
        "pixel_count": int(pixel_count),
        "mean_rgb": mean,  # R, G, B 順
        "std_rgb": std,    # R, G, B 順
    }

def main():
    parser = argparse.ArgumentParser(description="フォルダ以下の JPG の RGB mean/std（0-1 スケール）を計算します。")
    parser.add_argument("folder", help="画像フォルダのパス")
    parser.add_argument("--no-recursive", action="store_true", help="再帰せず直下のみを対象にします")
    parser.add_argument("--include-png", action="store_true", help="PNG も対象に含めます")
    parser.add_argument("--resize", type=str, default=None, help="リサイズ指定 例: 224x224（幅x高さ）")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="サンプル比率（0-1、速度優先時）")
    parser.add_argument("--seed", type=int, default=42, help="サンプリングの乱数シード")

    args = parser.parse_args()

    resize = None
    if args.resize:
        try:
            w, h = args.resize.lower().split("x")
            resize = (int(w), int(h))
        except Exception:
            print("[ERROR] --resize は '幅x高さ' 形式で指定してください（例: 224x224）", file=sys.stderr)
            sys.exit(1)

    stats = compute_mean_std(
        root_dir=args.folder,
        recursive=not args.no_recursive,
        include_png=args.include_png,
        resize=resize,
        sample_ratio=args.sample_ratio,
        seed=args.seed
    )

    mean = stats["mean_rgb"]
    std = stats["std_rgb"]
    print("\n=== 結果 ===")
    print(f"画像枚数（対象）：{stats['count_images']}  / スキップ：{stats['skipped_images']}")
    print(f"総ピクセル数：{stats['pixel_count']}")
    print(f"mean (R, G, B)：[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"std  (R, G, B)：[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")

    print("\nPyTorch での Normalize 例：")
    print(f"transforms.Normalize(mean={[round(float(x),6) for x in mean]}, std={[round(float(x),6) for x in std]})")

if __name__ == "__main__":
    main()
