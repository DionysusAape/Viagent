#!/usr/bin/env python3
"""
按目录结构等比例抽样，生成一个新的数据集 data1。

设计目标
--------
- 保持原有 `data/` 目录完全不动
- 在同级目录下生成 `data1/`，目录结构与 `data/` 一致
- 对每个“叶子目录”中的视频文件，按固定比例缩小
  - 默认规则：每个目录抽样数量约为  N_dir / base_count
  - 例如：某目录下有 560 个视频，base_count=56 → 抽样数 ≈ 560/56 = 10
  - 每个目录至少保留 1 个样本

用法示例（在 src 目录下）：
    python sample_dataset.py

可选参数：
    --src-root   源数据根目录（默认：../data）
    --dst-root   抽样后数据根目录（默认：../data1）
    --base-count 基准样本数（默认：56）
    --min-per-dir 每个目录最少保留多少个样本（默认：1）
    --dry-run    只打印计划抽样结果，不实际复制文件
"""

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".gif"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def collect_video_dirs(src_root: Path) -> List[Path]:
    """
    收集所有包含视频文件的目录（视为抽样的基本单元）。
    只要目录下直接包含至少一个视频文件，就认为是一个“叶子目录”。
    """
    video_dirs = []
    for dirpath, _dirnames, filenames in os_walk_sorted(src_root):
        p = Path(dirpath)
        if any(is_video_file(p / name) for name in filenames):
            video_dirs.append(p)
    return video_dirs


def os_walk_sorted(root: Path):
    """os.walk 的排序包装，保证每次遍历顺序稳定。"""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


def sample_indices(total: int, base_count: int, min_per_dir: int) -> List[int]:
    """
    根据总数 total 和基准 base_count 计算需要抽样的索引列表。

    规则：
    - 目标数 n_target ≈ total / base_count（四舍五入）
    - 至少 min_per_dir 个，最多不超过 total
    - 均匀采样，覆盖前中后
    """
    if total <= 0:
        return []

    # 计算目标样本数o1
    n_target = max(
        min_per_dir,
        int(round(total / float(base_count))) if base_count > 0 else total,
    )
    n_target = min(n_target, total)

    if n_target == total:
        return list(range(total))

    # 均匀采样索引（不随机），保证稳定复现
    step = total / float(n_target)
    indices = [int(i * step) for i in range(n_target)]
    # 确保最后一个样本是最后一条
    if indices[-1] != total - 1:
        indices[-1] = total - 1

    # 去重并排序（极端情况下 step 可能导致重复）
    indices = sorted(set(indices))
    # 如果去重后数量变少，补齐到 n_target（用随机补）
    while len(indices) < n_target:
        cand = random.randrange(total)
        if cand not in indices:
            indices.append(cand)
    return sorted(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="按目录结构等比例抽样生成 data1 数据集")
    default_src = (Path(__file__).resolve().parents[1] / "data").as_posix()
    default_dst = (Path(__file__).resolve().parents[1] / "data1").as_posix()

    parser.add_argument(
        "--src-root",
        type=str,
        default=default_src,
        help=f"原始数据根目录（默认：{default_src}）",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        default=default_dst,
        help=f"抽样后数据根目录（默认：{default_dst}）",
    )
    parser.add_argument(
        "--base-count",
        type=int,
        default=56,
        help="基准样本数，用于按 total/base_count 计算每个目录抽样数量（默认：56）",
    )
    parser.add_argument(
        "--min-per-dir",
        type=int,
        default=1,
        help="每个目录最少保留的视频数量（默认：1）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于打破均匀采样重复索引时的补齐（默认：42）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划抽样结果，不实际复制文件",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    base_count = args.base_count
    min_per_dir = args.min_per_dir

    if not src_root.exists():
        raise SystemExit(f"源目录不存在：{src_root}")

    random.seed(args.seed)

    print(f"📂 源数据根目录: {src_root}")
    print(f"📁 目标数据根目录: {dst_root}")
    print(f"⚙️  抽样参数: base_count={base_count}, min_per_dir={min_per_dir}, seed={args.seed}")
    if args.dry_run:
        print("🧪 运行模式: dry-run（不实际复制文件）")

    video_dirs = collect_video_dirs(src_root)
    if not video_dirs:
        print("⚠️ 未在源目录中找到任何包含视频文件的子目录。")
        return

    total_original = 0
    total_sampled = 0

    for vdir in video_dirs:
        # 相对路径，用于在 data1 下复刻目录结构
        rel_dir = vdir.relative_to(src_root)
        dst_dir = dst_root / rel_dir

        video_files = sorted(
            [p for p in vdir.iterdir() if p.is_file() and is_video_file(p)]
        )
        n = len(video_files)
        if n == 0:
            continue

        indices = sample_indices(n, base_count, min_per_dir)
        sampled_files = [video_files[i] for i in indices]

        total_original += n
        total_sampled += len(sampled_files)

        print(
            f"- 目录: {rel_dir} | 原始: {n} | 抽样: {len(sampled_files)} (base_count={base_count})"
        )

        if args.dry_run:
            continue

        # 创建目标目录并复制文件
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in sampled_files:
            dst_path = dst_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

    print("\n✅ 抽样完成")
    print(f"   原始视频总数: {total_original}")
    print(f"   抽样后视频总数: {total_sampled}")
    if total_original > 0:
        ratio = total_sampled / float(total_original)
        print(f"   整体缩放比例: {ratio:.3f} (~ 1/{int(round(1/ratio))} 倍)" if ratio > 0 else "")


if __name__ == "__main__":
    import os

    main()

