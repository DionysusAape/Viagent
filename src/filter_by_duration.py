#!/usr/bin/env python3
"""
按视频时长筛选生成一个新的数据集 data2。

设计目标
--------
- 保持原有 `data/` 目录完全不动
- 在同级目录下生成 `data2/`，目录结构与 `data/` 一致
- 只保留「视频时长 >= min_duration 秒」的文件

用法示例（在 src 目录下）：
    python filter_by_duration.py

可选参数：
    --src-root      源数据根目录（默认：../data）
    --dst-root      目标数据根目录（默认：../data2）
    --min-duration  最小保留时长（秒，默认：4.0）
    --dry-run       只打印计划结果，不实际复制文件
"""

import argparse
import shutil
from pathlib import Path

from apis.video_io.probe import ensure_ffmpeg, get_video_metadata
from sample_dataset import collect_video_dirs, is_video_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按视频时长筛选生成 data2 数据集（仅保留时长 >= 指定秒数的文件）"
    )

    default_src = (Path(__file__).resolve().parents[1] / "data").as_posix()
    default_dst = (Path(__file__).resolve().parents[1] / "data2").as_posix()

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
        help=f"筛选后数据根目录（默认：{default_dst}）",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="最小保留时长（秒，默认：4.0）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划结果，不实际复制文件",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    min_duration = float(args.min_duration)

    if not src_root.exists():
        raise SystemExit(f"源目录不存在：{src_root}")

    print(f"📂 源数据根目录: {src_root}")
    print(f"📁 目标数据根目录: {dst_root}")
    print(f"⏱  最小保留时长: {min_duration:.3f} 秒")
    if args.dry_run:
        print("🧪 运行模式: dry-run（不实际复制文件）")

    # 确认 ffmpeg / ffprobe 可用
    try:
        ensure_ffmpeg()
    except Exception as e:
        raise SystemExit(f"检测 ffmpeg/ffprobe 失败：{e}")

    video_dirs = collect_video_dirs(src_root)
    if not video_dirs:
        print("⚠️ 未在源目录中找到任何包含视频文件的子目录。")
        return

    total_seen = 0
    total_kept = 0

    for vdir in video_dirs:
        rel_dir = vdir.relative_to(src_root)
        dst_dir = dst_root / rel_dir

        video_files = sorted(
            [p for p in vdir.iterdir() if p.is_file() and is_video_file(p)]
        )
        if not video_files:
            continue

        kept_files = []
        for vp in video_files:
            total_seen += 1
            try:
                meta = get_video_metadata(vp)
                duration = float(meta.get("duration_sec") or 0.0)
            except Exception as e:
                print(f"[跳过] 读取时长失败: {vp} ({e})")
                continue

            if duration >= min_duration:
                kept_files.append(vp)

        if not kept_files:
            print(f"- 目录: {rel_dir} | 符合时长的视频数量: 0 / {len(video_files)}")
            continue

        total_kept += len(kept_files)
        print(
            f"- 目录: {rel_dir} | 符合时长的视频数量: {len(kept_files)} / {len(video_files)}"
        )

        if args.dry_run:
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in kept_files:
            dst_path = dst_dir / src_path.name
            if dst_path.exists():
                continue
            shutil.copy2(src_path, dst_path)

    print("\n✅ 筛选完成")
    print(f"   总视频数: {total_seen}")
    print(f"   保留下的视频数: {total_kept}")
    if total_seen > 0:
        ratio = total_kept / float(total_seen)
        print(f"   保留比例: {ratio:.3f}")


if __name__ == "__main__":
    main()

