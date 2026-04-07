#!/usr/bin/env python3
"""
视频分析系统主入口

转换视频请使用（在 src 目录下）：
  python convert.py --label <category>
  python convert.py --label all  # 转换所有视频

用法（在 src 目录下）：
  # 分析单个视频
  python main.py --label Real/msrvtt_899.mp4

  # 分析一个目录（如 Real）
  python main.py --label Real

  # 分析所有视频
  python main.py --label all
"""

import argparse
import json
import time
from pathlib import Path

from database.db_helper import ViagentDB
from pipeline.analyze import analyze_video, list_videos
from util.config import get_batch_delay_sec, load_config
from util.paths import get_data_root


def batch_analyze(
    label: str = "all",
    generator: str | None = None,
    delay: float | None = None,
    config_path: str | None = None,
) -> None:
    """批量分析视频（简单版本，不做额外异常包装）"""
    if delay is None:
        delay = get_batch_delay_sec()

    # 支持路径格式 label，如 "Fake/WildScrape"
    if "/" in label and label != "all":
        parts = label.split("/")
        if len(parts) == 2:
            label, generator = parts[0], parts[1]
            print(f"📝 检测到路径格式，解析为: label={label}, generator={generator}")

    print(f"📹 获取视频列表 (label={label}, generator={generator})...")

    batch_size = 500
    success = 0
    failed_count = 0
    skipped = 0
    total_processed = 0

    db = ViagentDB()
    offset = 0

    # 加载当前使用的 config，并确定实验名（用于实验隔离）
    current_config = load_config(config_path)
    # 默认实验名：来自配置文件名（去掉后缀），如 config.yaml -> config
    if config_path:
        experiment_name = Path(config_path).stem
    else:
        experiment_name = "config"
    # 如果配置中显式提供 experiment_name，则优先生效
    if "experiment_name" in current_config:
        experiment_name = current_config["experiment_name"]

    while True:
        result = list_videos(label=label, generator=generator, limit=batch_size, offset=offset)
        batch = result.get("items", [])
        if not batch:
            break

        offset += len(batch)
        batch_count = len(batch)
        total_processed += batch_count

        print(f"\n📦 批次：获取到 {batch_count} 个视频（累计: {total_processed}）")

        for i, video in enumerate(batch, 1):
            video_id = video["video_id"]
            rel_path = video.get("rel_path", "")
            file_name = video.get("file", "unknown")

            # 检查是否已在当前实验名下分析过（实验隔离）
            runs = db.get_analysis_runs_by_case_id(video_id)
            should_skip = False
            if runs:
                # 检查是否有相同 experiment_name 的分析结果
                for run in runs:
                    # db_helper 已经把 config 解析成 dict，这里直接使用
                    run_config = run.get("config") or {}
                    if not isinstance(run_config, dict):
                        continue

                    run_experiment_name = run_config.get("experiment_name")
                    if run_experiment_name != experiment_name:
                        continue

                    # 找到相同 experiment_name 的分析，检查是否有 verdict
                    verdict = db.get_verdict_by_run_id(run["run_id"])
                    if verdict:
                        should_skip = True
                        break

            if should_skip:
                skipped += 1
                print(f"  [{i}/{batch_count}] {file_name}... ⏭️  已分析（实验: {experiment_name}），跳过")
                continue

            print(f"  [{i}/{batch_count}] {file_name}...", end=" ", flush=True)

            # 从 rel_path 推断真实标签（用于统计）
            parts = rel_path.split("/")
            inferred_label = parts[0].lower() if parts else None
            if inferred_label not in ["real", "fake"]:
                inferred_label = None

            analyze_video(video_path=rel_path, label=inferred_label, config_path=config_path)
            success += 1

            print("✅")

            if delay and delay > 0:
                time.sleep(delay)

        if batch_count < batch_size:
            break

        time.sleep(0.1)

    print("\n" + "=" * 80)
    print("📊 批量分析完成")
    print("=" * 80)
    print(f"   成功: {success}")
    print(f"   失败: {failed_count}")
    print(f"   跳过: {skipped}")
    print(f"   总计: {success + failed_count + skipped}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="分析视频真伪",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="分析目标：单个媒体路径（如 'Real/msrvtt_899.mp4'、'Fake/WildScrape/x.gif'）、目录或 'all'",
    )

    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="假视频生成器名称（仅用于 fake 标签）",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=f"请求间隔秒数（仅用于批量模式，默认: {get_batch_delay_sec()}）",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: src/config/config.yaml）"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果"
    )

    args = parser.parse_args()

    label = args.label

    # 1. 如果是 "all"，批量分析所有视频
    if label.lower() == "all":
        batch_analyze(
            label="all",
            generator=args.generator,
            delay=args.delay,
            config_path=args.config,
        )
    else:
        # 2. 检查是否是单个视频文件（相对 data_root 的路径）
        data_root = get_data_root()
        video_path = data_root / label

        if video_path.is_file():
            # 单个视频模式
            print(f"🔍 正在分析视频: {video_path}")
            result = analyze_video(
                video_path=label,
                label=None,
                config_path=args.config,
            )

            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                verdict = result.get("verdict", {})
                print("\n✅ 分析完成!")
                print(f"   视频ID: {result.get('case', {}).get('case_id', 'N/A')}")
                print(f"   判定: {verdict.get('label', 'N/A')}")
                print(f"   置信度: {verdict.get('confidence', 0):.2f}")
                print(f"   理由: {verdict.get('rationale', 'N/A')}")
                print("\n📊 智能体结果:")
                for agent_key, agent_result in result.get("results", {}).items():
                    status = agent_result.get("status", "unknown")
                    score = agent_result.get("score_fake", "N/A")
                    if isinstance(score, float):
                        score = f"{score:.2f}"
                    print(f"   - {agent_key}: {status}, score_fake={score}")
                print(
                    "\n📁 缓存位置: "
                    f"cache/evidence/{result.get('case', {}).get('case_id', 'N/A')}/"
                )
        else:
            # 批量分析模式（目录或生成器）
            batch_analyze(
                label=label,
                generator=args.generator,
                delay=args.delay,
                config_path=args.config,
            )


if __name__ == "__main__":
    main()
