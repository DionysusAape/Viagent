#!/usr/bin/env python3
"""
批量转换视频为 LLM 友好格式（支持断点续传）
用法（在 src 目录下）：
  python convert.py [选项]
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Set
from pipeline.analyze import list_videos
from pipeline.evidence import is_evidence_ready, build_llm_inputs
from util.config import get_batch_delay_sec
from util.paths import PROGRESS_DIR

PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def load_progress(label: str, generator: str = None) -> tuple[Set[str], list]:
    """加载已处理的视频ID列表和失败列表
    
    统一从 all_progress.json 读取进度，并根据 label/generator 过滤
    """
    processed = set()
    failed = []
    
    # 统一从 all_progress.json 读取进度
    all_progress_file = _get_progress_file("all", None)
    if all_progress_file.exists():
        try:
            with open(all_progress_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                # 只合并属于当前 label/generator 的视频
                all_processed = set(all_data.get("processed", []))
                all_failed = all_data.get("failed", [])
                
                # 如果 label == "all"，返回所有视频
                if label == "all":
                    processed = all_processed
                    failed = all_failed
                else:
                    # 过滤出属于当前 label/generator 的视频
                    for video_id in all_processed:
                        # 通过 video_id 判断是否属于当前 label/generator
                        # video_id 是 base64 编码的 rel_path，需要解码检查
                        try:
                            decoded = base64.b64decode(video_id).decode('utf-8')
                            parts = decoded.split('/')
                            if len(parts) >= 2:
                                vid_label = parts[0]
                                vid_generator = parts[1] if len(parts) >= 3 else None
                                if vid_label.lower() == label.lower():
                                    if generator is None or vid_generator == generator:
                                        processed.add(video_id)
                        except Exception:
                            pass
                    
                    # 过滤失败的视频
                    for failed_video in all_failed:
                        try:
                            decoded = base64.b64decode(failed_video.get("video_id", "")).decode('utf-8')
                            parts = decoded.split('/')
                            if len(parts) >= 2:
                                vid_label = parts[0]
                                vid_generator = parts[1] if len(parts) >= 3 else None
                                if vid_label.lower() == label.lower():
                                    if generator is None or vid_generator == generator:
                                        failed.append(failed_video)
                        except Exception:
                            pass
        except Exception:
            pass
    
    return processed, failed


def save_progress(_label: str = None, _generator: str = None, processed: Set[str] = None, failed: list = None):
    """保存已处理的视频ID列表和失败列表
    
    统一保存到 all_progress.json，不生成单独的进度文件
    _label 和 _generator 参数保留用于兼容性，但不再使用
    """
    processed = processed or set()
    failed = failed or []
    
    # 统一保存到 all_progress.json
    all_progress_file = _get_progress_file("all", None)
    all_progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取现有的 all_progress.json（如果存在）
    all_processed = set()
    all_failed = []
    if all_progress_file.exists():
        try:
            with open(all_progress_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                all_processed = set(all_data.get("processed", []))
                all_failed = all_data.get("failed", [])
        except Exception:
            pass
    
    # 合并进度和失败列表
    all_processed.update(processed)
    all_failed_dict = {failed_video.get("video_id"): failed_video for failed_video in all_failed}
    for failed_video in failed:
        vid_id = failed_video.get("video_id")
        all_failed_dict[vid_id] = failed_video
    
    # 从失败列表中移除已经成功转换的视频
    for vid_id in all_processed:
        all_failed_dict.pop(vid_id, None)
    
    # 保存到 all_progress.json
    all_data = {
        "label": "all",
        "generator": None,
        "processed": list(all_processed),
        "failed": list(all_failed_dict.values()),
        "last_updated": time.time()
    }
    with open(all_progress_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def _get_progress_file(label: str, generator: str = None) -> Path:
    """获取进度文件路径"""
    filename = f"{label}"
    if generator:
        filename += f"_{generator}"
    filename += "_progress.json"
    return PROGRESS_DIR / filename


def batch_convert(
    label: str = "all",
    generator: str = None,
    limit: int = None,
    delay: float = None,
    verbose: bool = True,
    skip_converted: bool = True,
    resume: bool = True,
    retry_failed_only: bool = False
):
    """批量转换视频（支持断点续传和跳过已转换）
    
    支持路径格式的 label，如 "Fake/WildScrape" 会自动解析为 label="Fake", generator="WildScrape"
    如果使用传统格式（--label Fake --generator WildScrape），会自动转换为路径格式
    统一使用 all_progress.json 和 all_failed.json，不生成单独的进度文件
    """
    # 解析路径格式的 label（如 "Fake/WildScrape"）
    if "/" in label and label != "all":
        parts = label.split("/")
        if len(parts) == 2:
            label, generator = parts[0], parts[1]
            if verbose:
                print(f"📝 检测到路径格式，解析为: label={label}, generator={generator}")
    
    # 如果使用传统格式（generator 不为 None），转换为路径格式
    if generator and label != "all":
        if verbose:
            print(f"📝 检测到传统格式，转换为路径格式: {label}/{generator}")
        # 保持 label 和 generator 不变，但统一使用 all_progress.json
    
    print(f"📹 获取视频列表 (label={label}, generator={generator})...")

    # 加载进度（如果启用断点续传）
    if resume:
        processed_videos, previous_failed = load_progress(label, generator)
        if processed_videos:
            print(f"📋 加载进度：已处理 {len(processed_videos)} 个视频")
        if previous_failed:
            print(f"📋 之前失败：{len(previous_failed)} 个视频")
    else:
        processed_videos, previous_failed = set(), []
    
    # 如果只重试失败的视频，构建失败视频ID集合
    failed_video_ids = set()
    if retry_failed_only and previous_failed:
        failed_video_ids = {fv.get("video_id") for fv in previous_failed}
        print(f"🔄 只重试失败的视频：{len(failed_video_ids)} 个")
    
    # 流式处理：边获取边转换
    batch_size = 500
    success = 0
    failed_count = 0
    skipped = 0
    total_processed = 0
    newly_processed = set()
    failed_videos = previous_failed.copy() if previous_failed else []

    try:
        if limit is None:
            # 自动分批获取并转换所有视频
            if verbose:
                print("🔄 自动分批获取并转换所有视频（每次500个，流式处理）...")

            offset = 0
            while True:
                # 获取一批视频
                result = list_videos(label=label, generator=generator, limit=batch_size, offset=offset)
                batch = result.get("items", [])
                if not batch:
                    break
                
                offset += len(batch)
                batch_count = len(batch)
                total_processed += batch_count

                if verbose:
                    print(f"\n📦 批次：获取到 {batch_count} 个视频（累计: {total_processed}）")
                
                # 转换这批视频
                for i, video in enumerate(batch, 1):
                    video_id = video["video_id"]
                    file_name = video.get("file", "unknown")
                    rel_path = video.get("rel_path", "")
                    
                    # 如果只重试失败的视频，跳过不在失败列表中的视频
                    if retry_failed_only and failed_video_ids:
                        if video_id not in failed_video_ids:
                            skipped += 1
                            if verbose:
                                print(f"  [{i}/{batch_count}] {file_name}... ⏭️  不在失败列表中，跳过")
                            continue
                    
                    # 检查是否已转换
                    if skip_converted:
                        if video_id in processed_videos or is_evidence_ready(video_id, min_frames=2):
                            skipped += 1
                            if verbose:
                                print(f"  [{i}/{batch_count}] {file_name}... ⏭️  已转换，跳过")
                            continue

                    try:
                        if verbose:
                            print(f"  [{i}/{batch_count}] {file_name}...", end=" ", flush=True)
                        
                        # Convert video (build evidence cache)
                        build_llm_inputs(video_id, include_data_urls=False)
                        success += 1
                        newly_processed.add(video_id)
                        processed_videos.add(video_id)
                        
                        # 如果这个视频之前在失败列表中，现在转换成功了，从失败列表中移除
                        failed_videos = [fv for fv in failed_videos if fv.get("video_id") != video_id]
                        
                        # 每处理10个视频保存一次进度
                        if len(newly_processed) % 10 == 0 and resume:
                            save_progress(label, generator, processed_videos, failed_videos)
                        
                        if verbose:
                            print("✅")
                        
                        if delay and delay > 0:
                            time.sleep(delay)
                            
                    except Exception as e:
                        failed_count += 1
                        error_msg = str(e)
                        # 检查是否已经在失败列表中（避免重复）
                        existing_failed_ids = {fv.get("video_id") for fv in failed_videos}
                        if video_id not in existing_failed_ids:
                            failed_videos.append({
                                "video_id": video_id,
                                "file": file_name,
                                "rel_path": rel_path,
                                "error": error_msg
                            })
                        if verbose:
                            print(f"❌ 错误: {e}")
                
                # 如果返回的视频少于 batch_size，说明已经获取完所有视频
                if batch_count < batch_size:
                    break
                
                time.sleep(0.1)
        else:
            # 限制数量，只获取并转换指定数量
            api_limit = min(limit, batch_size)
            result = list_videos(label=label, generator=generator, limit=api_limit)
            all_videos = result.get("items", [])
            
            if not all_videos:
                print("❌ 没有找到视频")
                return
            
            total = len(all_videos)
            print(f"✅ 找到 {total} 个视频，开始转换...\n")
            
            for i, video in enumerate(all_videos, 1):
                video_id = video["video_id"]
                file_name = video.get("file", "unknown")
                rel_path = video.get("rel_path", "")
                
                # 如果只重试失败的视频，跳过不在失败列表中的视频
                if retry_failed_only and failed_video_ids:
                    if video_id not in failed_video_ids:
                        skipped += 1
                        if verbose:
                            print(f"[{i}/{total}] {file_name}... ⏭️  不在失败列表中，跳过")
                        continue
                
                # 检查是否已转换
                if skip_converted:
                    if video_id in processed_videos or is_evidence_ready(video_id, min_frames=2):
                        skipped += 1
                        if verbose:
                            print(f"[{i}/{total}] {file_name}... ⏭️  已转换，跳过")
                        continue
                
                try:
                    if verbose:
                        print(f"[{i}/{total}] 转换: {file_name}...", end=" ", flush=True)
                    
                    # Convert video (build evidence cache)
                    build_llm_inputs(video_id, include_data_urls=False)
                    success += 1
                    newly_processed.add(video_id)
                    processed_videos.add(video_id)
                    
                    # 如果这个视频之前在失败列表中，现在转换成功了，从失败列表中移除
                    failed_videos = [fv for fv in failed_videos if fv.get("video_id") != video_id]
                    
                    if verbose:
                        print("✅")
                    
                    if delay and delay > 0 and i < total:
                        time.sleep(delay)
                        
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    # 检查是否已经在失败列表中（避免重复）
                    existing_failed_ids = {fv.get("video_id") for fv in failed_videos}
                    if video_id not in existing_failed_ids:
                        failed_videos.append({
                            "video_id": video_id,
                            "file": file_name,
                            "rel_path": rel_path,
                            "error": error_msg
                        })
                    if verbose:
                        print(f"❌ 错误: {e}")
            
            total_processed = total
        
        # 保存最终进度
        if resume and (newly_processed or failed_videos):
            save_progress(label, generator, processed_videos, failed_videos)
        
        print(f"\n📊 转换完成:")
        print(f"   📹 处理总数: {total_processed}")
        print(f"   ✅ 成功: {success}")
        print(f"   ⏭️  跳过: {skipped}")
        print(f"   ❌ 失败: {failed_count}")
        print(f"   📁 缓存位置: cache/evidence/")
        if resume:
            print(f"   💾 进度已保存: cache/progress/")
        
        # 显示失败统计
        if failed_videos:
            print(f"\n❌ 失败视频统计（共 {len(failed_videos)} 个）:")
            # 按错误类型分组
            error_types = {}
            for fv in failed_videos:
                error = fv.get("error", "Unknown")
                error_type = error.split(":")[0] if ":" in error else error[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print(f"   错误类型分布:")
            for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     - {err_type}: {count} 个")
            
            # 保存失败列表到统一文件 all_failed.json
            failed_file = PROGRESS_DIR / "all_failed.json"
            
            # 读取现有的失败列表（如果存在）
            all_failed = []
            if failed_file.exists():
                try:
                    with open(failed_file, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                        all_failed = all_data.get("failed_videos", [])
                except Exception:
                    pass
            
            # 合并失败列表（去重，基于 video_id）
            all_failed_dict = {fv.get("video_id"): fv for fv in all_failed}
            for fv in failed_videos:
                vid_id = fv.get("video_id")
                all_failed_dict[vid_id] = fv
            
            # 从失败列表中移除已经成功转换的视频
            for vid_id in processed_videos:
                all_failed_dict.pop(vid_id, None)
            
            # 保存到 all_failed.json
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "label": "all",
                    "generator": None,
                    "total_failed": len(all_failed_dict),
                    "failed_videos": list(all_failed_dict.values())[:200]  # 只保存前200个，避免文件太大
                }, f, indent=2, ensure_ascii=False)
            print(f"\n   💾 详细失败列表已保存到: {failed_file}")
            print(f"      （只保存前200个，共 {len(all_failed_dict)} 个失败）")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  检测到中断信号 (CTRL+C)")
        
        if resume and (newly_processed or failed_videos):
            save_progress(label, generator, processed_videos, failed_videos)
            print(f"💾 已保存进度：{len(processed_videos)} 个视频已处理，{len(failed_videos)} 个失败")
        
        print(f"\n📊 当前状态:")
        print(f"   📹 处理总数: {total_processed}")
        print(f"   ✅ 成功: {success}")
        print(f"   ⏭️  跳过: {skipped}")
        print(f"   ❌ 失败: {failed_count}")
        print(f"\n💡 提示：重新运行相同命令即可从中断处继续")
        print(f"   命令: python convert.py --label {label}" + (f" --generator {generator}" if generator else ""))
        
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量转换视频为 LLM 友好格式")
    parser.add_argument("--label", default="all",
                       help="视频标签 (默认: all)")
    parser.add_argument("--generator", type=str, default=None,
                       help="假视频生成器名称 (仅用于 fake 标签)")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制转换数量 (默认: 全部)")
    default_delay = get_batch_delay_sec()
    parser.add_argument("--delay", type=float, default=default_delay,
                       help=f"请求间隔秒数 (默认: {default_delay})")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（不显示详细进度）")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已转换的视频（默认：跳过）")
    parser.add_argument("--no-resume", action="store_true",
                       help="不启用断点续传（默认：启用）")
    parser.add_argument("--retry-failed-only", action="store_true",
                       help="只重试失败的视频（默认：处理所有视频）")
    
    args = parser.parse_args()
    
    batch_convert(
        label=args.label,
        generator=args.generator,
        limit=args.limit,
        delay=args.delay,
        verbose=not args.quiet,
        skip_converted=not args.no_skip,
        resume=not args.no_resume,
        retry_failed_only=args.retry_failed_only
    )
