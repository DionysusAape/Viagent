#!/usr/bin/env python3
"""
批量转换视频为 LLM 友好格式（支持断点续传）
用法: python server/convert.py [选项]
或: python -m server.convert [选项]
"""
import argparse
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from src.config import VIDEO_API_BASE_URL, EVID_DIR as CACHE_DIR, PROGRESS_DIR, BATCH_DELAY_SEC

API_BASE = VIDEO_API_BASE_URL
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

def get_videos(label: str = "all", generator: Optional[str] = None, limit: int = 500, offset: int = 0) -> List[Dict]:
    """获取视频列表（API限制：最多500，支持分页）"""
    url = f"{API_BASE}/videos"
    params = {"label": label, "limit": limit, "offset": offset}
    if generator:
        params["generator"] = generator
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])

def convert_video(video_id: str, include_data_urls: bool = False) -> Dict:
    """转换单个视频"""
    url = f"{API_BASE}/videos/{video_id}/evidence"
    params = {}
    if include_data_urls:
        params["include_data_urls"] = "true"
    
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def is_video_converted(video_id: str, min_frames: int = 1) -> bool:
    """检查视频是否已经转换（通过检查缓存目录）"""
    cache_path = CACHE_DIR / video_id
    if not cache_path.exists():
        return False

    meta_path = cache_path / "meta.json"
    frames_dir = cache_path / "frames"
    
    # 检查 meta.json 和至少 min_frames 个帧文件
    if not meta_path.exists():
        return False
    
    if not frames_dir.exists():
        return False
    
    frame_count = len(list(frames_dir.glob("*.jpg")))
    return frame_count >= min_frames

def load_progress(label: str, generator: Optional[str] = None) -> Set[str]:
    """加载已处理的视频ID列表"""
    progress_file = _get_progress_file(label, generator)
    if not progress_file.exists():
        return set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("processed", []))
    except Exception:
        return set()

def save_progress(label: str, generator: Optional[str] = None, processed: Set[str] = None):
    """保存已处理的视频ID列表"""
    progress_file = _get_progress_file(label, generator)
    data = {
        "label": label,
        "generator": generator,
        "processed": list(processed or set()),
        "last_updated": time.time()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _get_progress_file(label: str, generator: Optional[str] = None) -> Path:
    """获取进度文件路径"""
    filename = f"{label}"
    if generator:
        filename += f"_{generator}"
    filename += "_progress.json"
    return PROGRESS_DIR / filename

def batch_convert(
    label: str = "all",
    generator: Optional[str] = None,
    limit: int = None,
    delay: float = BATCH_DELAY_SEC,
    verbose: bool = True,
    skip_converted: bool = True,
    resume: bool = True
):
    """批量转换视频（支持断点续传和跳过已转换）"""
    print(f"📹 获取视频列表 (label={label}, generator={generator})...")

    # 加载进度（如果启用断点续传）
    processed_videos = set()
    if resume:
        processed_videos = load_progress(label, generator)
        if processed_videos:
            print(f"📋 加载进度：已处理 {len(processed_videos)} 个视频")
    
    # 流式处理：边获取边转换，避免一次性加载所有视频到内存
    batch_size = 500  # API 限制
    success = 0
    failed = 0
    skipped = 0
    total_processed = 0
    newly_processed = set()

    try:
        if limit is None:
            # 自动分批获取并转换所有视频（流式处理）
            if verbose:
                print("🔄 自动分批获取并转换所有视频（每次500个，流式处理）...")

            offset = 0
            while True:
                # 获取一批视频（使用 offset 分页）
                batch = get_videos(label=label, generator=generator, limit=batch_size, offset=offset)
                if not batch:
                    break
                
                offset += len(batch)  # 更新 offset 为下次获取做准备
                batch_count = len(batch)
                total_processed += batch_count

                if verbose:
                    print(f"\n📦 批次：获取到 {batch_count} 个视频（累计: {total_processed}）")
                
                # 立即转换这批视频
                for i, video in enumerate(batch, 1):
                    video_id = video["video_id"]
                    file_name = video.get("file", "unknown")
                    
                    # 检查是否已转换（如果启用跳过）
                    if skip_converted:
                        if video_id in processed_videos or is_video_converted(video_id, min_frames=1):
                            skipped += 1
                            if verbose:
                                print(f"  [{i}/{batch_count}] {file_name}... ⏭️  已转换，跳过")
                            continue

                    try:
                        if verbose:
                            print(f"  [{i}/{batch_count}] {file_name}...", end=" ", flush=True)
                        
                        convert_video(video_id)
                        success += 1
                        newly_processed.add(video_id)
                        processed_videos.add(video_id)
                        
                        # 每处理10个视频保存一次进度
                        if len(newly_processed) % 10 == 0 and resume:
                            save_progress(label, generator, processed_videos)
                        
                        if verbose:
                            print("✅")
                        
                        if delay > 0:
                            time.sleep(delay)
                            
                    except Exception as e:
                        failed += 1
                        if verbose:
                            print(f"❌ 错误: {e}")
                
                # 如果返回的视频少于 batch_size，说明已经获取完所有视频
                if batch_count < batch_size:
                    break
                
                # 批次间短暂延迟
                time.sleep(0.1)
        else:
            # 限制数量，只获取并转换指定数量
            api_limit = min(limit, batch_size)
            all_videos = get_videos(label=label, generator=generator, limit=api_limit)
            
            if not all_videos:
                print("❌ 没有找到视频")
                return
            
            total = len(all_videos)
            print(f"✅ 找到 {total} 个视频，开始转换...\n")
            
            for i, video in enumerate(all_videos, 1):
                video_id = video["video_id"]
                file_name = video.get("file", "unknown")
                
                # 检查是否已转换（如果启用跳过）
                if skip_converted:
                    if video_id in processed_videos or is_video_converted(video_id, min_frames=1):
                        skipped += 1
                        if verbose:
                            print(f"[{i}/{total}] {file_name}... ⏭️  已转换，跳过")
                        continue
                
                try:
                    if verbose:
                        print(f"[{i}/{total}] 转换: {file_name}...", end=" ", flush=True)
                    
                    convert_video(video_id)
                    success += 1
                    newly_processed.add(video_id)
                    processed_videos.add(video_id)
                    
                    if verbose:
                        print("✅")
                    
                    if delay > 0 and i < total:
                        time.sleep(delay)
                        
                except Exception as e:
                    failed += 1
                    if verbose:
                        print(f"❌ 错误: {e}")
            
            total_processed = total
        
        # 保存最终进度
        if resume and newly_processed:
            save_progress(label, generator, processed_videos)
        
        print(f"\n📊 转换完成:")
        print(f"   📹 处理总数: {total_processed}")
        print(f"   ✅ 成功: {success}")
        print(f"   ⏭️  跳过: {skipped}")
        print(f"   ❌ 失败: {failed}")
        print(f"   📁 缓存位置: cache/evidence/")
        if resume:
            print(f"   💾 进度已保存: cache/progress/")
    
    except KeyboardInterrupt:
        # 处理 CTRL+C 中断
        print(f"\n\n⚠️  检测到中断信号 (CTRL+C)")
        
        # 保存当前进度
        if resume and newly_processed:
            save_progress(label, generator, processed_videos)
            print(f"💾 已保存进度：{len(processed_videos)} 个视频已处理")
        
        print(f"\n📊 当前状态:")
        print(f"   📹 处理总数: {total_processed}")
        print(f"   ✅ 成功: {success}")
        print(f"   ⏭️  跳过: {skipped}")
        print(f"   ❌ 失败: {failed}")
        print(f"\n💡 提示：重新运行相同命令即可从中断处继续")
        print(f"   命令: python server/convert.py --label {label}" + (f" --generator {generator}" if generator else ""))
        
        # 正常退出，不显示错误
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量转换视频为 LLM 友好格式")
    parser.add_argument("--label", choices=["all", "real", "fake"], default="all",
                       help="视频标签 (默认: all)")
    parser.add_argument("--generator", type=str, default=None,
                       help="假视频生成器名称 (仅用于 fake 标签)")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制转换数量 (默认: 全部)")
    parser.add_argument("--delay", type=float, default=BATCH_DELAY_SEC,
                       help=f"请求间隔秒数 (默认: {BATCH_DELAY_SEC})")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（不显示详细进度）")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已转换的视频（默认：跳过）")
    parser.add_argument("--no-resume", action="store_true",
                       help="不启用断点续传（默认：启用）")
    
    args = parser.parse_args()
    
    try:
        # 检查 API 是否可用
        resp = requests.get(f"{API_BASE}/ping", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ 无法连接到 API 服务器 ({API_BASE})")
        print(f"   请确保服务器正在运行: uvicorn server.fast_api:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    batch_convert(
        label=args.label,
        generator=args.generator,
        limit=args.limit,
        delay=args.delay,
        verbose=not args.quiet,
        skip_converted=not args.no_skip,
        resume=not args.no_resume
    )
