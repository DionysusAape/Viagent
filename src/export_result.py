#!/usr/bin/env python3
"""
导出分析结果到 JSON 文件

用法：
  python export_result.py [run_id] [--output output.json]
  如果不提供 run_id，则导出最近一次的分析结果
"""
import argparse
import json
import sys
from pathlib import Path
from database.db_helper import ViagentDB


def export_analysis_to_json(run_id: str = None, output_path: str = None) -> dict:
    """Export analysis result to JSON format"""
    db = ViagentDB()
    
    # 如果没有提供 run_id，获取最新的
    if not run_id:
        import sqlite3
        conn = db._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT run_id FROM analysis_run ORDER BY created_at DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("❌ 错误: 数据库中没有分析记录", file=sys.stderr)
            sys.exit(1)
        
        run_id = row[0]
        print(f"📋 使用最新的 run_id: {run_id}")
    
    # 获取完整分析结果
    complete_analysis = db.get_complete_analysis(run_id)
    if not complete_analysis:
        print(f"❌ 错误: 找不到 run_id={run_id} 的分析结果", file=sys.stderr)
        sys.exit(1)
    
    # 格式化输出
    run = complete_analysis['run']
    agent_results = complete_analysis['agent_results']
    verdict = complete_analysis['verdict']
    
    result = {
        "run_id": run_id,
        "created_at": run['created_at'],
        "case": {
            "case_id": run['case_id'],
            "video_path": run.get('video_path'),
            "label": run.get('label')
        },
        "agents": {},
        "verdict": None
    }
    
    # 格式化 agent results
    for agent_result in agent_results:
        agent_name = agent_result['agent']
        result["agents"][agent_name] = {
            "status": agent_result['status'],
            "score_fake": agent_result['score_fake'],
            "confidence": agent_result['confidence'],
            "error": agent_result['error'],
            "evidence": agent_result['evidence']
        }
    
    # 格式化 verdict
    if verdict:
        result["verdict"] = {
            "label": verdict['label'],
            "score_fake": verdict['score_fake'],
            "confidence": verdict['confidence'],
            "rationale": verdict['rationale'],
            "evidence": verdict['evidence']
        }
    
    # 输出到文件或 stdout
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    
    if output_path:
        output_file = Path(output_path)
        output_file.write_text(json_str, encoding='utf-8')
        print(f"✅ 结果已导出到: {output_file.absolute()}")
    else:
        print(json_str)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="导出分析结果到 JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "run_id",
        type=str,
        nargs='?',
        default=None,
        help="分析运行 ID（如果不提供，则导出最近一次）"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（如果不提供，则输出到 stdout）"
    )
    
    args = parser.parse_args()
    
    try:
        export_analysis_to_json(args.run_id, args.output)
    except Exception as e:
        print(f"❌ 导出失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
