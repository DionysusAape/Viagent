#!/usr/bin/env python3
"""
删除指定实验的所有数据

用法（在 src 目录下运行，直接删除指定实验的所有记录）:
    python delete_experiment.py --experiment gt-3
"""
import argparse
import json
import sqlite3
from database.db_setup import DB_PATH


def delete_experiment_data(experiment_name: str) -> None:
    """删除指定实验的所有数据"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 首先查询所有匹配的 run_id
    cursor.execute('''
        SELECT run_id, case_id, video_path, config
        FROM analysis_run
    ''')
    
    matching_runs = []
    for row in cursor.fetchall():
        try:
            config = json.loads(row['config']) if row['config'] else {}
            db_experiment_name = config.get('experiment_name')
            if db_experiment_name == experiment_name:
                matching_runs.append({
                    'run_id': row['run_id'],
                    'case_id': row['case_id'],
                    'video_path': row['video_path']
                })
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    if not matching_runs:
        print(f"❌ 未找到实验 '{experiment_name}' 的数据")
        conn.close()
        return
    
    print(f"📊 找到 {len(matching_runs)} 条实验 '{experiment_name}' 的记录")
    # 确认删除
    print(f"\n⚠️  警告：即将删除 {len(matching_runs)} 条实验 '{experiment_name}' 的记录")
    print("   这将删除所有相关的 analysis_run, agent_result, evidence, verdict, verdict_evidence 记录")
    confirm = input("   确认删除？(yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ 已取消删除操作")
        conn.close()
        return
    
    # 执行删除（由于 CASCADE，会自动删除关联的记录）
    deleted_count = 0
    try:
        for run in matching_runs:
            cursor.execute('DELETE FROM analysis_run WHERE run_id = ?', (run['run_id'],))
            deleted_count += 1
        
        conn.commit()
        print(f"✅ 成功删除 {deleted_count} 条记录")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"❌ 删除失败: {e}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="删除指定实验的所有数据",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="实验名称（如 gt-3）"
    )
    args = parser.parse_args()
    delete_experiment_data(args.experiment)


if __name__ == "__main__":
    main()
