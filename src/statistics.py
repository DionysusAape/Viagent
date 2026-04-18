#!/usr/bin/env python3
"""
统计分析结果的准确率、精确率、召回率等指标

用法（在 src 目录下）：
    python statistics.py                    # 统计所有结果
    python statistics.py --config config.yaml    # 统计指定配置文件的结果
    python statistics.py --list-experiments      # 列出所有试验
"""
import sqlite3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from database.db_setup import DB_PATH, init_database
from util.config import load_config
from pipeline.evidence import parse_video_id


def extract_config_signature(config_dict: Dict) -> Dict[str, Any]:
    signature = {}

    # LLM 配置
    if 'llm' in config_dict:
        llm = config_dict['llm']
        signature['llm_provider'] = llm.get('provider')
        signature['llm_model'] = llm.get('model')

    # Decision Policy
    if 'decision_policy' in config_dict:
        dp = config_dict['decision_policy']
        signature['threshold_fake'] = dp.get('threshold_fake')
        signature['threshold_real'] = dp.get('threshold_real')

    # Judge Dynamic
    if 'judge_dynamic' in config_dict:
        jd = config_dict['judge_dynamic']
        signature['judge_dynamic_enabled'] = jd.get('enabled')
        if jd.get('enabled'):
            signature['vote_s_high'] = jd.get('vote_s_high')
            signature['vote_s_low'] = jd.get('vote_s_low')
            signature['strong_s'] = jd.get('strong_s')

    # Workflow
    if 'workflow_analysts' in config_dict:
        signature['workflow_analysts'] = sorted(config_dict['workflow_analysts'])

    if 'enable_skills' in config_dict:
        signature['enable_skills'] = bool(config_dict['enable_skills'])

    return signature


def config_matches(config_json_str: str, target_signature: Dict) -> bool:
    """
    检查数据库中的配置是否匹配目标配置签名

    Args:
        config_json_str: 数据库中的配置 JSON 字符串
        target_signature: 目标配置签名

    Returns:
        True if matches, False otherwise
    """
    if not config_json_str:
        return False

    try:
        db_config = json.loads(config_json_str)
        db_signature = extract_config_signature(db_config)

        # 检查所有目标签名中的键值是否匹配
        for key, value in target_signature.items():
            if key not in db_signature:
                return False
            if db_signature[key] != value:
                return False

        return True
    except (json.JSONDecodeError, TypeError):
        return False


def get_all_results(
    config_path: Optional[str] = None
) -> Tuple[
    List[Tuple[str, str, str, float, float, Optional[float]]],
    Optional[str],
]:
    """
    从数据库获取分析结果，可选按配置文件过滤

    Args:
        config_path: 可选的配置文件路径，如果提供则只统计该配置的结果

    Returns:
        Tuple of (results, experiment_name)
        results: List of (case_id, true_label, predicted_label, score_fake, confidence, duration_sec)
        experiment_name: 试验名称（从配置文件名或 experiment_name 字段提取）
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 如果指定了配置文件，加载并获取 experiment_name
    target_experiment_name = None
    experiment_name = None

    if config_path:
        try:
            config = load_config(config_path)

            # 从配置文件名提取试验名称
            config_file = Path(config_path)
            experiment_name = config_file.stem  # 例如 "config1.yaml" -> "config1"

            # 如果配置中有 experiment_name 字段，使用它
            if 'experiment_name' in config:
                experiment_name = config['experiment_name']
                target_experiment_name = experiment_name
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"⚠️  警告: 无法加载配置文件 {config_path}: {exc}")
            print("   将统计所有结果")
            config_path = None

    # 查询所有有 verdict 的分析结果，并获取真实标签和配置
    cursor.execute('''
        SELECT
            ar.case_id,
            ar.label as true_label,
            ar.config as config_json,
            v.label as predicted_label,
            v.score_fake,
            v.confidence,
            ar.duration_sec
        FROM analysis_run ar
        INNER JOIN verdict v ON ar.run_id = v.run_id
        WHERE ar.label IS NOT NULL
          AND ar.label IN ('real', 'fake')
          AND v.label IN ('real', 'fake', 'uncertain')
        ORDER BY ar.created_at DESC
    ''')

    results = []
    for row in cursor.fetchall():
        # 如果指定了配置文件，检查 experiment_name 是否匹配
        if target_experiment_name:
            try:
                db_config = json.loads(row['config_json']) if row['config_json'] else {}
                db_experiment_name = db_config.get('experiment_name')
                if db_experiment_name != target_experiment_name:
                    continue
            except (json.JSONDecodeError, TypeError, AttributeError):
                # 如果无法解析配置或没有 experiment_name，跳过
                continue

        results.append(
            (
                row["case_id"],
                row["true_label"].lower(),
                row["predicted_label"].lower(),
                row["score_fake"],
                row["confidence"],
                row["duration_sec"],
            )
        )

    conn.close()
    return results, experiment_name


def get_misclassified_and_uncertain(
    results: List[Tuple[str, str, str, float, float, Optional[float]]]
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    从结果中提取预测错误的视频列表。
    
    兼容旧接口形式，第二个返回值（uncertain）现在始终为空列表，
    因为系统已经不再使用 uncertain 标签。
    """
    misclassified: List[Tuple[str, str, str]] = []
    for case_id, true_label, predicted_label, _score_fake, _confidence, _duration_sec in results:
        if predicted_label != true_label:
            misclassified.append((case_id, true_label, predicted_label))
    return misclassified, []


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile from sorted values with linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    ratio = idx - lower
    return values[lower] * (1.0 - ratio) + values[upper] * ratio


def calculate_metrics(results: List[Tuple[str, str, str, float, float, Optional[float]]]) -> Dict:
    """
    计算各种评估指标

    Args:
        results: List of (case_id, true_label, predicted_label, score_fake, confidence, duration_sec)

    Returns:
        Dict with various metrics
    """
    # 统计混淆矩阵
    confusion_matrix = defaultdict(int)
    total = len(results)

    # 按真实标签和预测标签分类
    by_true_label = defaultdict(list)
    by_predicted_label = defaultdict(list)

    for _, true_label, predicted_label, score_fake, confidence, _duration_sec in results:
        key = (true_label, predicted_label)
        confusion_matrix[key] += 1
        by_true_label[true_label].append((predicted_label, score_fake, confidence))
        by_predicted_label[predicted_label].append((true_label, score_fake, confidence))

    # 计算基本指标（二分类：real vs fake）
    # TP: 真实fake，预测fake
    # TN: 真实real，预测real
    # FP: 真实real，预测fake
    # FN: 真实fake，预测real

    tp = confusion_matrix[('fake', 'fake')]
    tn = confusion_matrix[('real', 'real')]
    fp = confusion_matrix[('real', 'fake')]
    fn = confusion_matrix[('fake', 'real')]

    # 准确率
    total_clear = tp + tn + fp + fn
    accuracy = (tp + tn) / total_clear if total_clear > 0 else 0.0

    # 精确率（Precision）：预测为fake中，真正是fake的比例
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 召回率（Recall）：真实fake中，被正确预测为fake的比例
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1分数
    f1_fake = (
        2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)
        if (precision_fake + recall_fake) > 0 else 0.0
    )

    # 对于real类别
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_real = (
        2 * (precision_real * recall_real) / (precision_real + recall_real)
        if (precision_real + recall_real) > 0 else 0.0
    )

    # 计算平均置信度
    avg_confidence_correct = 0.0
    avg_confidence_wrong = 0.0
    correct_count = 0
    wrong_count = 0

    for _, true_label, predicted_label, score_fake, confidence, _duration_sec in results:
        if predicted_label == 'uncertain':
            continue
        is_correct = true_label == predicted_label
        if is_correct:
            avg_confidence_correct += confidence
            correct_count += 1
        else:
            avg_confidence_wrong += confidence
            wrong_count += 1

    avg_confidence_correct = avg_confidence_correct / correct_count if correct_count > 0 else 0.0
    avg_confidence_wrong = avg_confidence_wrong / wrong_count if wrong_count > 0 else 0.0

    durations = [row[5] for row in results if row[5] is not None]
    durations_sorted = sorted(float(x) for x in durations)
    timing_metrics = {
        'count': len(durations_sorted),
        'total_sec': sum(durations_sorted) if durations_sorted else 0.0,
        'avg_sec': (sum(durations_sorted) / len(durations_sorted)) if durations_sorted else 0.0,
        'median_sec': _percentile(durations_sorted, 0.5) if durations_sorted else 0.0,
        'p90_sec': _percentile(durations_sorted, 0.9) if durations_sorted else 0.0,
        'min_sec': durations_sorted[0] if durations_sorted else 0.0,
        'max_sec': durations_sorted[-1] if durations_sorted else 0.0,
    }

    slowest_cases = sorted(
        (
            (case_id, float(duration_sec))
            for case_id, _tl, _pl, _sf, _cf, duration_sec in results
            if duration_sec is not None
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        'total': total,
        'total_clear': total_clear,
        'confusion_matrix': {
            'TP (fake->fake)': tp,
            'TN (real->real)': tn,
            'FP (real->fake)': fp,
            'FN (fake->real)': fn,
        },
        'accuracy': accuracy,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'precision_real': precision_real,
        'recall_real': recall_real,
        'f1_real': f1_real,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_wrong': avg_confidence_wrong,
        'timing': timing_metrics,
        'slowest_cases': slowest_cases,
    }


def print_statistics(experiment_name: Optional[str] = None):
    """打印统计结果"""
    print("=" * 80)
    print("📊 视频分析结果统计")
    if experiment_name:
        print(f"   试验: {experiment_name}")
    print("=" * 80)
    print()


def print_error_cases(
    results: List[Tuple[str, str, str, float, float, Optional[float]]],
    metrics: Dict,
) -> None:
    """
    打印预测错误的视频名字。

    视频名字通过解析 case_id -> 相对路径获得；
    如果解析失败，则回退为原始 case_id。
    """
    misclassified, _uncertain = get_misclassified_and_uncertain(results)

    print("=" * 80)
    print("📂 预测错误的视频列表")
    print("=" * 80)

    if not misclassified:
        print("   ✅ 没有预测错误的视频（不含 uncertain）")
    else:
        for case_id, true_label, predicted_label in misclassified:
            try:
                rel_path = parse_video_id(case_id)
                name = str(rel_path)
            except (TypeError, ValueError, KeyError):
                name = case_id
            print(f"   {name}  (真值: {true_label}, 预测: {predicted_label})")
        print(f"\n   总计预测错误视频数: {len(misclassified)}")

    print("📈 总体统计:")
    print(f"   总样本数: {metrics['total']}")
    print(f"   明确预测数: {metrics['total_clear']}")
    print()
    
    print("📋 混淆矩阵:")
    cm = metrics['confusion_matrix']
    for key, value in cm.items():
        print(f"   {key}: {value}")
    print()

    print("✅ 准确率 (Accuracy):")
    print(f"   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print()

    print("🎯 Fake 类别指标:")
    precision_fake_pct = metrics['precision_fake'] * 100
    print(f"   精确率 (Precision): {metrics['precision_fake']:.4f} ({precision_fake_pct:.2f}%)")
    recall_fake_pct = metrics['recall_fake'] * 100
    print(f"   召回率 (Recall):   {metrics['recall_fake']:.4f} ({recall_fake_pct:.2f}%)")
    print(f"   F1 分数:           {metrics['f1_fake']:.4f}")
    print()

    print("🎯 Real 类别指标:")
    precision_real_pct = metrics['precision_real'] * 100
    print(f"   精确率 (Precision): {metrics['precision_real']:.4f} ({precision_real_pct:.2f}%)")
    recall_real_pct = metrics['recall_real'] * 100
    print(f"   召回率 (Recall):   {metrics['recall_real']:.4f} ({recall_real_pct:.2f}%)")
    print(f"   F1 分数:           {metrics['f1_real']:.4f}")
    print()

    print("💡 置信度分析:")
    print(f"   正确预测的平均置信度: {metrics['avg_confidence_correct']:.4f}")
    print(f"   错误预测的平均置信度: {metrics['avg_confidence_wrong']:.4f}")
    print()

    timing = metrics.get('timing', {})
    if timing.get('count', 0) > 0:
        print("⏱️  耗时统计（analysis_run.duration_sec）:")
        print(f"   样本数: {timing['count']}")
        print(f"   总耗时: {timing['total_sec']:.2f}s")
        print(f"   平均耗时: {timing['avg_sec']:.2f}s")
        print(f"   中位数: {timing['median_sec']:.2f}s")
        print(f"   P90: {timing['p90_sec']:.2f}s")
        print(f"   最快/最慢: {timing['min_sec']:.2f}s / {timing['max_sec']:.2f}s")
        print()

        slowest_cases = metrics.get('slowest_cases', [])
        if slowest_cases:
            print("🐢 最慢样本 Top 5:")
            for case_id, duration_sec in slowest_cases:
                try:
                    rel_path = parse_video_id(case_id)
                    name = str(rel_path)
                except (TypeError, ValueError, KeyError):
                    name = case_id
                print(f"   {name}  ({duration_sec:.2f}s)")
            print()
    else:
        print("⏱️  耗时统计: 当前结果缺少 duration_sec，无法计算。")
        print()

    print("=" * 80)


def list_experiments():
    """列出数据库中的所有试验（基于不同的配置）"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT ar.config as config_json
        FROM analysis_run ar
        WHERE ar.config IS NOT NULL
    ''')

    experiments = []
    for row in cursor.fetchall():
        if not row['config_json']:
            continue

        try:
            config = json.loads(row['config_json'])
            signature = extract_config_signature(config)

            # 生成试验标识
            provider = signature.get('llm_provider')
            model = signature.get('llm_model')
            exp_id = f"provider={provider}, model={model}"
            if signature.get('threshold_fake'):
                exp_id += f", threshold_fake={signature.get('threshold_fake')}"

            if exp_id not in experiments:
                experiments.append(exp_id)
        except (json.JSONDecodeError, TypeError):
            continue

    conn.close()

    print("=" * 80)
    print("📋 数据库中的试验列表")
    print("=" * 80)
    if experiments:
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp}")
    else:
        print("  未找到任何试验配置")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="统计分析结果的准确率、精确率、召回率等指标",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "配置文件路径（如 config.yaml 或 config1.yaml），"
            "如果指定则只统计该配置的结果"
        )
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="列出数据库中的所有试验配置"
    )

    args = parser.parse_args()

    if args.list_experiments:
        list_experiments()
        return

    config_path = args.config
    if config_path:
        print(f"🔍 正在从数据库读取分析结果（配置文件: {config_path}）...")
    else:
        print("🔍 正在从数据库读取所有分析结果...")

    results, experiment_name = get_all_results(config_path)

    if not results:
        if config_path:
            print(f"❌ 未找到使用配置文件 {config_path} 的分析结果。")
            print(
                "   提示: 请确保已使用该配置文件运行过分析"
                "（python main.py --label all --config <config_file>）"
            )
        else:
            print("❌ 未找到任何分析结果。请先运行 convert.py 和 main.py 进行分析。")
        return

    print(f"✅ 找到 {len(results)} 条分析结果")
    if experiment_name:
        print(f"   试验名称: {experiment_name}")
    print()

    metrics = calculate_metrics(results)
    print_statistics(experiment_name)

    # 额外输出：预测错误的视频名字列表
    print_error_cases(results, metrics)


if __name__ == "__main__":
    main()
