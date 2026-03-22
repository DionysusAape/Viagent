# 多智能体系统输出格式说明

## 输出结构概览

系统运行后会返回一个 JSON 格式的结果字典，包含以下部分：

```json
{
  "run_id": "abc12345",           // 运行ID（8位UUID）
  "case": { ... },                 // 视频案例信息
  "artifacts": { ... },            // 提取的证据（元数据、帧）
  "results": { ... },              // 各个智能体的分析结果
  "verdict": { ... }               // 最终判定结果
}
```

---

## 1. Case（视频案例信息）

```json
{
  "case_id": "UmVhbC92aWRlb18xLm1wNA",  // Base64编码的视频相对路径
  "video_path": "/path/to/data/Real/video_1.mp4",  // 视频文件绝对路径
  "label": "real",                   // 可选：已知标签（"real" 或 "fake"）
  "source": "file"                   // 来源（"file" 表示本地文件）
}
```

---

## 2. Artifacts（提取的证据）

```json
{
  "meta": {
    "width": 1920,                   // 视频宽度（像素）
    "height": 1080,                  // 视频高度（像素）
    "duration_sec": 5.5,            // 视频时长（秒）
    "codec": "h264",                // 视频编码格式
    "frame_rate": 30.0,             // 帧率
    "bitrate": 5000000,             // 比特率
    "video_id": "...",              // 视频ID
    "rel_path": "Real/video_1.mp4", // 相对路径
    "file_name": "video_1.mp4",      // 文件名
    "label": "real",                 // 标签
    "sha256": "abc123...",           // 文件SHA256哈希
    "frame_timestamps_sec": [0.0, 0.23, 0.46, ...]  // 提取帧的时间戳
  },
  "frame_count": 24                  // 提取的帧数量
}
```

---

## 3. Results（各智能体分析结果）

每个智能体返回一个结果对象：

```json
{
  "metadata": {
    "agent": "metadata",             // 智能体名称
    "status": "ok",                  // 状态：ok（成功）、skipped（跳过）、error（错误）
    "score_fake": 0.4,              // 假视频分数（0-1，越高越假）
    "confidence": 0.3,              // 置信度（0-1）
    "evidence_count": 2,            // 证据项数量
    "error": null                    // 错误信息（如果有）
  },
  "spatial": {
    "agent": "spatial",
    "status": "skipped",            // 当前为占位实现，状态为 skipped
    "score_fake": null,
    "confidence": null,
    "evidence_count": 0,
    "error": null
  },
  "temporal": {
    "agent": "temporal",
    "status": "skipped",            // 当前为占位实现
    "score_fake": null,
    "confidence": null,
    "evidence_count": 0,
    "error": null
  },
  "avsync": {
    "agent": "avsync",
    "status": "skipped",            // 当前为占位实现
    "score_fake": null,
    "confidence": null,
    "evidence_count": 0,
    "error": null
  },
  "watermark": {
    "agent": "watermark",
    "status": "skipped",            // 当前为占位实现
    "score_fake": null,
    "confidence": null,
    "evidence_count": 0,
    "error": null
  }
}
```

### 智能体说明

| 智能体 | 功能 | 当前状态 |
|--------|------|----------|
| **metadata** | 分析视频元数据（分辨率、时长、编码等） | ✅ 已实现 |
| **spatial** | 分析空间特征（帧内异常） | ⏸️ 占位实现 |
| **temporal** | 分析时间特征（帧间一致性） | ⏸️ 占位实现 |
| **avsync** | 分析音视频同步 | ⏸️ 占位实现 |
| **watermark** | 检测水印和生成器标记 | ⏸️ 占位实现 |

### score_fake 含义

- **0.0 - 0.3**：很可能是真实视频
- **0.3 - 0.7**：不确定
- **0.7 - 1.0**：很可能是假视频

---

## 4. Verdict（最终判定）

```json
{
  "label": "real",                   // 判定结果："real"（真实）、"fake"（假）、"uncertain"（不确定）
  "score_fake": 0.4,                // 平均假视频分数（所有智能体的平均值）
  "confidence": 0.2,                // 置信度（基于参与分析的智能体数量）
  "rationale": "Analyzed with 1 agent(s). Average fake score: 0.400. Score range: 0.400 - 0.400.",  // 判定理由
  "evidence_count": 2                // 总证据项数量
}
```

### 判定规则

系统使用规则融合（Rule-based Fusion）来生成最终判定：

1. **收集所有智能体的 `score_fake`**（忽略 `None` 和错误状态）
2. **计算平均分数**
3. **应用阈值**：
   - 如果 `avg_score >= threshold_fake`（默认 0.7）→ **fake**
   - 如果 `avg_score <= (1 - threshold_real)`（默认 0.3）→ **real**
   - 否则 → **uncertain**

### 置信度计算

- 基于参与分析的智能体数量
- 最多 5 个智能体时达到最大置信度（1.0）
- 公式：`confidence = min(1.0, len(scores) / 5.0)`

---

## 完整输出示例

### 命令行输出（非 JSON 模式）

```bash
$ python -m src.main Real/video_1.mp4

🔍 正在分析视频: /path/to/data/Real/video_1.mp4

✅ 分析完成!
   视频ID: UmVhbC92aWRlb18xLm1wNA
   判定: real
   置信度: 0.20
   理由: Analyzed with 1 agent(s). Average fake score: 0.400. Score range: 0.400 - 0.400.

📊 智能体结果:
   - metadata: ok, score_fake=0.40
   - spatial: skipped, score_fake=N/A
   - temporal: skipped, score_fake=N/A
   - avsync: skipped, score_fake=N/A
   - watermark: skipped, score_fake=N/A

📁 缓存位置: cache/evidence/UmVhbC92aWRlb18xLm1wNA/
```

### JSON 输出（使用 `--json` 参数）

```bash
$ python -m src.main Real/video_1.mp4 --json
```

```json
{
  "run_id": "abc12345",
  "case": {
    "case_id": "UmVhbC92aWRlb18xLm1wNA",
    "video_path": "/path/to/data/Real/video_1.mp4",
    "label": null,
    "source": "file"
  },
  "artifacts": {
    "meta": {
      "width": 1920,
      "height": 1080,
      "duration_sec": 5.5,
      "codec": "h264",
      "frame_rate": 30.0,
      "bitrate": 5000000,
      "video_id": "UmVhbC92aWRlb18xLm1wNA",
      "rel_path": "Real/video_1.mp4",
      "file_name": "video_1.mp4",
      "label": "real",
      "sha256": "abc123...",
      "frame_timestamps_sec": [0.0, 0.23, 0.46, ...]
    },
    "frame_count": 24
  },
  "results": {
    "metadata": {
      "agent": "metadata",
      "status": "ok",
      "score_fake": 0.4,
      "confidence": 0.3,
      "evidence_count": 2,
      "error": null
    },
    "spatial": {
      "agent": "spatial",
      "status": "skipped",
      "score_fake": null,
      "confidence": null,
      "evidence_count": 0,
      "error": null
    },
    "temporal": {
      "agent": "temporal",
      "status": "skipped",
      "score_fake": null,
      "confidence": null,
      "evidence_count": 0,
      "error": null
    },
    "avsync": {
      "agent": "avsync",
      "status": "skipped",
      "score_fake": null,
      "confidence": null,
      "evidence_count": 0,
      "error": null
    },
    "watermark": {
      "agent": "watermark",
      "status": "skipped",
      "score_fake": null,
      "confidence": null,
      "evidence_count": 0,
      "error": null
    }
  },
  "verdict": {
    "label": "real",
    "score_fake": 0.4,
    "confidence": 0.2,
    "rationale": "Analyzed with 1 agent(s). Average fake score: 0.400. Score range: 0.400 - 0.400.",
    "evidence_count": 2
  }
}
```

---

## 使用建议

1. **查看完整结果**：使用 `--json` 参数获取完整的 JSON 输出
2. **关注 `verdict.label`**：这是最终的判定结果
3. **检查 `verdict.confidence`**：置信度越高，结果越可靠
4. **查看 `results`**：了解各个智能体的分析情况
5. **查看 `artifacts.meta`**：了解视频的基本信息

---

## 配置阈值

可以在 `src/config/config.yaml` 中调整判定阈值：

```yaml
decision_policy:
  threshold_fake: 0.7   # 假视频阈值（>= 此值判定为假）
  threshold_real: 0.7   # 真实视频阈值（<= 1-此值判定为真）
```
