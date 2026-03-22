# 🚀 快速开始：视频真伪分析系统

## 其他用户使用指南（最简单）⭐

如果你想用自己的视频进行分析，只需要 **2 步**：

### 步骤 1：放置视频文件

**方式 A：放在 data 目录（推荐）**

如果你的数据集是 `Real/Fake` 结构：
```bash
# 将你的视频放到 data/ 目录下
cp /path/to/your/video.mp4 data/Real/your_video.mp4
# 或者
cp /path/to/your/video.mp4 data/Fake/your_video.mp4
```

如果你的数据集是任意结构：
```bash
# 可以放在 data/ 目录下的任意位置
cp /path/to/your/video.mp4 data/your_video.mp4
# 或者
cp /path/to/your/video.mp4 data/category1/your_video.mp4
```

**方式 B：使用绝对路径（不需要移动文件）**
```bash
# 直接使用视频的绝对路径，不需要放到 data 目录
python -m src.main /path/to/your/video.mp4
```

### 步骤 2：运行分析

```bash
# 如果视频在 data/Real/ 目录下
python -m src.main Real/your_video.mp4

# 如果视频在 data/Fake/ 目录下
python -m src.main Fake/your_video.mp4

# 如果使用绝对路径
python -m src.main /path/to/your/video.mp4
```

**就这么简单！** 系统会：
- ✅ 提取视频元数据（分辨率、时长、编码等）
- ✅ 提取关键帧（默认 24 帧）
- ✅ 运行多智能体分析（metadata、spatial、temporal、avsync、watermark）
- ✅ 生成真伪判定结果

**重要提示**：
- ⚠️ **必须先转换视频**：运行分析前，需要先运行 `python -m src.convert --label all` 转换视频
  - 如果数据集是 `Real/Fake` 结构：可以使用 `--label real` 或 `--label fake`
  - 如果数据集是任意结构：使用 `--label all`（系统会自动递归扫描所有视频）
- ⚠️ **Python 版本要求**：需要 Python >= 3.9（推荐 3.11+），因为 `langgraph` 需要 Python >= 3.9
- ✅ 转换后的缓存会保存在 `cache/evidence/` 目录，下次分析相同视频时会直接使用缓存，更快
- ✅ **支持任意数据集结构**：如果数据集不是 `Real/Fake` 结构，可以不配置 `dataset_structure`，系统会自动处理

**输出内容**：
- ✅ **判定结果**：`real`（真实）、`fake`（假）、`uncertain`（不确定）
- ✅ **置信度**：0-1，越高越可靠
- ✅ **各智能体分析结果**：metadata、spatial、temporal、avsync、watermark
- ✅ **视频元数据**：分辨率、时长、编码格式等
- ✅ **提取的帧信息**：帧数、时间戳等

详细输出格式说明请查看 [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md)

---

## 批量转换（可选，用于大量视频预处理）

如果你想提前转换大量视频，可以使用批量工具：

### 步骤 1：环境准备

**Python 版本要求**：>= 3.9（推荐 3.11+）

**使用 Conda（推荐）**：
```bash
# 激活 conda 环境（如果使用 conda）
conda activate Viagent

# 如果环境是 Python 3.8，升级到 3.11
conda install python=3.11 -y

# 安装 Python 包
pip install --upgrade pip
pip install pyyaml python-dotenv langgraph
```

**或使用 pip**：
```bash
# 确保 Python >= 3.9
python3 --version

# 安装 Python 包
pip3 install pyyaml python-dotenv langgraph
```

**安装 FFmpeg（系统依赖）**：
```bash
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# 验证安装
ffmpeg -version
```

### 步骤 2：准备视频和配置

**如果你的数据集是 `Real/Fake` 结构**：

确保视频文件在 `data/` 目录下：
```
data/
├── Real/        # 真实视频
│   └── *.mp4
└── Fake/        # 假视频（可选）
    └── *.mp4
```

在 `src/config/config.yaml` 中配置：
```yaml
paths:
  dataset_structure:
    real_dir: Real
    fake_dir: Fake
```

**如果你的数据集是任意结构**：

视频可以放在 `data/` 目录下的任意位置：
```
data/
├── category1/   # 任意目录名
│   └── *.mp4
└── *.mp4        # 甚至可以直接在 data/ 下
```

不需要配置 `dataset_structure`。

### 步骤 3：运行转换

**如果配置了 `dataset_structure`**：
```bash
# 转换真实视频
python -m src.convert --label real

# 转换假视频
python -m src.convert --label fake

# 转换所有视频
python -m src.convert --label all
```

**如果未配置 `dataset_structure`**：
```bash
# 转换所有视频（递归扫描所有 .mp4 文件）
python -m src.convert --label all
```

**就这么简单！** 转换后的缓存会保存在 `cache/evidence/` 目录。

---

## 常用命令

```bash
# 转换所有视频（支持任意数据集结构）
python -m src.convert --label all

# 如果配置了 dataset_structure，可以使用：
# 转换所有真实视频
python -m src.convert --label real

# 转换所有假视频
python -m src.convert --label fake

# 只转换 10 个视频（测试用）
python -m src.convert --label all --limit 10

# 转换特定生成器的假视频（需要配置 dataset_structure）
python -m src.convert --label fake --generator Sora

# 查看帮助
python -m src.convert --help
```

---

## 转换结果

转换完成后，每个视频会在 `cache/evidence/{video_id}/` 目录下生成：
- `meta.json` - 视频元数据（分辨率、时长、编码等）
- `frames/` - 提取的帧图片（默认 24 帧）

---

## 其他使用方式

### 方式一：直接运行主程序（最简单）⭐

**分析单个视频**：
```bash
# 在项目根目录执行
# 分析单个视频（相对路径，相对于 data 目录）
python -m src.main Real/video.mp4

# 分析单个视频（绝对路径）
python -m src.main /absolute/path/to/video.mp4

# 指定标签
python -m src.main Real/video.mp4 --label real

# 输出 JSON 格式
python -m src.main Real/video.mp4 --json

# 查看帮助
python -m src.main --help
```

**输出示例**：
```
🔍 正在分析视频: /path/to/video.mp4

✅ 分析完成!
   视频ID: xxxxxx
   判定: real
   置信度: 0.85
   理由: Average fake score (0.15) is below the real threshold (0.30).

📊 智能体结果:
   - metadata: ok, score_fake=0.20
   - spatial: skipped, score_fake=N/A
   - temporal: skipped, score_fake=N/A

📁 缓存位置: cache/evidence/xxxxxx/
```

### Python 代码调用（适合集成）
```python
from src.pipeline.analyze import analyze_video

# 分析单个视频
result = analyze_video("Real/video.mp4")
print(result['verdict'])
```

---

## 遇到问题？

1. **找不到模块**：确保在项目根目录执行命令
2. **FFmpeg 错误**：检查 `ffmpeg -version` 是否正常
3. **视频路径错误**：确保视频在 `data/Real/` 或 `data/Fake/` 目录下
