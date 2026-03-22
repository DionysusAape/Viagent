# Viagent

基于 **LangGraph** 的多智能体视频真伪分析系统：先抽取帧与证据缓存，再经 Planner / Analysts / Judge 流水线给出判定，结果写入本地 SQLite。

---

## 如何跑通（最短路径）

### 1. 系统依赖

需要 **FFmpeg / ffprobe**（抽帧与元数据）。未安装时：

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 2. Python 与虚拟环境

推荐使用 **Python 3.11+**（至少 3.10）。

```bash
cd Viagent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 3. 环境变量与数据库

在项目根目录复制示例并编辑（**不要提交真实密钥**）：

```bash
cp env.example .env
```

至少配置：

| 变量 | 说明 |
|------|------|
| `DB_PATH` | SQLite 路径，默认示例为 `DB/database.db`（会先创建目录） |
| 与 LLM 对应的 API Key | 见下表 |

`src/config/config.yaml` 里 `llm.provider` 决定使用哪把密钥，例如默认若为 `yizhan`，需在 `.env` 中设置 `YIZHAN_API_KEY`。常见对应关系：

| `llm.provider` | 环境变量 |
|----------------|----------|
| `openai` | `OPENAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `kimi` | `KIMI_API_KEY` |
| `yizhan` | `YIZHAN_API_KEY` |
| `alibaba`（通义） | `QWEN_API_KEY` |
| `aihubmix` | `AIHUBMIX_API_KEY` |

`load_dotenv()` 从**当前工作目录**加载 `.env`。下面命令均在**项目根目录**执行，因此把 `.env` 放在根目录即可。

### 4. 准备视频数据

将 `.mp4` 放在数据根目录下任意子目录中（会递归扫描）。默认数据根为项目下的 `data/`。

可选：使用其他目录时设置环境变量（或在运行前 `export`）：

```bash
export DATA_ROOT=/path/to/your_videos
export CACHE_ROOT=/path/to/cache
```

未设置时默认为 `<项目根>/data` 与 `<项目根>/cache`。

### 5. 第一步：转换（抽帧，必需）

分析依赖 `cache` 下的证据（帧与 `meta.json`），**必须先做转换**：

```bash
# 在项目根目录
python src/convert.py --label all
```

测试时可限制数量：

```bash
python src/convert.py --label all --limit 5
```

若在 `src/` 目录下工作，等价命令为：`python convert.py --label all`。

### 6. 第二步：分析

```bash
# 单个视频（路径相对于 DATA_ROOT，例如 data/Real/xxx.mp4 → 参数写 Real/xxx.mp4）
python src/main.py --label Real/your_video.mp4

# 某一类目录下全部
python src/main.py --label Real

# 全部视频
python src/main.py --label all

# 指定配置文件（仅文件名，对应 src/config/ 下的 yaml）
python src/main.py --label all --config km.yaml

# 仅打印 JSON
python src/main.py --label Real/your_video.mp4 --json
```

### 7.（可选）统计

```bash
python src/statistics.py
python src/statistics.py --config config.yaml
python src/statistics.py --list-experiments
```

---

## 常见问题

**Q: 提示找不到证据 / Evidence cache not found？**  
先完成第 5 步转换，并确认 `CACHE_ROOT`（或默认 `cache/`）与转换时一致。

**Q: 提示缺少 API Key？**  
检查 `.env` 是否与 `config.yaml` 里 `llm.provider` 一致，且在**项目根目录**运行命令以便加载 `.env`。

**Q: 批量运行时很多条被跳过？**  
同一 `experiment_name` 下已有完整分析会跳过。可换配置文件（不同实验名）或清理对应实验数据后再跑。

**Q: 想用 `cd src` 再运行？**  
可以：此时请把 `.env` 复制到 `src/.env`，或在 shell 里先 `export` 所需变量。

---

## 配置说明（摘要）

- 主配置：`src/config/config.yaml`（`experiment_name`、`llm`、`workflow_analysts`、`human_eyes_enabled`、`planner_mode` 等）。
- 路径优先级：**环境变量 `DATA_ROOT` / `CACHE_ROOT`** 高于默认的 `data/`、`cache/`。
- 实验隔离：不同配置中的 `experiment_name`（或不同 yaml 文件名推导出的实验名）用于区分结果与断点续跑逻辑。

更细的架构说明、工作流与各 Agent 职责，见仓库内代码与 `src/config/` 下各 yaml 注释。

## 许可证

[待补充]
