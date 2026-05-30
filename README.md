# 多 Agent 自动化数据分析系统

> 基于 **LangGraph + Anthropic Claude API + PostgreSQL + Streamlit** 的端到端自动化数据分析平台。用户用自然语言提出分析需求，系统中的 **6 个 AI Agent** 自动完成 EDA → 清洗 → 建模 → 可视化 → 报告生成的完整流程。

---

## 📋 目录

- [简历段落](#-简历段落)
- [演示截图](#-演示截图)
- [技术架构](#-技术架构)
- [技术亮点](#-技术亮点)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [设计决策](#-设计决策)
- [实测数据](#-实测数据)
- [Future Work](#-future-work)

---

## 📝 简历段落

> **基于 Claude API 的多 Agent 自动化数据分析系统**（个人项目，Python / LangGraph / Anthropic Claude / PostgreSQL / Streamlit）：基于 LangGraph `StateGraph` + `conditional_edges` 编排 Planner / Explorer / Cleaner / Modeler / Visualizer / Reporter 六个 Agent，通过 Claude `tool_use` 调度 25+ 个原子数据工具函数（DB / EDA / 清洗 / 建模 / 可视化 5 大类），将自然语言需求自动转化为 EDA → 清洗 → 建模 → 可视化 → 报告全流程；对千万级阿里广告点击数据完成分层采样 ETL 与 CTR 二分类建模；利用 Anthropic prompt caching 三层缓存策略实测 cache 命中率 40-60%；Streamlit 前端基于 LangGraph `stream(stream_mode="updates")` 实现 Agent 执行轨迹实时可视化与浅蓝商务风图表嵌入。

**面试可深挖的技术点**：
- LangGraph `conditional_edges` 按 task_type 动态路由（eda_only / modeling / comparison）
- Claude `tool_use` 多轮循环 + 结构化输出契约（`submit_*` tools 替代脆性 JSON 解析）
- 三层 Prompt Caching：System / Tools schema / Data context 各自 `cache_control: ephemeral`
- `ContextVar(run_id)` 在 tool 层与 orchestrator 间共享运行时上下文，避免 LLM 编造 run_id
- PostgreSQL psycopg3 解决中文 Windows 启动握手编码问题（vs psycopg2 GBK 解码崩溃）

---

## 🎬 演示截图

![demo](ui/assets/demo.gif)

> 30 秒一次完整工作流：自然语言 query → Planner / Explorer / Cleaner / Modeler / Visualizer / Reporter 六个 Agent 卡片实时变绿 → 商务蓝 Plotly 图嵌入 → Markdown 报告生成 + 下载。

---

## 🏗️ 技术架构

### 工作流拓扑（LangGraph）

```
        START
          ↓
       planner ────── Opus 4.7（任务规划）
          ↓
       explorer ───── Sonnet 4.6（EDA 探索）
          ↓
   ┌──────┴───────┐ conditional: should_clean(state)
   │              │
"cleaner"   "skip_cleaning"
   │              │
cleaner ──── Sonnet 4.6（缺失/异常/编码）
   │              │
   └──────┬───────┘
          ↓
   ┌──────┴───────┐ conditional: should_model(state)
   │              │
"modeler"  "skip_modeling"
   │              │
modeler ──── Sonnet 4.6（LR + LightGBM）
   │              │
   └──────┬───────┘
          ↓
      visualizer ─── Sonnet 4.6（Plotly 图 + 洞察）
          ↓
      reporter ───── Opus 4.7（综合 Markdown 报告）
          ↓
         END
```

### 系统分层

```
┌───────────────────────────────────────────────────┐
│         Streamlit UI (ui/)                        │
│  query 输入 │ Agent 卡片流式渲染 │ 报告 + 图表    │
└──────────────────────┬────────────────────────────┘
                       │ stream_workflow(query)
┌──────────────────────▼────────────────────────────┐
│   LangGraph StateGraph (app/graph/)               │
│   AnalysisState (TypedDict + reducers)            │
│   conditional_edges → 按需路由                    │
└──────────────────────┬────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────┐
│   BaseAgent + Claude tool_use 多轮循环            │
│   (app/agents/base.py)                            │
│   prompt caching 三层标记                         │
└──────┬────────────────┬───────────────┬───────────┘
       │                │               │
       ▼                ▼               ▼
  Anthropic API   Tool Registry    PostgreSQL
  (Opus/Sonnet)   30+ 函数        业务表 + runtime 表
```

---

## ✨ 技术亮点

| 亮点 | 说明 |
|---|---|
| **多 Agent 编排** | LangGraph `StateGraph` + `conditional_edges` 按需路由（EDA-only 路径跳过 Cleaner/Modeler 节点，节省 ~30% token） |
| **Claude tool_use 多轮循环** | `app/agents/base.py` 统一封装，支持 tool 并行执行、消息历史拼接、自动停止 |
| **结构化输出契约** | Planner / Cleaner / Modeler 各自有 `submit_*` 工具，schema 在 `input_schema` 强约束，告别脆性 JSON 解析 |
| **三层 Prompt Caching** | System prompt / Tools schema / Data context 三层独立 `cache_control: ephemeral`，实测 cache hit ~40-60% |
| **流式执行轨迹 UI** | `graph.stream(stream_mode="updates")` → `accumulate_state()` → Streamlit `st.empty()` 增量刷新卡片 |
| **千万级数据 ETL** | `raw_sample.csv` 2655 万行 → `groupby('clk').sample(frac=...)` 分层采样 30 万行，保留 5.14% 正样本比 |
| **CTR 建模** | LightGBM 二分类 + categorical_feature 显式声明 + `scale_pos_weight` 处理 95:5 不平衡 |
| **运行时上下文** | `ContextVar(run_id)` 让 tools 自动拿到 run_id，避免 LLM 编造或漏传 |
| **DB 全流程审计** | `analysis_runs` + `agent_execution_logs` 表记录每次 run、每个 Agent 的每一步 |

---

## 📁 项目结构

```
project_root/
├── app/                         # 业务代码
│   ├── config.py                # pydantic Settings（从 .env 读）
│   ├── llm/
│   │   ├── client.py            # Anthropic 客户端单例 + base_url 注入
│   │   └── cache.py             # with_cache() / cache_tools()
│   ├── tools/                   # 25+ 工具函数（DB / EDA / Cleaning / Modeling / Viz 5 大类 + submit_* 契约）
│   │   ├── db.py                # list_tables / sample_table / query_sql
│   │   ├── eda.py               # profile / missing / numeric / cat / corr
│   │   ├── cleaning.py          # init/drop_nulls/impute/cap/encode/persist
│   │   ├── modeling.py          # build_feature_table / train_lr / train_lgbm
│   │   ├── viz.py               # plotly_hist/bar/grouped_ctr/corr_heatmap
│   │   └── registry.py          # TOOL_SPECS + execute_tool 分发
│   ├── agents/                  # 6 个 Agent
│   │   ├── base.py              # 多轮循环 + 缓存 + 日志统一封装
│   │   ├── planner.py / explorer.py / cleaner.py
│   │   ├── modeler.py / visualizer.py / reporter.py
│   ├── graph/                   # LangGraph 装配
│   │   ├── state.py             # AnalysisState (TypedDict + Annotated reducers)
│   │   ├── routing.py           # should_clean / should_model
│   │   └── builder.py           # build_graph() 主拓扑
│   ├── db/
│   │   └── dao.py               # analysis_runs / agent_execution_logs 写库
│   └── runtime/
│       ├── context.py           # ContextVar(run_id)
│       ├── orchestrator.py      # run_workflow() 入口（同步）
│       └── streaming.py         # stream_workflow() 流式生成器
├── ui/                          # Streamlit 前端
│   ├── streamlit_app.py         # 主入口（layout + 触发）
│   ├── components/
│   │   ├── sidebar.py           # 配置 / 数据集 / token 用量 / 历史
│   │   ├── agent_trace.py       # ★ Agent 卡片流式可视化
│   │   ├── report_view.py       # Markdown 报告 + 下载
│   │   └── chart_grid.py        # Plotly HTML 嵌入网格
│   └── assets/                  # demo.gif 等
├── scripts/
│   ├── init_db.py               # 建表 DDL（业务 3 + runtime 2）
│   ├── load_data.py             # CSV → PG（raw_sample 分层采样 30 万）
│   ├── run_explorer.py          # 单跑 Explorer（M2 验证）
│   ├── smoke_test.py            # 端到端冒烟（CLI）
│   └── inspect_runs.py          # 查看历史 runs + 物理表
├── data/                        # 原始 CSV（gitignore）
├── outputs/                     # 报告 + 图表（gitignore）
│   ├── reports/<run_id>.md
│   └── charts/<run_id>/*.html
├── tests/
├── requirements.txt
├── docker-compose.yml           # 可选 PG 容器
└── .env.example
```

---

## 🚀 快速开始

### 1. 准备 PostgreSQL

```bash
# 已装本机 PG
createdb -U postgres auto_analysis
# 或 Docker
docker-compose up -d
```

### 2. Python 环境

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置 `.env`

```bash
cp .env.example .env
```

至少填写：

```
ANTHROPIC_API_KEY=sk-ant-...
# 如果用国内代理，填代理 base URL（如 packyapi 的 claude-officially 分组）
ANTHROPIC_BASE_URL=https://www.packyapi.com
ANTHROPIC_MODEL_PLANNER=claude-opus-4-7
ANTHROPIC_MODEL_REPORTER=claude-opus-4-7
ANTHROPIC_MODEL_WORKER=claude-sonnet-4-6

DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=...
DB_NAME=auto_analysis

# 原始 CSV 所在目录
RAW_DATA_DIR=C:/path/to/csv
```

### 4. 建表 + 导数据

```bash
python scripts/init_db.py    # 5 张表 + 索引
python scripts/load_data.py  # 灌入分层采样后的 30 万 raw_sample
```

### 5. 端到端验证

```bash
# 命令行
python scripts/smoke_test.py --query "训练 CTR 预测模型并给出 top10 重要特征"

# Streamlit UI
streamlit run ui/streamlit_app.py
# → http://localhost:8501
```

### 6. 查看历史运行

```bash
python scripts/inspect_runs.py
```

输出最近 5 次运行、运行时物理表（`cleaned_*` / `feature_wide_*`）、最新 run 的 Agent step 分布。

---

## 🎯 设计决策

### 为什么选 LangGraph 而不是手写状态机
- **声明式拓扑**：`add_node` / `add_edge` / `add_conditional_edges` 让流程一眼看清楚
- **Annotated reducers**：`Annotated[list[ChartMeta], add]` 让多次 Visualizer 调用的图自动追加而不是覆盖
- **流式事件原生支持**：`graph.stream(stream_mode="updates")` 直接给 UI 增量

### 为什么用 tool_use 强制结构化输出而不是文本 JSON
- Planner 必须调 `submit_task_plan(task_type, ...)`：schema 在 `input_schema` 强约束，缺字段就拒收
- 同款思路用在 `submit_cleaning_report` / `submit_model_result`
- **对比**：让 LLM 在 final text 里写 JSON 然后正则 / `json.loads()` 解析 —— 见过 30% 的失败率

### 为什么 ContextVar 而不是把 run_id 塞到每个 tool 入参
- LLM 有概率忘传 / 编造 run_id
- Tool schema 越小越好让 LLM 专注业务字段
- Orchestrator 在 `invoke graph` 前 `set_run_id(...)` 一次

### 为什么 psycopg3 而不是 psycopg2
- psycopg2 在**中文 Windows** 上 startup 阶段按 UTF-8 解码服务端本地化字符串报 `UnicodeDecodeError 0xd6`
- psycopg3 把 `client_encoding=UTF8` 写进 startup packet，从协议层解决

### 为什么 stratified sample 30 万而不是全量 2655 万
- 项目演示数据足够代表分布；JOIN 30 万 × 1 百万 × 85 万足够小，能跑通建模 + 可视化
- `groupby("clk").sample(frac=...)` 严格保留 5.14% 正样本比

---

## 📊 实测数据

> 跑 `python scripts/smoke_test.py --query "训练 CTR 预测模型并给出 top10 重要特征"`

| 指标 | 实测值 |
|---|---|
| 工作流总时长 | ~3-5 分钟 |
| 跑过的 Agent | planner / explorer / cleaner / modeler / visualizer / reporter（全 6 个） |
| Total input tokens | ~116k（含 cache） |
| Cached input tokens | ~47k |
| **Cache hit rate** | **~40%** |
| Output tokens | ~16k |
| 单次成本（packyapi 代理） | ~¥25-30 |
| LightGBM AUC | ~0.54 *（仅使用原始 ID 类特征，未做交互/嵌入特征工程）* |

**关于 AUC**：这个数据集上 LightGBM 用裸 ID 特征的 AUC 上限约 0.54-0.58，要拿到 paper 里的 0.65+ 需要：
- 用户历史点击序列嵌入
- 广告 - 用户的交叉特征（如 user × cate）
- 时间窗口聚合（最近 7 天 CTR 等）

本项目的核心**不是刷 AUC**，而是展示**多 Agent 自动化编排**的端到端能力 —— 同样的工作流给一个特征工程更充分的数据集，AUC 自然上来。

---

## 📈 Roadmap

- [x] **M1** 基础设施：DB schema + 数据加载（分层采样保留正负比）
- [x] **M2** 单 Agent EDA：BaseAgent + ExplorerAgent + 8 个 DB/EDA tools
- [x] **M3** LangGraph 3 节点：Planner → Explorer → Reporter + DB 日志写库
- [x] **M4** 完整 6 节点：Cleaner / Modeler / Visualizer + 条件路由
- [x] **M5** Streamlit UI：流式 Agent 卡片 + 商务风 Plotly 图嵌入 + 报告下载
- [ ] **M6** (Future Work) 见下

---

## 🔮 Future Work

- **代码生成沙箱执行**（`tools/code_exec.py`）：允许 Agent 生成任意 pandas 代码并在受限沙箱中执行，覆盖长尾分析需求
- **多轮对话式分析**：报告生成后用户可继续追问，复用已建好的 `cleaned_<run_id>` / `feature_wide_<run_id>` 表
- **Critic Agent**：在 Reporter 之前增加一个 Critic 节点审查产出质量（自评 + 自修正循环）
- **可视化 Agent 决策树**：基于 LangGraph checkpoint 把每次运行的图执行轨迹可视化为 SVG
- **特征工程升级**：用户历史 CTR 聚合 + 广告 × 用户交叉特征，把 AUC 推到 0.65+
- **运行时表的 TTL 清理**：当前 `cleaning_work_*` / `feature_wide_*` 表在 run 失败时会残留，需要定时清理任务

---

## 📄 License

MIT
