# 公交线路运行数据分析系统

## 原始需求

> 用 Python 分析公交线路运行数据，输入车辆 GPS 轨迹、站点到离站时间、刷卡人数、班次计划、道路拥堵指数和乘客投诉。分析过程需要清洗 GPS 漂移、缺失到站、车辆跳站、临时绕行和节假日班表差异，再计算站间运行时间、发车间隔、准点率、满载时段、拥堵路段和投诉高发点。输出每条线路的瓶颈站点、晚点原因拆分和调度建议，让车队知道该加车、调整发车间隔还是优化站点停靠。

## 项目简介

本系统用于公交公司对线路运行数据进行深度分析，支持从多源数据输入到最终调度建议的全流程自动化处理。系统输出包括：每条线路的瓶颈站点识别、晚点原因归因拆分、以及加车/调整间隔/优化停靠等具体调度建议。

## 技术栈

- Python 3.10+
- pandas 2.0+ - 数据处理与分析
- numpy 1.24+ - 数值计算
- geopy 2.4+ - GPS 坐标距离计算
- scikit-learn 1.3+ - 统计分析
- matplotlib / seaborn - 可视化（可选）

## 目录结构

```
wl-298/
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py      # 数据清洗模块
│   ├── metrics_calculator.py # 指标计算模块
│   └── analysis.py           # 瓶颈分析与调度建议
├── main.py                   # 主入口脚本
├── generate_sample_data.py   # 示例数据生成脚本
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## 输入数据格式

在 `sample_data/` 目录下放置以下 CSV 文件：

| 文件 | 关键字段 |
|------|---------|
| `gps_data.csv` | vehicle_id, route_id, timestamp, latitude, longitude, speed_kmh |
| `stop_data.csv` | route_id, direction, trip_id, vehicle_id, stop_id, stop_sequence, arrival_time, departure_time |
| `swipe_data.csv` | card_id, route_id, stop_id, vehicle_id, swipe_time, swipe_type |
| `schedule_data.csv` | route_id, direction, trip_id, vehicle_id, stop_id, stop_sequence, scheduled_arrival, scheduled_departure, service_date |
| `congestion_data.csv` | road_segment_id, timestamp, congestion_index (0-10), avg_speed_kmh |
| `complaint_data.csv` | complaint_id, route_id, stop_id, complaint_type, complaint_time, severity |

## 启动方式

### 前置要求

- Python 3.10 或更高版本
- pip 包管理器
- 可选：Docker 20.10+ 和 Docker Compose v2+

### 启动步骤

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 生成示例数据（可选，用于演示）

```bash
python generate_sample_data.py
```

示例数据会生成到 `sample_data/` 目录下。

#### 3. 运行分析

```bash
# 使用示例数据
python main.py --generate-sample

# 使用已有数据
python main.py --data-dir ./sample_data --output-dir ./output
```

#### 4. 查看结果

分析完成后，控制台会打印汇总报告，详细 JSON 报告保存在 `output/` 目录下，文件名格式为 `analysis_report_YYYYMMDD_HHMMSS.json`。

## Docker 一键启动（推荐）

### 前置要求

- Docker 20.10 或更高版本
- Docker Compose v2 或更高版本

### 启动步骤

#### 1. 构建并启动

```bash
docker compose up --build
```

后台运行：

```bash
docker compose up --build -d
```

#### 2. 查看日志

```bash
docker compose logs -f
```

#### 3. 停止并清理

```bash
docker compose down
```

#### 4. 结果输出

容器运行结束后，分析结果保存在宿主机 `./output/` 目录，示例数据保存在 `./sample_data/` 目录，可直接查看。

## 功能说明

### 数据清洗模块 (`src/data_cleaning.py`)

- **GPS 漂移清洗**：基于速度阈值和相邻点距离过滤异常 GPS 点，使用线性插值补全
- **缺失到站修复**：根据前后站点时间插值补全漏记的到站记录
- **跳站检测**：识别车辆在计划站点序列中跳过的站点
- **临时绕行识别**：检测不在计划站点列表中的异常停靠站序列
- **节假日班表区分**：自动标记工作日/周末/节假日，分别计算指标

### 指标计算模块 (`src/metrics_calculator.py`)

- **站间运行时间**：平均、中位数、P95、变异系数
- **发车间隔**：分时段平均间隔、间隔不规则度（CV）
- **准点率**：线路级、站点级、早晚偏差统计
- **满载时段**：站点小时级客流分级（低/中/高/超载）
- **拥堵路段**：路段拥堵指数分级、严重拥堵占比
- **投诉高发点**：按线路/站点/时段/类型聚合投诉

### 分析与建议模块 (`src/analysis.py`)

- **瓶颈站点识别**：综合低准点率、高延误累积、高客流、高运行时间变异度四维度评分
- **晚点原因拆分**：早高峰拥堵、晚高峰拥堵、绕行、乘客投诉、上下客时间等归因百分比
- **时段分段对比**：工作日早高峰、平峰、晚高峰和周末分开分析，识别通勤时段问题
- **雨天影响分析**：综合评估雨天对运营的影响，判断是否需要雨天临时加密班次
- **调度建议输出**：
  - 加车建议（客流高峰时段）
  - 发车间隔调整建议（间隔不规则度过高）
  - 班次计划调整建议（整体准点率过低）
  - 线路优化建议（绕行频发）
  - 站点停靠优化建议（特定投诉类型高发）
  - 雨天临时加密班次建议（雨天运行时间/满载率显著增加）

## 输出示例

分析报告 JSON 结构：

```json
{
  "cleaning_report": { "gps_drift_removed": 12, "missing_stops_filled": 5 },
  "route_metrics": { "on_time_rate": {...}, "headway": {...} },
  "route_analysis": {
    "101路": {
      "bottleneck_stops": [
        { "stop_id": "S101005", "severity_score": 78.5, "bottleneck_type": "low_on_time_rate" }
      ],
      "delay_breakdown": {
        "percentage_breakdown": { "morning_peak_congestion": 45.2, "evening_peak_congestion": 30.1 }
      },
      "dispatch_suggestions": [
        { "priority": "high", "action": "在客流高峰时段增加运力，建议加车 2 辆" }
      ]
    }
  }
}
```

## 注意事项

- 准点率默认阈值为 ±2 分钟，可在 `MetricsCalculator(on_time_threshold_minutes=N)` 中调整
- 节假日列表可在 `main.py` 的 `holidays` 参数中自定义
- 所有时间字段均需为可解析的 datetime 格式字符串
