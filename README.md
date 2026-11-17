# MCAP 转 LeRobot v3

将 MCAP 转换为官方 LeRobot v3 数据集，包含时间对齐、质检和 HTML 看板。默认适配 Juliet：50 维 state、17 维 action、三路 RGB 相机；其他数据可通过 JSON profile 配置。

## 1. 安装环境

推荐 **Python 3.12、64 位 Windows**，CPU 即可，不需要安装 ROS 或 CUDA。依赖版本已写在 requirements 中。Linux 可安装相同依赖，尚未做实机验证。

下载或克隆仓库后，在项目目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip check
```

无需激活环境，后续直接调用虚拟环境中的 Python。只做预检时，可在新环境中仅安装 `requirements-audit.txt`。

## 2. 预检与转换

将输入路径改为自己的采集目录，输出路径指定一个**尚不存在的目录**。

先预检：

```powershell
.\.venv312\Scripts\python.exe .\juliet_mcap_to_lerobot.py --input "E:\test-11-12" --output ".\output\juliet_v3" --repo-id "local/juliet_v3" --config .\configs\juliet_quality.json --audit-only
```

确认报告后，去掉 `--audit-only` 正式转换：

```powershell
.\.venv312\Scripts\python.exe .\juliet_mcap_to_lerobot.py --input "E:\test-11-12" --output ".\output\juliet_v3" --repo-id "local/juliet_v3" --config .\configs\juliet_quality.json
```

`--repo-id` 是数据集标识，不会自动上传。默认按 **30 FPS** 对齐：state 和图像选最近样本，action 使用当前时间之前最近的指令。写出后检查结构、帧数，并用官方加载器抽查，通过后才发布正式输出目录。

## 3. 输出文件

```text
output/
├── juliet_v3/
│   ├── data/                        # state、action 等 Parquet 数据
│   ├── videos/                      # 各相机视频
│   └── meta/
│       ├── info.json                # 数据结构、版本、设备与标定
│       ├── stats.json               # 均值、标准差、分位数等统计
│       ├── tasks.parquet
│       ├── episodes/                # 官方 episode 元数据
│       └── juliet/
│           ├── episode_status.jsonl # 来源、设备/标定引用、质检和问题区间
│           ├── annotations.jsonl    # 每集标注，子任务默认留空
│           └── robot.urdf           # 源数据提供时保存
└── juliet_v3.quality_report.html     # 浏览器打开，离线查看逐集质检
```

台账每个输入 episode 一行，被拒绝的也保留，输出编号为 `null`。问题帧从 0 编号，区间为 `[start_frame, end_frame)`。仅预检或转换失败时，台账放在输出目录旁边；转换成功后只保留内部这一份。

state/action 的逐维名称仅保存在 `info.json` 的 `features` 中，不保存完整读取 profile。数据集版本为 `juliet.dataset_version`，当前 `0.0.1`，在主脚本的 `DATASET_VERSION` 中修改。

## 4. 质检与通用适配

阈值在 `configs/juliet_quality.json` 中。缺流、时间戳异常、对齐超限和指令过旧等错误会拒绝整集；图像模糊、亮度异常、图像复用及适用的动作跳变、关节越界会记录警告，仍可写入。当前阈值是工程默认值，尚未针对数据校准，也未接入冻结或静止检测。通过表示没有触发已启用规则。

其他数据集可复制以下示例，修改 topic、字段和输出名称，再通过 `--profile` 指定：

| 配置 | 用途 |
|---|---|
| `configs/generic_array.json` | 固定顺序数组，state/action 可以不同维度 |
| `configs/generic_joints.json` | 按关节名称匹配、多 topic、标量与相机 |

`mapping: "by_name"` 按原始关节名称取值；`array` 按数组顺序或 `indices` 取值；`scalar` 读取单个数值。配置列表和 `names` 的顺序决定最终向量顺序。一份输出数据集需保持固定维度和结构。

支持 JSON、ROS2 CDR 消息及原始 `rgb8/bgr8` 图像；暂不支持任意 Protobuf、ROS1 或 H.264 压缩图像输入。完整参数可查看：

```powershell
.\.venv312\Scripts\python.exe .\juliet_mcap_to_lerobot.py --help
```

上传 GitLab 时提交四个 Python 源文件、`configs/`、两份 requirements、README 和 `.gitignore`。不要提交虚拟环境、原始数据、转换结果和缓存。
