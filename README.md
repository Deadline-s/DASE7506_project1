# DASE7501-mp1

**姓名 / Name:** 李昊澎 / Haopeng Li  
**学号 / Student ID:** 3036803563  
**课程 / Course:** DASE7506

从课程提供的 GPT 基线出发，比较 dropout、SwiGLU、训练步数、宽度与深度，并训练 RoPE + GELU + Dropout 模型。最终报告模型为 **6 层、宽度 256、6000 步**，完整测试 CPU FP32 BPB 为 **1.543998614826954**，越低越好。

## 报告与实验记录

- [中文报告 PDF](output/pdf/report_zh.pdf)
- [中文报告源文件](reports/report_zh.md) / [English report](reports/report_en.md)
- [实验对照表](reports/experiment_tables.md) / [实验记录汇总](reports/experiment_evidence.json)
- [资源预检查报告](reports/rope_resource_check/resource_report.md)
- [课程运行说明与技术规则](code/README.md) / [作业要求](GUIDE.md)

## 主要结果

| 模型 | CPU FP32 验证 BPB | 完整测试 BPB | 参数量 | 训练步数 |
|---|---:|---:|---:|---:|
| 课程基线，本项目运行 | 2.07109 | 2.10127 | 1,088,256 | 1200 |
| RoPE + GELU + Dropout | 1.52770 | **1.54400** | 5,263,360 | 6000 |

完整测试相对基线降低约 26.52%。训练预算、模型容量和位置编码等同时变化，因此该提升属于整套方案，不能全部归因于 RoPE。开发与模型比较采用验证成绩；完整消融和实验时间顺序见报告。

## 模型结构

| 组件 | 配置 |
|---|---|
| 实现 / 配置 | `student.py` / `configs/rope_width256_depth6.json` |
| 层数 / 宽度 | 6 / 256 |
| 注意力头数 / 每头维度 | 4 / 64 |
| 前馈网络 | GELU，256 → 1024 → 256 |
| 位置编码 | Q、K 上的全头 RoPE，base=10000 |
| 归一化 | Pre-LayerNorm 和最终 LayerNorm |
| Dropout | 0.10，embedding、注意力权重和残差分支 |
| 词表 / 上下文 | BPE-2048 / 256 |
| 权重共享 | token embedding 与输出层共享 |

模型使用因果注意力，各评分窗口独立，不跨窗口保留文本状态。最终 RoPE 分支使用 GELU；SwiGLU 是此前对照实验中的方案。

## 安装

使用 Python 3.12，在项目根目录运行：

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

macOS CPU：

```bash
python -m pip install torch==2.7.1
python -m pip install -r requirements.txt
```

Linux/Windows CPU 请将 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA CUDA 12.6 环境请改用：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。完整环境说明见 [code/README.md](code/README.md)。

## 下载模型并直接评分

**模型下载链接：待补充。** 当前 README 中的本地路径不代表权重已上传。下载链接需要指向与下列 SHA-256 一致的 checkpoint；代码链接应固定到对应 Git commit。

将下载的 `checkpoint.pt` 放到 `code/runs/rope-width256-depth6-6000/checkpoint.pt`。保留仓库中的模型源码、配置、分词器与数据；评分不需要重新训练。

```text
SHA-256:
49d0141168eb606d6be697087aa85aef6c3038d3a8b52a27fb6994fe726103f4
```

从 `code/` 目录执行完整测试：

```bash
python evaluate.py   --checkpoint runs/rope-width256-depth6-6000/checkpoint.pt   --device cpu --precision fp32 --threads 4 --split test
```

结果写入对应运行目录的 `test_cpu_fp32.json`，提交其中的 `bpb`。验证集评分使用相同命令，将 `--split test` 改为 `--split validation`。验证集用于开发；冻结后的测试评分用于报告与复现。

## 重新训练

以下命令从 `code/` 运行，输出目录必须尚不存在。训练记录使用 NVIDIA A100、BF16、seed 17、batch size 32 和 6000 步：

```bash
python train.py   --implementation student   --config configs/rope_width256_depth6.json   --device cuda --precision bf16 --threads 4   --seed 17 --steps 6000 --batch-size 32   --run-dir runs/reproduce-rope
```

macOS 可将设备和精度改为 `--device cpu --precision fp32`，耗时和数值可能不同。训练器保存最后一步权重；训练不能保证跨设备逐位复现已提交 checkpoint。开发时可加 `--eval-every 500` 记录中间验证成绩；已有 RoPE 运行未保存中间验证曲线。

## 资源与正确性检查

| 项目 | 实测值 | 测量范围 |
|---|---:|---|
| 最终模型完整测试评分时间 | 16.261 秒 | 单次 CPU FP32 测试 |
| 基线 / RoPE 评分时间中位数 | 4.250 / 10.808 秒 | 同机、4 线程，验证集各三次 |
| 时间比例 | 2.543× | 验证集预检查，限制为 5× |
| RoPE 峰值 RSS | 1.735 GiB | 验证集独立进程，限制为 4 GiB |
| checkpoint 大小 | 20.103 MiB | 单个权重文件；全部推理资产限制为 64 MiB |

这些结果尚不能替代完整测试上的成套资源认证：还需核对完整测试峰值 RAM、同期基线计时及全部未压缩推理资产。详情见资源报告。

原始小配置的五项接口测试通过。已训练模型的归一化残差约 1.62×10⁻⁶，高于严格单元测试容差 10⁻⁶，但低于评分器阈值 10⁻³，实际评分成功；不将其表述为所有严格检查均通过。

## 文件结构

```text
README.md                 项目首页与复现说明
GUIDE.md                  课程要求
code/
  student.py              学生模型与 RoPE 实现
  model.py                原始基线
  train.py                训练入口
  evaluate.py             课程固定评分器
  configs/                实验配置
  data/                   课程数据与分词器
  runs/                   本地训练输出；权重需另行提供下载
reports/                  双语报告、实验表与资源记录
output/pdf/report_zh.pdf  中文报告 PDF
```

## 个人工作、AI 协助与复用说明

我阅读并逐步理解课程 GPT 代码，编辑模型与配置，执行训练、检查日志并比较实验；主动提出增加 block 数量、讨论在继承 GPT 的基础上加入 RoPE，并补充 GELU、SwiGLU 和 dropout 对照。实验选择和结论需由我依据真实结果判断、理解并负责。

OpenAI Codex/ChatGPT 帮助解释概念与规则、建议实验、提供部分代码示例、诊断错误、将提供的 RoPE 示例整合进模型，运行部分接口与资源检查及一次早期测试评分，并整理实验记录、起草报告和 README。AI 的贡献包括实质性实现与写作协助，不仅是格式整理。

GPT 基线、训练和评分框架来自课程包；RoPE 实现参考了提供的示例文件。报告及 README 的展示结构参考 [vectorBH6/HKU_DASE7506_MP1](https://github.com/vectorBH6/HKU_DASE7506_MP1)，实验数据和个人贡献说明来自本项目。

## 数据声明

使用课程提供的 WikiText-2，原始文本来自 Wikipedia，分词器仅在训练集拟合。保留 [课程 README 的数据归属及许可说明](code/README.md#6-data-attribution)，包括 CC BY-SA 3.0 和 GNU Free Documentation License 声明。
