# 实验结果 / Experiment results

来源：`code/runs/*/metrics.json`，汇总日期：2026-09-30。所有正式实验使用 seed=17、batch=32、context=256、4 CPU threads、FP32、PyTorch 2.7.1。

## 1. 验证集 / Validation

| Run | Width | Dropout | Steps | Targets | Parameters | Validation BPB | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 128 | 0.00 | 1,200 | 9,830,400 | 1,088,256 | 2.07109 | 209.56 |
| dropout-s17 | 128 | 0.10 | 1,200 | 9,830,400 | 1,088,256 | 2.08875 | 218.09 |
| dropout005-s17 | 128 | 0.05 | 1,200 | 9,830,400 | 1,088,256 | 2.08191 | 232.16 |
| swiglu-s17 | 128 | 0.00 | 1,200 | 9,830,400 | 1,088,424 | 2.00788 | 187.87 |
| swiglu-dropout-2400 | 128 | 0.05 | 2,400 | 19,660,800 | 1,088,424 | 1.83138 | 474.00 |
| swiglu-dropout-6000 | 128 | 0.05 | 6,000 | 49,152,000 | 1,088,424 | 1.73074 | 1255.36 |
| swiglu-dropout-width192-6000 | 192 | 0.05 | 6,000 | 49,152,000 | 2,223,232 | 1.66915 | 1644.08 |

训练时间取 `train_seconds`，不含记录的中间验证计时；不是评分时间。不同步数使用按总步数计算的不同余弦调度，不能把长训练解释为短训练 checkpoint 的直接延续。

## 2. 已有完整测试结果 / Existing full-test results

| Run | Test BPB | Scoring seconds |
|---|---:|---:|
| baseline | 2.10126520 | 4.6794 |
| swiglu-dropout-2400 | 1.86282771 | 4.5065 |
| smoke | 3.61884773 | 4.7396 |

宽度 192 和宽度 128 的 6000 步模型均没有已保存的 test JSON；不得为其填写其他模型的测试成绩。Smoke 为 10 步流程检查，排除在正式方法比较之外。

## 3. 记录覆盖的成本 / Recorded cost

共 7 个正式训练和 1 个 10 步 smoke：157,368,320 个训练目标，训练累计 4,222.98 秒（70.38 分钟），训练进程累计 4,502.94 秒（75.05 分钟）。不包含失败、删除或未保存的运行、独立评分、安装和人工开发时间，因此不是所有投入的完整上界。

## 4. 资源状态 / Resource status

宽度 192 checkpoint：8,913,653 bytes = 8.50 MiB，仅为单个 checkpoint 大小，尚未核算完整推理包。CPU 峰值 RAM 未记录；CPU 运行的 `peak_allocated_gb=0` 是 CUDA 指标占位值，不能当成 RAM=0。最终候选的完整测试耗时比尚未验证。

详尽来源、曲线和哈希见 `experiment_evidence.json`。
