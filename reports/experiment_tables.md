# 实验汇总 / Experiment summary

| ID | Model | Width / depth | Dropout | Steps | Parameters | Val BPB | Train s | Device / precision |
|---|---|---|---:|---:|---:|---:|---:|---|
| E1 | GELU | 128 / 4 | 0.00 | 1200 | 1,088,256 | 2.07109 | 209.56 | CPU / FP32 |
| E2 | GELU | 128 / 4 | 0.10 | 1200 | 1,088,256 | 2.08875 | 218.09 | CPU / FP32 |
| E3 | GELU | 128 / 4 | 0.05 | 1200 | 1,088,256 | 2.08191 | 232.16 | CPU / FP32 |
| E4 | SwiGLU | 128 / 4 | 0.00 | 1200 | 1,088,424 | 2.00788 | 187.87 | CPU / FP32 |
| E5 | SwiGLU | 128 / 4 | 0.05 | 2400 | 1,088,424 | 1.83138 | 474.00 | CPU / FP32 |
| E6 | SwiGLU | 128 / 4 | 0.05 | 6000 | 1,088,424 | 1.73074 | 1255.36 | CPU / FP32 |
| E7 | SwiGLU | 192 / 4 | 0.05 | 6000 | 2,223,232 | 1.66915 | 1644.08 | CPU / FP32 |
| E8 | SwiGLU | 192 / 6 | 0.05 | 6000 | 3,113,472 | 1.63767 | 2361.73 | CPU / FP32 |
| E9 | SwiGLU | 192 / 6 | 0.00 | 6000 | 3,113,472 | 1.66182 | 2411.34 | CPU / FP32 |
| E10 | GELU | 192 / 6 | 0.05 | 6000 | 3,111,936 | 1.60688 | 2611.32 | CPU / FP32 |
| E11 | RoPE + GELU | 256 / 6 | 0.10 | 6000 | 5,263,360 | 1.52770 | 158.95 | A100 / BF16 |

所有正式实验 seed=17，batch=32，context=256。Targets = steps × 8192。Val BPB 来源为训练日志；RoPE 的独立 CPU FP32 验证值为 1.5276982899500708。训练时间跨设备或并发运行不可用于证明结构加速。

## Run mapping

- E1: `code/runs/baseline`
- E2: `code/runs/dropout-s17`
- E3: `code/runs/dropout005-s17`
- E4: `code/runs/swiglu-s17`
- E5: `code/runs/swiglu-dropout-2400`
- E6: `code/runs/swiglu-dropout-6000`
- E7: `code/runs/swiglu-dropout-width192-6000`
- E8: `code/runs/swiglu-dropout-width192-depth6-6000`
- E9: `code/runs/swiglu-width192-depth6-6000`
- E10: `code/runs/gelu-dropout-width192-depth6-6000`
- E11: `code/runs/rope-width256-depth6-6000`

包括现存 smoke 记录：CPU 训练 11607.37 秒；A100 训练 158.95 秒；累计训练目标 353,976,320。不包括丢失、失败或未记录的运行及本次检验以外的额外成本。

RoPE 完整 CPU FP32 test BPB：1.543998614826954；单次评分 16.2610 秒。验证集资源复测的耗时比为 2.543 倍、峰值 RSS 为 1.735 GiB。完整测试集峰值内存尚未记录。
