# RoPE 检验与 CPU 推理测量

Validation set only; unchanged scorer; CPU FP32 4 threads; 3 alternating fresh processes each.

| 项目 | 结果 |
|---|---:|
| 验证 BPB | 1.52769829 |
| 基线评分中位数 | 4.2503 秒 |
| RoPE 评分中位数 | 10.8083 秒 |
| 耗时比 | 2.5430 倍 |
| 最高进程峰值 RAM | 1.7348 GiB |
| checkpoint 大小 | 20.1027 MiB |

Original small randomized RoPE tests: 5/5 pass. Trained checkpoint: normalization residual 1.6242265701293945e-6 exceeds unit tolerance 1e-6 but below evaluator tolerance 1e-3. Other tests pass; batch independence separately verified.

评分时间不含加载，RAM 包含加载和评分。三次完整验证评分成功。尚未做完整测试集资源验证，checkpoint 大小不是全部资产大小。没有修改模型或评分器。原训练日志的评分来自 A100 GPU，不能当成 CPU 时间。
