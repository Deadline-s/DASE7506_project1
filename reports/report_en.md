# SwiGLU, Dropout, and Model Capacity in a Small Language Model

**DASE7506 · MP1 · English report draft · 30 September 2026**  
Name and student ID: [TO COMPLETE]  
Status: evidence-based draft, not a complete submission. No full-test result is recorded for the width-192 model. Resource verification, the final submission version, and artifact links remain to be confirmed. This Markdown document is not paginated; the exported report must be checked against the ten-page limit, including references.

## Abstract

This project trains small autoregressive language models from random initialization on the supplied WikiText-2 benchmark with a fixed BPE-2048 tokenizer. It investigates gated feed-forward layers, dropout, training budget, and model capacity. With the same 9,830,400 processed training targets and seed 17, replacing GELU with an approximately parameter-matched SwiGLU layer reduces validation bits per byte (BPB) from 2.07109 to 2.00788. Adding dropout alone at 0.05 or 0.10 does not improve the baseline. A width-128 SwiGLU model with dropout 0.05 reaches 1.73074 validation BPB after 6000 updates. Increasing width to 192 at the same training-target budget reduces this to 1.66915. These gains cannot all be attributed to a single mechanism, and validation results are not test results. The recorded full-test BPB of the earlier width-128, 2400-step combined model is 1.86283. All principal comparisons use one seed and do not establish statistical significance.

## 1. Task and Baseline

The task is to reduce full-test BPB while preserving the supplied data, tokenizer, and evaluator. Protocol `7506-mp1-wt2-v2` fixes the vocabulary at 2048 and uses independent causal windows of 256 targets without temporary state crossing windows. Every target except the first token of each split is scored, including the final short window. The metric is:

`BPB = sum[-ln p(x[t+1] | x[<=t])] / (ln(2) × raw UTF-8 bytes in the split)`.

Validation contains 376,599 scored targets and 1,148,007 bytes; test contains 428,405 targets and 1,292,013 bytes. Validation is intended for development and selection; test is intended for evaluating the frozen method. Dataset attribution is provided in the supplied README. All four current data/tokenizer hashes match the supplied data manifest.

The baseline is a four-layer, four-head GPT with width 128 and 1,088,256 parameters. It uses learned absolute positions, pre-layer normalization, causal attention, residual connections, and tied input/output embedding weights. These mechanisms are retained; the experiments change the feed-forward module, dropout, and width as specified.

## 2. Methods and Hypotheses

### 2.1 Approximately Parameter-Matched SwiGLU

The baseline feed-forward network is `Linear(d,4d) → GELU → Linear(4d,d)`. The replacement uses three affine projections:

`FFN(x) = down(SiLU(gate(x)) ⊙ value(x))`.

Here ⊙ denotes elementwise multiplication, and all three projections include biases. An input-dependent gate modulates the content branch. This is an implementation and evaluation of the GLU family studied by Shazeer [1], not a claim of a novel architecture. The hypothesis is that multiplicative gating improves feature transformation at a similar parameter budget.

The intermediate width is `round(8d/3)`, giving 341 for d=128 and 512 for d=192. At width 128, the baseline and SwiGLU models contain 1,088,256 and 1,088,424 parameters respectively, a difference of only 168. New linear layers use the baseline initialization: normal weights with standard deviation 0.02 and zero biases.

### 2.2 Feed-Forward Output Dropout

Dropout is applied after the feed-forward output and before residual addition: `x + Dropout(FFN(LayerNorm(x)))`. It is disabled during evaluation. The motivation is to reduce feature co-adaptation [2]. Baseline experiments use p=0.10 and p=0.05; later combined models use p=0.05. The baseline experiments do not support a benefit, and its separate contribution within the combined model remains unisolated.

### 2.3 Capacity and Training Budget

Width is increased from 128 to 192 with depth and head count unchanged, raising the parameter count to 2,223,232. Scaling research [3] motivates examining capacity but does not predict the gain in this small-data setting or identify 192 as an optimal width. Training-budget comparisons use 2400 and 6000 updates. Since the cosine schedule depends on the total number of steps, these runs also follow different learning-rate trajectories; the comparison is not simply a continuation of the same earlier checkpoint.

## 3. Experimental Setup and Provenance

All recorded main runs use seed 17, batch size 32, 256 targets per sequence, four CPU threads, FP32, and PyTorch 2.7.1. Training uses AdamW with peak learning rate 0.001, weight decay 0.1, a 100-step linear warmup, the supplied cosine factor with a minimum factor approximately 0.1, and gradient clipping at norm 1.0. Training windows are sampled randomly from the supplied training-token sequence and may overlap or repeat.

Processed targets equal `steps × 32 × 256`; they are not a count of unique tokens. The trainer contains no checkpoint-resume path, so each recorded run starts from random initialization. At report preparation, the environment is macOS 27.0/arm64 with Python 3.12.9. Historical logs identify the device only as CPU and do not record the chip model; the current environment is not a complete historical hardware record.

Evidence comes from `code/runs/*/metrics.json` and corresponding evaluation JSON files. Extracted records, curves, and hashes are included in `experiment_evidence.json`. Repository HEAD at draft preparation is `1235d3acae75bf19ada33b701446ea4a399a037d`; this does not establish that every training run used that commit and is not a designation of the final submission commit.

## 4. Results and Ablations

| Method | Width | Steps | Parameters | Validation BPB | Train (s) |
|---|---:|---:|---:|---:|---:|
| Baseline GELU | 128 | 1200 | 1,088,256 | 2.07109 | 209.56 |
| GELU + dropout 0.10 | 128 | 1200 | 1,088,256 | 2.08875 | 218.09 |
| GELU + dropout 0.05 | 128 | 1200 | 1,088,256 | 2.08191 | 232.16 |
| SwiGLU | 128 | 1200 | 1,088,424 | 2.00788 | 187.87 |
| SwiGLU + dropout | 128 | 2400 | 1,088,424 | 1.83138 | 474.00 |
| SwiGLU + dropout | 128 | 6000 | 1,088,424 | 1.73074 | 1255.36 |
| SwiGLU + dropout | 192 | 6000 | 2,223,232 | 1.66915 | 1644.08 |


Training time is the logged `train_seconds`, excluding the intermediate validation time accounted for by the trainer. It is not CPU scoring time. Exact run IDs and values are provided in `experiment_tables.md`.

### 4.1 Mechanism Comparisons at Matched Training Volume

At 1200 steps, SwiGLU reduces validation BPB by 0.06320, approximately 3.05%, relative to the baseline. Both runs process the same number of targets and have nearly equal parameter counts. This provides a feed-forward mechanism comparison and ablation: removing SwiGLU and restoring GELU gives the baseline configuration. However, replacing modules consumes additional random numbers, so a common seed does not make this a weight-by-weight matched experiment. Multiple seeds are needed to assess robustness.

Dropout 0.10 and 0.05 increase baseline validation BPB by approximately 0.01766 and 0.01082 respectively. Neither helps under this 1200-step recipe. One possible explanation is that regularization slows fitting under a limited budget; these experiments do not establish that explanation.

### 4.2 Training Budget and Capacity

For the width-128 combined model, increasing the budget from 2400 to 6000 steps reduces validation BPB from 1.83138 to 1.73074, a decrease of approximately 0.10064. The 6000-step validation curve declines from 1.84721 at step 2000 to 1.75890 at step 4000, 1.74111 at step 5000, and 1.73074 at the end. Late gains diminish, but no validation reversal is recorded.

At the same 49,152,000 processed targets, width 192 improves BPB by 0.06158, approximately 3.56%, over width 128. Recorded training time increases from 1255.36 to 1644.08 seconds, approximately 31.0%, while parameters increase by approximately 104.3%. This is a capacity-quality trade-off, not an isolated SwiGLU gain or a compute-matched comparison.

The best validation score is approximately 19.41% below the initial baseline. That overall difference combines changes to architecture, regularization, training budget, and capacity and cannot be attributed to one component.

### 4.3 Recorded Full-Test Results and Version Boundaries

| Model | Full-test BPB | Scoring time (s) |
|---|---:|---:|
| Baseline, width 128, 1200 steps | 2.10127 | 4.6794 |
| SwiGLU + dropout, width 128, 2400 steps | 1.86283 | 4.5065 |
| SwiGLU + dropout, width 192, 6000 steps | Not recorded | Not recorded |

The earlier 2400-step model was tested before subsequent 6000-step and width-expansion experiments. This chronology is disclosed rather than claiming that all development preceded the first test evaluation. Subsequent analysis here uses validation records, but the chronology still needs to be explained to the instructor under the requirement to freeze the method before testing; acceptance is for the instructor to determine. The score 1.86283 belongs only to the 2400-step checkpoint and cannot be assigned to the wider model. No recorded result establishes BPB below 1.5.

## 5. Cost, Resources, and Reproduction

The seven main training runs and one ten-step smoke run process 157,368,320 targets in total. Logged training time sums to 4222.98 seconds (70.38 minutes), and training-process time sums to 4502.94 seconds (75.05 minutes). These totals cover surviving logs only and exclude independent evaluation, failed or deleted runs, installation, and human development time.

The width-192 checkpoint occupies 8,913,653 bytes (8.50 MiB), below 64 MiB, but this is not an inventory of the complete uncompressed inference bundle. The earlier 2400-step model's recorded scoring time is about 0.963 times the baseline's, based on single historical measurements; this does not establish the wider model's scoring ratio. Peak CPU RAM has not been measured. CUDA memory fields that are zero on CPU must not be interpreted as zero RAM usage. Consequently, full resource compliance of the best-validation candidate remains unverified.

Install Python 3.12, PyTorch 2.7.1, and the remaining requirements as specified in the supplied README. Run commands from `code/`, using a fresh training output directory:

```bash
python train.py --implementation student --config configs/swiglu_dropout_width192.json --device cpu --threads 4 --seed 17 --steps 6000 --eval-every 500 --run-dir runs/reproduce-width192
python evaluate.py --checkpoint runs/swiglu-dropout-width192-6000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

Reproduce the existing 2400-step test result without retraining:

```bash
python evaluate.py --checkpoint runs/swiglu-dropout-2400/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The 2400-step checkpoint SHA-256 is `3468a9a6b28053ad13a9e746822fc9036ff7c6896ec15e6e206a8ea09df62895`. The width-192 checkpoint SHA-256 is `9ef14cf732103589f05c96f0bcffd1e5fd23eccbb6b3a9b4d1d362087d959757`. Both match the current files. The downloadable bundle must include the matching `student.py`, `model.py`, configuration, and all other required scoring files.

## 6. Limitations and Conclusions

These single-seed experiments support approximately parameter-matched SwiGLU and show additional validation gains from a larger training budget and width. They do not support a standalone dropout benefit. A 6000-step GELU control at the same width and a same-budget pure-SwiGLU control are missing, preventing separation of every component's contribution in the final combination. Single timing observations do not establish reliable acceleration, and single-seed results do not establish statistical significance.

Before submission, complete the author information, identify the frozen submission checkpoint, supply its evaluation and resource evidence, provide immutable code and checkpoint links, and export a report of at most ten pages. Width 192 is the current best-validation candidate, not automatically the model already submitted.

## 7. AI Assistance and Reuse Disclosure

The project reuses the supplied course GPT, training framework, and evaluation protocol. OpenAI Codex/ChatGPT assisted with interpreting the assignment, suggesting experiments, providing SwiGLU/dropout code examples, diagnosing errors, aggregating logs, executing one test evaluation of the 2400-step model, and drafting the Chinese and English reports. Early assistant-written modifications were reverted on request; subsequent code was edited by the student using the examples. The student must review the implementation and text and is responsible for the final submission. The work should not be described as entirely unaided. This disclosure must also be included in the repository README.

## References

[1] Shazeer, N. (2020). *GLU Variants Improve Transformer*. https://arxiv.org/abs/2002.05202  
[2] Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15, 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html  
[3] Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. https://arxiv.org/abs/2001.08361  
[4] DASE7506 MP1 supplied `GUIDE.md`, `code/README.md`, model and scoring code. Local assignment specification and experimental artifacts.
