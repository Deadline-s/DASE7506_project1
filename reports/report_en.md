# DASE7501-mp1

DASE7501-mp1 · English report · 30 September 2026  
Name: Haopeng Li (李昊澎)  
Student ID: 3036803563  

## Abstract

Starting from the supplied GPT baseline, this project studies dropout, SwiGLU, training budget, width, depth, and rotary position embeddings. At 1200 updates, an approximately parameter-matched SwiGLU model reduces validation BPB from 2.07109 to 2.00788. However, at width 192, six layers, and 6000 updates, GELU with dropout achieves 1.60688, outperforming SwiGLU with dropout at 1.63767. The benefits therefore depend on the experimental setting. A subsequent width-256 combination of RoPE, GELU, and dropout at multiple locations achieves CPU FP32 validation BPB of 1.52770 and full-test BPB of 1.54400. Several factors change in this combination, so its entire gain cannot be attributed to RoPE. The report presents successful and unsuccessful experiments, matched comparisons, ablations, costs, and limitations.

## 1. Task, Data, and Baseline

The task uses protocol `7506-mp1-wt2-v2`, the supplied WikiText-2 splits, and the fixed BPE-2048 tokenizer. Prediction is causal, with independent windows of 256 targets and no text-dependent state carried across windows. Validation contains 376,599 scored targets and 1,148,007 UTF-8 bytes; test contains 428,405 targets and 1,292,013 bytes. Data and the evaluation protocol remain fixed.

BPB is `sum[-ln p(actual next token | observed prefix)] / [ln(2) × raw UTF-8 bytes]`. Lower is better. It is neither accuracy nor directly comparable to perplexities using other tokenization protocols.

The supplied baseline has four layers, width 128, four attention heads, and 1,088,256 parameters. It uses learned absolute positions, pre-layer normalization, causal attention, GELU feed-forward layers, residual connections, and tied embeddings. This project implements and evaluates known techniques using the course code rather than claiming a new algorithm.

## 2. Final Model Design

### 2.1 Architecture Overview

| Component | Final RoPE configuration |
|---|---|
| Transformer depth / width | 6 / 256 |
| Attention heads / head dimension | 4 / 64 |
| Attention | Standard causal multi-head attention; full-head RoPE on Q and K |
| Feed-forward network | GELU, 256 → 1024 → 256, with biases |
| Normalization | Pre-LayerNorm and final LayerNorm |
| Positions | Fixed-frequency RoPE, base=10000; no learned position table |
| Dropout | 0.10 on embeddings, attention weights, and both residual branches |
| Embedding / output head | 2048 × 256, tied weights |
| Parameters | 5,263,360 |
| Training budget | 6000 steps; 49,152,000 processed targets |

The data flow is token IDs → embedding and dropout → six causal RoPE blocks → LayerNorm → tied vocabulary projection. Each block applies LayerNorm, Q/K rotation, causal attention and a residual addition, followed by LayerNorm, a GELU feed-forward network and a second residual addition. The evaluation interface converts logits to FP32 natural-log probabilities.

This model uses full-head RoPE and ordinary multi-head attention. It does not use GQA, QK normalization, learnable temperature, output gating, RMSNorm, or EMA. Those components occur in the reference repository but are not completed methods in this project.

### 2.2 From Baseline to Final Recipe


**Feed-forward dropout.** Dropout is inserted after the feed-forward output, before residual addition. Initial experiments use probabilities 0.10 and 0.05. The intended benefit is reduced feature co-adaptation, but effectiveness is assessed from validation results rather than assumed.

**SwiGLU.** The baseline feed-forward network is replaced by `down(SiLU(gate(x)) × value(x))`. Its intermediate width is `round(8d/3)`, limiting the extra parameters introduced by three affine projections. At width 128 the total parameter difference from the baseline is only 168, enabling an approximately parameter-matched comparison.

**Budget and capacity.** Experiments compare 2400 versus 6000 steps, widths 128 versus 192, and four versus six layers. Matching processed targets does not match computation. Also, changing the total step count changes the cosine learning-rate trajectory, not merely training duration.

**RoPE combination.** The latest candidate removes learned absolute position embeddings and rotates Q and K inside each attention head while leaving V unchanged. Causal masking is retained, and each independent window starts positions at zero. The implementation uses GELU and dropout on embeddings, attention weights, attention outputs, and feed-forward outputs. Width also increases to 256 and dropout to 0.10. This is a combined architecture/capacity/regularization experiment, not an isolated RoPE ablation. The implementation inherits the course GPT interface and incorporates a supplied RoPE example; code reuse should also be acknowledged in the README.

## 3. Experimental Setup

All main runs use seed 17, batch size 32, and context 256. Runs of 1200, 2400, and 6000 updates process 9,830,400, 19,660,800, and 49,152,000 training targets respectively, including repeated sampling. The trainer starts from random initialization and has no checkpoint-resume path.

AdamW uses peak learning rate 0.001, weight decay 0.1, 100-step warmup and the supplied cosine schedule, with gradient clipping at 1.0. Earlier runs use CPU FP32. The RoPE training log records an NVIDIA A100-SXM4-40GB with BF16 training; its validation function uses FP32, and CPU FP32 evaluation was performed separately. Training times across devices or different concurrent workloads do not establish architectural speedups. Historical CPU models and concurrency were not fully recorded.

The trainer saves the final step rather than the validation-best checkpoint. All principal experiments use one seed and do not establish statistical significance. The table comes from surviving metrics files; exact paths and evidence are provided alongside this report.

## 4. Headline Results and Experimental Progression

### 4.1 Headline Results

| Metric | Initial baseline | Final RoPE combination |
|---|---:|---:|
| Full-test CPU FP32 BPB | 2.10127 | **1.54400** |
| Validation BPB, CPU FP32 | 2.07109 | **1.52770** |
| Parameters | 1,088,256 | 5,263,360 |
| Training steps | 1200 | 6000 |

The full-test BPB reduction is approximately **26.52%** relative to the recorded initial baseline. This is a whole-recipe improvement: training volume increases fivefold, and capacity, positions, and regularization also change. It is not an estimate of RoPE's isolated contribution. The result remains above the original target of 1.5.

### 4.2 Complete Validation Results


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


### 4.3 Training Dynamics

| Step | Six-layer SwiGLU + dropout, width 192 | Six-layer GELU + dropout, width 192 |
|---|---:|---:|
| 4000 | 1.66502 | 1.63785 |
| 4500 | 1.65261 | 1.62524 |
| 5000 | 1.64538 | 1.61609 |
| 5500 | 1.64046 | 1.60820 |
| 6000 | 1.63767 | 1.60688 |

Both validation curves continue to improve late in training, with GELU ahead at each listed point. The RoPE metrics file has an empty `validation_history` and only a final validation result. No unrecorded intermediate RoPE validation curve is plotted or inferred. Curve completeness and checkpoint evaluability are separate issues.

## 5. Ablations: Benefits and Limitations

| Comparison | Matched conditions | Validation BPB change | Supported observation |
|---|---|---|---|
| GELU → SwiGLU | Width 128, 4 layers, 1200 steps, no dropout | 2.07109 → 2.00788 | Gating helps in the small, short-budget setting |
| SwiGLU → GELU | Width 192, 6 layers, 6000 steps, dropout 0.05 | 1.63767 → 1.60688 | GELU wins in the larger setting |
| Remove dropout | Width 192, 6 layers, 6000 steps, SwiGLU | 1.63767 → 1.66182 | Dropout helps in this setting |
| 4 → 6 layers | Width 192, 6000 steps, SwiGLU + dropout | 1.66915 → 1.63767 | Depth improves validation at additional cost |



**Initial comparisons (E1–E4).** With 1200 updates, SwiGLU lowers validation BPB by 0.06320 relative to GELU. Both standalone dropout settings worsen the baseline. SwiGLU therefore helps in this small-model, short-budget setting, while dropout does not show a benefit. A common seed does not make parameter initialization identical across architectures.

**Budget and capacity (E5–E8).** Extending the combined width-128 recipe from 2400 to 6000 updates lowers BPB from 1.83138 to 1.73074. At 6000 updates, increasing width to 192 lowers it to 1.66915; increasing depth to six lowers it further to 1.63767. The latter two comparisons isolate width and depth within their respective settings, with increased computational cost.

**Six-layer ablations (E8–E10).** Replacing SwiGLU with GELU while retaining dropout improves BPB from 1.63767 to 1.60688, a decrease of 0.03080. Removing dropout while retaining SwiGLU worsens it to 1.66182, an increase of 0.02415. Thus, SwiGLU is not universally superior, while dropout helps the six-layer SwiGLU configuration. These comparisons match training-target counts and provide mechanism ablations. Pure SwiGLU reached 1.66141 at step 5500, but those weights were not saved; that score must not be assigned to the available final checkpoint.

**RoPE combination (E11).** CPU FP32 validation BPB is 1.52769829. This is below E10, but width, dropout placement and probability, training hardware, and training precision also change. A width-256 absolute-position control with identical regularization and training conditions is missing. RoPE's independent contribution is therefore unisolated, and the earlier SwiGLU ablation cannot substitute for this control. If RoPE is presented as the sole key mechanism, this remains an experimental gap.

## 6. Full-Test Results and Resource Checks

| Model | Full-test CPU FP32 BPB | Single scoring time (s) |
|---|---:|---:|
| Initial baseline | 2.10127 | 4.6794 |
| Width-128, 2400-step SwiGLU + dropout | 1.86283 | 4.5065 |
| Width-256, six-layer RoPE combination | 1.54400 | 16.2610 |

The latest exact test score is **1.543998614826954**, which is not below 1.5. Each score belongs to its matching checkpoint. An earlier model was tested before later development continued; the report cannot claim that all development preceded the first test evaluation. This chronology should be disclosed to the instructor, who determines compliance with the freezing requirement.

Three fresh-process validation measurements per model, on the same CPU with FP32 and four threads, give median scoring times of 10.808 seconds for RoPE and 4.250 seconds for the baseline: a ratio of 2.543, below the 5x limit. Maximum RoPE process peak RSS is 1.735 GiB, below 4 GiB. The checkpoint is 20.103 MiB, but the complete final inference bundle has not been inventoried. RSS includes loading and scoring; scoring time excludes loading. A full-test scoring time of 16.261 seconds is recorded, but a contemporaneous repeated baseline comparison and full-test peak RAM are missing. Validation prechecks are not full-test resource certification.

All five original small-model contract tests pass with RoPE selected. For trained weights, causality, state reset, gradients, and separately checked batch independence pass. The normalization residual is approximately 1.62e-6, exceeding the unit-test tolerance of 1e-6 but below the evaluator threshold of 1e-3. Scoring succeeds; this is not described as passing every strict trained-model test.

## 7. Cost and Reproduction

Surviving records cover eleven main runs and one ten-step smoke run, totaling 353,976,320 processed targets. Logged CPU training totals 11607.37 seconds and A100 training 158.95 seconds. These are reported separately, not as a hardware-independent compute budget. Totals exclude unsaved, failed or deleted runs, independent scoring, and human development time.

Learning uses the supplied training text; development comparisons are recorded on validation. Existing model branches are preserved, with RoPE selected by `variant="rope"`. The RoPE checkpoint SHA-256 is `49d0141168eb606d6be697087aa85aef6c3038d3a8b52a27fb6994fe726103f4`, matching the current weights and test record. Install dependencies according to the course README. Evaluation requires no retraining:

```bash
# Run from code/ after installing the documented dependencies.
python train.py --implementation student --config configs/rope_width256_depth6.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 6000 --eval-every 500 --run-dir runs/reproduce-rope
python evaluate.py --checkpoint runs/rope-width256-depth6-6000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```


The training command restates the main logged settings, not a guarantee of bitwise reproducibility across devices. Provide immutable code and matching downloadable weights, with paths and configuration consistent with the score.

## 8. Discussion and Conclusions

The principal finding is that improvements depend on scale and training conditions. The small-model SwiGLU benefit does not persist in the larger comparison, and dropout's effect also changes. More training and capacity improve validation quality at additional cost. The RoPE combination achieves the best recorded validation score and 1.54400 full-test BPB, but its independent positional-encoding benefit remains unresolved. Priorities for further evidence are a matched RoPE control, multiple seeds, and full-test resource verification.

## 9. Personal Work and AI Assistance

**Personal contribution.** I read and progressively worked through the supplied GPT and training code, edited student.py and configuration files with AI guidance, executed multiple training experiments, inspected logs, and compared settings. I proposed exploring additional blocks, discussed integrating RoPE through GPT inheritance, and carried out GELU, SwiGLU, and dropout comparisons. I participated in choosing and implementing the experimental progression rather than simply submitting a ready-made result. My analysis must follow the measurements, including accepting that GELU outperforms SwiGLU in the larger setting and distinguishing validation scores, test scores, and resource costs.

**AI assistance.** OpenAI Codex/ChatGPT explained concepts and assignment rules, suggested experiments, supplied some implementation examples, diagnosed errors, integrated the provided RoPE example into student.py, ran some interface and resource checks and one earlier test evaluation, and aggregated logs and drafted the bilingual reports. The baseline is course-provided and RoPE also draws on the supplied example file. This assistance does not replace my experimental work, understanding, or responsibility for the final submission. I must review the report against my actual work and retain the substantive AI and code-reuse disclosure in the README.


## 10. Data and Report Organization

The project uses the supplied WikiText-2 data, derived from Wikipedia, and a tokenizer fitted only to the training split. Redistribution should retain the CC BY-SA 3.0 and GNU Free Documentation License notices in the course README. No separate bibliography is included.

This revision uses the presentation hierarchy of the vectorBH6/HKU_DASE7506_MP1 report as an organizational reference: architecture configuration, headline results, training dynamics, ablations, and reproduction. Its model components, experimental numbers, and personal-contribution claims are not adopted. All numerical results come from this project's logs. Organizational reference: https://github.com/vectorBH6/HKU_DASE7506_MP1/blob/master/report/report.md
