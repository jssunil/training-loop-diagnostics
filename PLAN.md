# PLAN: Training Loop Diagnostics

This plan uses ASD-STE100 Simplified Technical English rules:
- Procedure sentences have 20 words or fewer. Description sentences have 25 words or fewer.
- Use the active voice and one instruction in each step.
- Procedures use the imperative ("Do this").
- A word has one meaning in this document.
- Each term is defined before it is used.

---

## 1. Goal

Use a small model and a real training loop. Make the loop show the truth about itself.

The project has six parts:

| Part | Task | Output |
|---|---|---|
| 1 | Print every tensor shape in one training step. Write one line about each dimension. | Shape table |
| 2 | Change one weight by a small amount. Compare the loss change with `backward()`. | Agreement table |
| 3 | Break gradient accumulation on purpose. Plot the broken curve and the correct curve. | Plot + numbers |
| 4 | Log the grad norm at each step. Find a step where the grad norm moved before the loss. | Plot + step number |
| 5 | Calculate MFU. Report it honestly. Explain the distance to 40%. | MFU + causes |
| 6 | Write 0.1 in fp32, bf16 and fp8 E4M3, bit by bit. Select a training format. | Bit table + decision |

The submission is a GitHub repository with `README.md` and a Jupyter notebook with its outputs.

---

## 2. Words

| Word | Meaning in this document |
|---|---|
| weight | One number in the model that training can change. |
| loss | One number that tells how wrong the model is on a batch. A lower loss is better. |
| gradient | One number for one weight. It tells how much the loss changes when that weight changes. |
| backward pass | The calculation that finds the gradient of every weight. `loss.backward()` does it. |
| autograd | The PyTorch system that records the forward pass and does the backward pass. |
| step | One loop: read data, calculate loss, calculate gradients, change all weights. |
| batch | The group of sequences that one step uses. |
| micro-batch | One part of a batch. Use micro-batches when the full batch does not fit in memory. |
| gradient accumulation | Add the gradients of many micro-batches. Then do one step. |
| token | One unit of text. In this model, one token is one character. |
| padding | Empty positions that fill a short sequence to full length. The loss must not count them. |
| grad norm | The length of all gradients together: the square root of the sum of all squared gradients. |
| clipping | If the grad norm is more than a limit, make all gradients smaller by the same factor. |
| FLOP | One floating-point operation (one add or one multiply). |
| MFU | Model FLOPs Utilisation. The useful FLOPs per second divided by the peak FLOPs per second of the hardware. |
| sign bit | The bit that tells if a number is positive (0) or negative (1). |
| exponent | The bits that tell how large a number is (the power of 2). |
| mantissa | The bits that tell the digits of a number. |
| bias | A fixed number that the format adds to the exponent, so that the exponent field is never negative. |

---

## 3. Hardware and software

| Item | Value |
|---|---|
| Main GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, Windows 11) |
| Software | Python 3.13, PyTorch 2.7.0 + CUDA 12.8 |
| Other hosts | Google Colab, AWS EC2 GPU, a CPU-only VPS (see section 7) |

`src/device.py` selects the device, the data type and the peak FLOPs. You do not change the code for a different host.

---

### 3.1 Share the GPU with the display

The local GPU also drives the display. If a script keeps the GPU 100% busy, the display can stop.

`src/device.py` sets two limits:
1. `GPU_DUTY` (default 0.6). After each step, the code waits for the GPU to finish. Then it sleeps for `step time × (1/GPU_DUTY − 1)`. Thus, the GPU is busy for 60% of the time or less.
2. `GPU_MEM_FRACTION` (default 0.6). The process can use 60% of the GPU memory or less.

Timing does not include the sleep time. Thus, MFU stays correct.

The tests hide the GPU (`CUDA_VISIBLE_DEVICES=""`). They run on the CPU only.

On a dedicated cloud GPU, set both values to 1.0.

## 4. Model and data

### 4.1 Model

Use Karpathy's nanoGPT (`src/model.py`, MIT licence). It is a small, well-known reference model.

The changes are small and have the label `CHANGED:` in the file:
1. A trace hook prints the shape of each internal tensor (part 1).
2. `forward(..., all_logits=True)` gives the logits for all positions without a loss (part 3).
3. `estimate_mfu` accepts the peak FLOPs of any GPU (part 5).

The model has these parts, in this order:
1. A token embedding (`wte`) and a position embedding (`wpe`).
2. L transformer blocks. Each block has LayerNorm, causal self-attention, LayerNorm and an MLP. Each sub-layer has a residual add.
3. A final LayerNorm and an output layer (`lm_head`). The output layer shares its weights with `wte`.

Dropout is 0. Thus, the forward pass gives the same result each time. Part 2 needs this. (V4 also set dropout to 0 for reversible models for the same reason.)

### 4.2 Two sizes

| Name | Layers | Heads | Channels (C) | Context (T) | Weights | Used in |
|---|---|---|---|---|---|---|
| S | 4 | 4 | 128 | 128 | about 0.8 M | parts 1–4 |
| shakespeare_char | 6 | 6 | 384 | 256 | about 10.7 M | part 5 |

`shakespeare_char` is the nanoGPT configuration. Part 5 needs larger matrix multiplications to give a useful MFU.

### 4.3 Data

Use Tiny Shakespeare (1.1 MB of text). One token is one character. There are 65 different characters (V = 65). Use the first 90% for training and the last 10% for validation.

---

## 5. Parts 1 to 6

### Part 1: Tensor shapes

**Purpose.** Know the shape of each tensor in one step. A wrong shape can give a wrong result without an error.

**Theory.**
- A step has three phases: forward, backward and update.
- The forward pass changes token ids `(B, T)` into logits `(B, T, V)`. Then it calculates one loss number.
- Attention divides the C channels into nh heads. Each head has hs = C / nh channels.
- The attention scores have the shape `(B, nh, T, T)`. Each row tells how much one position looks at each earlier position.
- `scaled_dot_product_attention` does not store the scores. The script reports their shape from B, nh and T.
- Each gradient has the same shape as its weight.

**Symbols.** B = sequences in the batch, T = positions, C = channels, nh = heads, hs = head size, V = vocabulary size, L = blocks.

**Procedure.**
1. Select B = 4 and T = 16. These values are different from C, nh and hs, so you can read each dimension.
2. Install the trace hook with `set_trace`.
3. Run one forward pass with targets.
4. Remove the trace hook.
5. Run `loss.backward()`.
6. Print each parameter shape and its gradient shape.
7. Run `optimizer.step()`. Print the shapes of the AdamW state tensors.

**Expected result.** `idx (4,16)`, `tok_emb (4,16,128)`, `q (4,4,16,32)`, `scores (4,4,16,16)`, `logits (4,16,65)`, `loss ()`.

**Pass check.** Each gradient shape is equal to its weight shape.

**Run.** `python -m src.p1_shapes`

---

### Part 2: Verify one gradient by hand

**Purpose.** Show that `backward()` gives the correct gradient.

**Theory.**
- A gradient tells how much the loss changes when one weight changes by a very small amount.
- You can measure it directly. Change the weight by +ε and by −ε. Calculate the loss each time.
- The central difference is `(L(w+ε) − L(w−ε)) / 2ε`. Its error decreases with ε².
- The one-sided difference `(L(w+ε) − L(w)) / ε` has an error that decreases only with ε.
- The worked example shows this. The one-sided difference with ε = 0.001 gives 64.064. The true value is 64.
- A very small ε has a different problem. The two losses become almost equal. Then rounding error controls the result.
- fp32 keeps about 7 decimal digits. Thus, a loss of 4.2 has a rounding step of about 2.5 × 10⁻⁷.
- If the gradient is 10⁻⁵ and ε is 10⁻³, the loss changes by only 2 × 10⁻⁸. fp32 cannot see this change.
- float64 keeps about 16 decimal digits. Thus, float64 is necessary for a precise check.

**Procedure.**
1. Calculate the worked example: h = w1·x, y = w2·h, loss = (y − t)², with w1 = 3, w2 = 4, x = 2, t = 20.
2. Make sure that autograd gives dL/dw1 = 64.
3. Make the model in float64. Set the model to eval mode.
4. Select the weight `transformer.h.0.attn.c_attn.weight[3, 7]` (a query weight in block 0).
5. Run the forward and backward pass. Record the autograd gradient.
6. For each ε from 10⁻¹ to 10⁻⁷, calculate the central difference.
7. Put the weight back to its original value after each pair of calculations.
8. Do steps 3 to 7 again in float32.
9. Do steps 3 to 7 again in float32 for the weight with the largest gradient.

**Expected result.** In float64, the best agreement has a relative error near 10⁻⁸. In float32, a small gradient does not agree. A large gradient agrees to about 3 digits.

**Pass check.** The float64 relative error is less than 10⁻⁷ (test `test_nudge_matches_autograd_in_float64`).

**Run.** `python -m src.p2_gradcheck`

**Reference.** `torch.autograd.gradcheck` uses the same method in float64.

---

### Part 3: Break gradient accumulation on purpose

**Purpose.** See the error that existed in the major training frameworks until October 2024.

**Theory.**
- The correct loss for a batch is the sum of the loss of all real tokens, divided by the number of real tokens.
- Each token then has the same weight.
- The broken method calculates the average loss of each micro-batch. Then it calculates the average of those averages.
- The broken method gives each micro-batch the same weight, so a short micro-batch gets too much weight.
- Worked example: three micro-batches have 4, 4 and 2 real tokens, with average losses 2.0, 2.0 and 5.0.
  - Correct: (4·2.0 + 4·2.0 + 2·5.0) / 10 = 2.6.
  - Broken: (2.0 + 2.0 + 5.0) / 3 = 3.0. This is 15.4% too high.
- If all micro-batches have the same number of tokens, the two methods are equal. That is why the error stayed hidden.
- The fix (Hugging Face Transformers, Unsloth, October 2024):
  - Use `cross_entropy(..., reduction="sum")` for each micro-batch.
  - Divide by the number of real tokens in the whole accumulation step (`num_items_in_batch`).
- Goyal et al. (2017) give the same rule for many GPUs: divide by the total batch size, not by the size on one GPU.

**Procedure.**
1. Make three micro-batches for each step. They have 16 sequences each, with 128, 128 and 8 real tokens per sequence.
2. Give padding positions the target −1. The loss ignores −1.
3. Make three models with the same seed and initial weights.
4. Give the three models the same micro-batches in the same order.
5. Model "full": join the three micro-batches. Calculate the sum of losses. Divide by all real tokens.
6. Model "correct": for each micro-batch, divide the sum of losses by all real tokens of the step. Call `backward()`.
7. Model "bug": for each micro-batch, divide the mean loss by 3. Call `backward()`.
8. Before training, compare the step-0 gradients of the three models in float32.
9. Do the comparison again with three equal micro-batches (the control).
10. Train each model for 400 steps with AdamW (learning rate 10⁻³, clip 1.0).
11. For all three models, log the training loss with the correct formula. Then the curves are comparable.
12. Calculate the validation loss every 20 steps on full 128-token windows.
13. Plot the training loss and the validation loss of the three models.

**Expected result.**
- The step-0 gradient of "correct" is equal to "full" to about 10⁻⁷.
- The gradient of "bug" is different by a large amount.
- With equal micro-batches, "bug" is equal to "full".
- The "bug" curve stays above the "correct" curve.

**Pass check.** Tests `test_correct_accumulation_equals_full_batch_and_bug_does_not` and `test_bug_hides_with_equal_lengths`.

**Run.** `python -m src.p3_accum_bug`

---

### Part 4: The grad norm moves before the loss

**Purpose.** Show that the grad norm warns about trouble before the loss does.

**Theory.**
- The grad norm is the length of the full gradient vector.
- The grad norm becomes large when the model reaches a steep region of the loss surface. Pascanu et al. (2013) call this a "wall".
- The loss of the current batch can still be normal at that step.
- The large gradient then causes a large weight change. Thus, the loss of the next steps increases.
- `clip_grad_norm_` gives the norm before clipping. Log this value, not the clipped value.
- A common clip limit is 1.0. Pascanu et al. tell you to select the limit from the norm statistics of many steps.

**Event rule.** A step s is an event if all three conditions are true:
1. The grad norm at s is more than 3 times its median over the previous 50 steps.
2. The loss at s is less than its median + 3 × its robust spread over the previous 50 steps.
3. The loss goes above that limit in one of the next 10 steps.

**Procedure.**
1. Train model S for 1500 steps. Use a learning rate of 3 × 10⁻³, a batch of 32 × 128 and a clip limit of 1.0.
2. At each step, record the loss and the grad norm before clipping.
3. Apply the event rule to all steps.
4. If one or more events occur, report the clearest event: the one with the largest norm ratio.
5. Plot the loss and the grad norm on one figure with two y-axes. Add a zoom view of the event.
6. If no event occurs, write this in the README. Do not invent an event.

**Fallback (only if step 6 occurs).**
1. Use a higher learning rate, for example 10⁻², and do the run again.
2. Write in the README that the learning rate is higher on purpose to cause instability.

**Run.** `python -m src.p4_gradnorm`

---

### Part 5: MFU

**Purpose.** Measure the part of the GPU that does useful work. The loss curve cannot show this.

**Theory.**
- A forward and backward pass needs about 6N FLOPs for each token. N is the number of weights.
  - The forward pass needs 2N: one multiply and one add for each weight.
  - The backward pass needs 4N: one gradient for the activations and one for the weights.
- Attention adds 12·L·H·Q·T FLOPs for each token (PaLM, Appendix B). L = layers, H = heads, Q = head size, T = context.
- N does not include the position embeddings (nanoGPT `get_num_params`).
- MFU = (6N + 12LHQT) × tokens per second / peak FLOPs per second.
- MFU does not count recomputation. HFU (hardware FLOPs utilisation) counts it. MFU is better for comparisons (PaLM, section 4.1).
- PaLM 540B reached 46.2% MFU. A healthy large run is between 35% and 50%.
- The peak value for the RTX 5070 Ti (87.9 TFLOP/s bf16 dense) comes from a data aggregator. NVIDIA's whitepaper does not confirm it here.
- Thus, the script also measures a ceiling: the FLOPs per second of one large 8192 × 8192 bf16 matrix multiplication.

**Procedure.**
1. Make the shakespeare_char model on the GPU. Use bf16 autocast and fused AdamW.
2. Make 8 batches of 64 × 256 tokens before you start the timer.
3. Run 10 steps to warm up.
4. Synchronise the GPU. Start the timer.
5. Run 50 steps. Synchronise the GPU. Stop the timer.
6. Calculate tokens per second and MFU.
7. Calculate MFU without the attention term.
8. Calculate MFU with nanoGPT `estimate_mfu` and the same peak.
9. Measure the matmul ceiling.
10. Profile 5 steps with `torch.profiler`. Group the GPU time by kernel type.
11. Calculate the GPU busy time as a fraction of the wall time.
12. Try `torch.compile`. If it fails, record the error message.

**How to explain the distance to 40%.** Use the profile, not a guess. Possible causes:
- **Small matrices.** C = 384 is small. The tensor cores are not full.
- **Launch overhead.** Python starts hundreds of small kernels in each step. The GPU waits between kernels.
- **Memory-bound operations.** LayerNorm, GELU, softmax, cross-entropy and AdamW read and write memory but do few FLOPs.
- **Windows WDDM.** The Windows display driver model adds latency to each kernel launch.
- **Gradient clipping.** It calculates a norm over all gradients in each step.

**Run.** `python -m src.p5_mfu`

---

### Part 6: The number 0.1 in fp32, bf16 and fp8 E4M3

**Purpose.** Understand what each format keeps and what it loses.

**Theory.**
- A floating-point number = (−1)^sign × 1.mantissa × 2^(exponent − bias).
- Exponent bits give range. Mantissa bits give precision. More of one means less of the other.

| Format | Exponent bits | Mantissa bits | Bias | Smallest normal | Decimal digits |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | 127 | 1.18 × 10⁻³⁸ | 7.2 |
| bf16 | 8 | 7 | 127 | 1.18 × 10⁻³⁸ | 2.4 |
| fp8 E4M3 | 4 | 3 | 7 | 2⁻⁶ = 0.0156 | 1.2 |

- E4M3 has no infinity. S.1111.111 is NaN. The largest value is 448 (Micikevicius et al., 2022).

**Procedure (by hand).**
1. Write 0.1 as 1.6 × 2⁻⁴, because 0.1 × 16 = 1.6.
2. Calculate the sign: 0.1 is positive, so the sign is 0.
3. Write 0.6 in binary. Multiply by 2 again and again. Record the integer part each time.
   - 0.6 → 1.2 (1), 0.2 → 0.4 (0), 0.4 → 0.8 (0), 0.8 → 1.6 (1), 0.6 → … The pattern 1001 repeats.
   - Thus, 1.6 = 1.1001 1001 1001 … in binary. No binary format can keep 0.1 exactly.
4. Calculate the exponent field: −4 + bias.
5. Keep the first M mantissa bits. Round to nearest. If the dropped part is exactly one half, round to the even value.

**Results.**

| Format | Sign | Exponent field | Mantissa | Hex | Stored value | Relative error |
|---|---|---|---|---|---|---|
| fp32 | 0 | 01111011 (−4 + 127 = 123) | 10011001100110011001101 | 0x3DCCCCCD | 0.100000001490116 | 1.5 × 10⁻⁸ |
| bf16 | 0 | 01111011 (123) | 1001101 | 0x3DCD | 0.10009765625 | 9.8 × 10⁻⁴ |
| fp8 E4M3 | 0 | 0011 (−4 + 7 = 3) | 101 | 0x1D | 0.1015625 | 1.6 × 10⁻² |

Rounding details:
- **fp32:** the 23 bits are `10011001100110011001100`. The next bits are `1100…`, which is more than half. Round up to `…1101`.
- **bf16:** the 7 bits are `1001100`. The next bits are `1100…`, which is more than half. Round up to `1001101`.
- **E4M3:** 0.6 × 8 = 4.8. The 3 bits are `100` (4). The remainder 0.8 is more than half. Round up to `101` (5). The value is 1.625 × 2⁻⁴.

**Decision: which format to train in.**
1. Do the matrix multiplications in bf16. Accumulate the products in fp32.
2. Keep an fp32 master copy of the weights and the fp32 optimiser state (16 bytes for each weight).
3. Do not use fp16 without loss scaling. Its smallest normal value is 6.1 × 10⁻⁵, so small gradients become zero (Micikevicius et al., 2017).
4. bf16 has the same range as fp32. Thus, bf16 needs no loss scaling.
5. Use fp8 E4M3 only for the inputs of matrix multiplications. Use a scale for each tensor or block.
6. Use fp8 only on hardware that supports it (Hopper, Blackwell).
7. Do not use fp8 for the optimiser state, the master weights or the accumulators. Their error builds up over steps.

The reason, in one sentence: range is more important than precision in training, and the error must not build up over steps.

**Pass check.** Test `test_bits_of_point_one`. torch gives the same bits.

**Run.** `python -m src.p6_bits`

---

## 6. Time plan (14 October 2026, 6 hours)

| Time | Work |
|---|---|
| 0.5 h | Repository, model, data, device selection |
| 1.0 h | Parts 1 and 6 |
| 1.0 h | Part 2 |
| 1.0 h | Part 3 |
| 1.0 h | Parts 4 and 5 |
| 1.0 h | Notebook, README, tests, push |

---

## 7. Run in the cloud

The code is the same on each host. `src/device.py` selects:
- the device: CUDA, then Apple MPS, then CPU;
- the data type: bf16, or fp16 with a GradScaler on GPUs without bf16;
- the peak FLOPs for MFU, from a table of data-sheet values.

To use a different peak value, set the environment variable `PEAK_TFLOPS`.

| GPU | Where | Peak used for MFU | Note |
|---|---|---|---|
| RTX 5070 Ti | local | 87.9 TFLOP/s bf16 | Aggregator value. Not confirmed by NVIDIA. |
| T4 | Colab (free) | 65 TFLOP/s fp16 | No bf16 tensor cores. The code uses fp16 with loss scaling. |
| L4 | Colab Pro, AWS g6 | 121 TFLOP/s bf16 | |
| A10G | AWS g5 | about 70 TFLOP/s bf16 | Check the AWS or NVIDIA data sheet. |
| A100 | Colab Pro | 312 TFLOP/s bf16 | nanoGPT default. |
| CPU | VPS (for example Hostinger) | none | Part 5 reports tokens per second only. MFU is "n/a". |

### 7.1 Google Colab

1. Open the notebook with the "Open in Colab" badge in `README.md`.
2. Select Runtime → Change runtime type → GPU.
3. Run the first cell. It clones the repository and installs the requirements.
4. Run all cells.
5. Save the notebook with its outputs: File → Save a copy in GitHub. Alternatively, download the `.ipynb` file and commit it.

### 7.2 AWS EC2 GPU

1. Open Service Quotas. Find "Running On-Demand G and VT instances".
2. If the quota is 0, request 4 vCPUs or more. Wait for approval.
3. Launch `g6.xlarge` (L4, 24 GB) or `g5.xlarge` (A10G).
4. Select the AMI "Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu)".
5. In the security group, allow SSH (port 22) from your IP address only.
6. Connect: `ssh -i key.pem ubuntu@<ip>`.
7. Activate the PyTorch environment that the login message shows.
8. Run these commands:
   ```
   git clone https://github.com/jssunil/training-loop-diagnostics && cd training-loop-diagnostics
   pip install -r requirements.txt
   pytest -q
   python -m src.p5_mfu
   jupyter nbconvert --to notebook --execute --inplace notebooks/training_loop_diagnostics.ipynb
   ```
9. Copy the results to your computer: `scp -r ubuntu@<ip>:training-loop-diagnostics/results ./results_aws`.
10. **Stop or terminate the instance.** AWS charges for each hour that the instance runs.

### 7.3 Hostinger (CPU-only VPS)

1. Make sure that your plan has a GPU. Hostinger VPS plans are usually CPU-only.
2. Install the tools:
   ```
   sudo apt install -y python3-venv git
   python3 -m venv .venv && . .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Run parts 1, 2, 3, 4 and 6. They use model S and run on a CPU.
4. Run part 5 for tokens per second only. MFU on a CPU against a GPU peak has no meaning.
5. Use the VPS to show that the code runs on a clean computer. Do not use it for GPU numbers.

### 7.4 Local computer (main path)

1. Open a terminal in the repository folder.
2. Run `python -m src.p1_shapes`, and then parts 2 to 6.
3. Run `pytest -q`.
4. Run the notebook from the first cell to the last cell.

The README numbers come from this computer, unless the README tells you a different host.

---

## 8. Risks

| Risk | Effect | Action |
|---|---|---|
| Triton does not work on Windows | `torch.compile` fails | Record the error in the README. Report eager MFU. |
| The peak FLOPs value is not confirmed | MFU can be wrong | Also report MFU against the measured matmul ceiling. |
| No natural grad-norm event | Part 4 has no example | Use a higher learning rate. Write in the README that this is on purpose. |
| A Colab T4 has no bf16 | Different data type | The code uses fp16 with a GradScaler. The README names the GPU for each number. |
| The AWS quota is 0 | No GPU instance | Request the quota before the day you need it. |

---

## 9. References

1. Course notes on the training loop (the worked examples in parts 2 and 3).
2. P. Micikevicius et al., *Mixed Precision Training*, 2017. arXiv:1710.03740. Section 3: fp32 master weights, loss scaling, fp32 accumulation.
3. P. Micikevicius et al., *FP8 Formats for Deep Learning*, 2022. arXiv:2209.05433. The E4M3 and E5M2 formats.
4. P. Goyal et al., *Accurate, Large Minibatch SGD*, 2017. arXiv:1706.02677. Sections 2–3: linear scaling, warm-up, loss normalisation.
5. R. Pascanu, T. Mikolov, Y. Bengio, *On the difficulty of training recurrent neural networks*, 2013. arXiv:1211.5063. Gradient norm clipping.
6. A. Chowdhery et al., *PaLM*, 2022. arXiv:2204.02311. Section 4.1 and Appendix B: MFU.
7. Hugging Face, *Fixing Gradient Accumulation*, October 2024. https://huggingface.co/blog/gradient_accumulation
8. Unsloth, *Bug Fixes in LLM Training: Gradient Accumulation*, October 2024. https://unsloth.ai/blog/gradient
9. A. Karpathy, nanoGPT. https://github.com/karpathy/nanoGPT (`model.py`, `estimate_mfu`).
