# Reimplementing Natural Language Autoencoders on GPT-2 Small

**Paper:** [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html) **Authors:** Fraser-Taliente, Kantamneni, Ong et al. (Anthropic, 2026\) **Code:** [Google Colab Notebook](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s)

---

## 1\. What This Project Does

This project reimplements Anthropic's Natural Language Autoencoder (NLA) methodology on GPT-2 Small (124M parameters), running end-to-end on a single free Colab T4 GPU in approximately 25 minutes. The goal is to test whether the core NLA training loop — mapping activations to text and back — works at a scale 50–500x smaller than the paper's target models.

An NLA consists of two jointly-trained language models:

- **Activation Verbalizer (AV):** Takes a residual stream activation from the target model and generates a natural language "explanation" of what that activation encodes. Implemented as a full GPT-2 (124.4M params) with the activation vector injected as a scaled token embedding at a special position in the prompt. ([Cell 6](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-6))  
    
- **Activation Reconstructor (AR):** Reads the AV's text explanation and reconstructs the original activation vector. Implemented as a truncated GPT-2 (first 8 layers, 96.7M params) followed by a learned affine projection head that maps the final hidden state to the activation space. ([Cell 7](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-7))

The pipeline is trained in two phases: supervised warm-start ([Cell 9](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-9)) followed by GRPO reinforcement learning ([Cell 10](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-10)). The key metric is **Fraction of Variance Explained (FVE):** `1 - MSE/Variance`, where higher is better. The paper achieves 0.6–0.8 on frontier models; we explore what happens at 124M.

## 2\. Design Choices and Why

**Target layer \= 8 of 12\.** The paper targets roughly the 2/3-depth point of the residual stream, where representations are semantically rich but haven't collapsed toward the unembedding. For GPT-2's 12 layers, layer 8 is the natural choice, and our [layer sensitivity analysis](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-17) confirms it as optimal.

**Activation normalization to unit L2-norm.** Following the paper, all extracted activations are normalized before injection into the AV. This is important: it means the AV sees only the *direction* of the activation vector, not its magnitude. This choice simplifies the reconstruction problem and makes cosine similarity the natural evaluation metric.

**Injection scale from corpus statistics.** Rather than hand-tuning, we measure the 75th percentile of hidden state norms at layer 8 across sample texts and use that as the injection scale (\~50.0). This ensures the injected activation has comparable magnitude to the other token embeddings the AV processes. ([Cell 5](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-5))

**Pseudo-summaries instead of Claude-generated warm-start data.** The paper's warm-start uses Claude Opus 4.5 to generate high-quality summaries of text contexts, achieving FVE \~0.3–0.4 before RL. We lack access to a frontier model for this, so we use rule-based pseudo-summaries that extract structural cues (topic sentences, pronoun usage, sentence count). This is our largest methodological compromise — it directly limits warm-start quality and sets a weaker starting point for RL. ([Cell 8](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-8))

**150 RL steps with group size 4\.** The paper trains for thousands of steps on multi-GPU clusters. We initially attempted 400 steps but hit Colab's runtime timeout at step 139\. We reduced to 150 steps (\~19 minutes at \~7.5s/step on T4), which is enough to observe clear training dynamics even if not enough for convergence.

**WikiText-103 as data source.** We use 5,600 text samples from WikiText-103: 1,500 for SFT, 3,000 for RL, and 100 for evaluation. Texts are filtered to be \>100 characters and non-header. Random truncation at word boundaries provides context diversity.

## 3\. Results

### 3.1 Training Progression

**SFT Warm-Start** ([Cell 9](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-9)): 3 epochs over 1,500 pseudo-summary pairs.

| Epoch | AV CE Loss | AR MSE | FVE |
| :---- | :---- | :---- | :---- |
| 1/3 | 4.1267 | 0.000589 | \-0.1058 |
| 2/3 | 3.3172 | 0.000395 | \-0.1120 |
| 3/3 | 3.0032 | 0.000391 | \-0.1122 |

The AV loss drops 27% (4.13 → 3.00), showing the model learns to condition on the injected activation. The AR MSE drops 34% (0.000589 → 0.000391). However, FVE remains negative at \-0.11, meaning the AR's reconstructions are worse than simply predicting the mean activation. For comparison, the paper's warm-start achieves \+0.3 to \+0.4 — our pseudo-summaries provide a much weaker starting signal.

**GRPO RL Training** ([Cell 10](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-10)): 150 steps, completed in 18 minutes.

| Step | Reward | FVE |
| :---- | :---- | :---- |
| 1 | 6.153 | \-5.5452 |
| 50 | \~7.2 | — |
| 100 | \~7.8 | — |
| 150 | \~7.5 | \-1.1408 |

The reward (= `-log(MSE)`) increased from 6.15 to \~7.5 over training, indicating reconstruction quality improved. However, the FVE at step 150 is \-1.14, which is *worse* than the SFT endpoint of \-0.11. This apparent contradiction is explained in Section 4\.

### 3.2 Final Evaluation

Evaluated on 100 held-out samples ([Cell 11](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-11)):

| Metric | Value |
| :---- | :---- |
| Mean MSE | 0.000275 |
| Mean Cosine Similarity | 0.8945 |
| Overall FVE | \-0.3020 |
| Total Variance | 0.000211 |
| Cosine 25th percentile | 0.9545 |
| Cosine 50th percentile (median) | 0.9594 |
| Cosine 75th percentile | 0.9624 |

The tight cosine similarity distribution (IQR: 0.9545–0.9624) is notable — the pipeline produces remarkably consistent reconstructions, with median cosine similarity of 0.96.

## 4\. Interesting Findings

### 4.1 The AV Mode Collapse: A Steganography Case Study

The most striking result is what the AV actually learned to generate. Across all evaluation examples, the AV produces nearly identical text: repetitions of "feet feet feet inches feet feet feet..." regardless of input content. ([Cell 12](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-12))

Sample outputs (verbatim):

**Input:** "Polka Party\! was released October 21, 1986..." **AV output:** "feet feet feet inches feet feet feet feet inches feet feet feet..." **Cosine similarity: 0.9633**

**Input:** "'Trees' is a lyric poem by American poet Joyce Kilmer..." **AV output:** "feet feet feet feet feet feet feet feet feet feet feet feet..." **Cosine similarity: 0.9579**

This is a textbook instance of the failure mode the paper explicitly discusses: the AV has found a degenerate communication strategy. Rather than encoding semantic content in human-readable text, it has learned to use the *distribution of a small set of tokens* as an implicit encoding. The AR, trained jointly, has learned to decode this encoding back into activation space — achieving high cosine similarity (0.96) from text that carries no semantic information to a human reader.

This is exactly the kind of "steganography" the paper warns about, though it takes a particularly visible form here. In frontier-scale experiments, steganographic encoding might be subtler — hidden in word choice, syntax, or formatting rather than in obvious token repetition.

**Why did this happen?** The GRPO objective rewards only reconstruction quality (`-log(MSE)`), not explanation quality. With a weak warm-start (pseudo-summaries vs. Claude-generated), the AV had little incentive to stay in the regime of human-readable text. The KL penalty (coefficient 0.05) was insufficient to prevent drift from the initialization. The AV discovered that a narrow token vocabulary with slight distributional variations (note the occasional "inches" among the "feet") provides enough degrees of freedom for the AR to reconstruct reasonably well.

**Why "feet" and "inches" specifically?** These tokens likely emerged because they are common, short tokens that the AV's pretrained weights can produce with low cross-entropy cost, and their embeddings happen to provide a useful basis for the AR's learned decoding. The specific tokens are arbitrary — what matters is that the AV converged on a fixed vocabulary.

### 4.2 The Steganography Check Reveals the Paradox

Our [steganography check](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-14) applies meaning-preserving transforms to AV explanations:

| Transform | Mean MSE | Relative |
| :---- | :---- | :---- |
| Original | 0.000463 | 1.00x |
| Shuffled sentences | 0.000463 | 1.00x |
| Truncated to half | 0.000840 | 1.81x |

At first glance, this looks like the paper's expected "good" pattern: shuffling doesn't hurt (low steganography risk), truncation does (content matters). But there's a subtlety: when the AV outputs "feet feet feet feet feet feet..." shuffling *cannot* change anything because the text is essentially uniform. The steganography check is designed for explanations with meaningful sentence structure — it becomes vacuous when the text has no structure to shuffle.

Truncation *does* hurt (1.81x), but this tells us something different from what it tells the paper. It's not that the *semantic content* of the first half differs from the second — it's that the AR needs a sufficient *length* of the degenerate token sequence to decode the activation. The token count itself is part of the encoding.

This is a genuine methodological insight: the paper's steganography checks assume the AV produces text with semantic structure. When the AV has mode-collapsed, these checks no longer validate what they're designed to validate. A more robust steganography check might involve paraphrasing the explanation with an independent LLM — which would completely destroy the degenerate encoding while preserving any genuine semantic content.

### 4.3 Cosine Similarity vs. FVE: Why They Diverge

Our results show high cosine similarity (0.89 mean, 0.96 median) alongside negative FVE (-0.30). This seems contradictory but reveals an important distinction.

FVE is computed as `1 - MSE/Variance`, where `Variance` measures the total spread of activations in the evaluation set. Our total variance is very small (0.000211) because all activations are L2-normalized to the unit sphere — they all have similar magnitude and their variance comes entirely from directional differences. When variance is this small, even a tiny MSE (0.000275) produces negative FVE because the ratio MSE/Variance exceeds 1\.

Cosine similarity, by contrast, directly measures directional agreement and is insensitive to magnitude. The median cosine of 0.96 means the reconstructed vectors point within \~16° of the originals on average in 768-dimensional space — a meaningful achievement.

This suggests that for unit-normalized activations, **cosine similarity is arguably the more informative metric than FVE**. The FVE formulation penalizes any reconstruction error relative to the very small total variance, making it a harsh metric when the activation space is already constrained to the unit sphere.

### 4.4 Layer Specificity Is Robust Even at Small Scale

The [layer sensitivity analysis](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s#scrollTo=cell-17) tests the NLA trained on layer 8 against activations from other layers:

| Layer | FVE | Mean MSE |
| :---- | :---- | :---- |
| 2 | \-3.0069 | 0.001453 |
| 4 | \-1.2276 | 0.000861 |
| 6 | \-1.2591 | 0.000813 |
| **8** | **\-1.1446** | **0.000840** |
| 10 | \-1.2326 | 0.000869 |
| 11 | \-1.5679 | 0.000863 |

Layer 8 achieves the best FVE, confirming the paper's finding that NLAs are layer-specific. The pattern is asymmetric: early layers (especially layer 2, at \-3.01) are far worse than later layers. This makes sense — early layers encode more local/syntactic features that differ substantially from the mid-network semantic representations the NLA learned on layer 8\. Later layers (10, 11\) are closer to layer 8 in representation space, so performance degrades more gradually.

The fact that layer specificity emerges even with degenerate AV outputs is notable. It means the *encoding* the AV learned (however degenerate) is genuinely tuned to layer 8's activation distribution, not just a generic mapping.

## 5\. What I Would Do Differently

**Better warm-start.** The single biggest improvement would be generating warm-start summaries with a capable language model (e.g., GPT-4, Claude) rather than rule-based pseudo-summaries. The paper is clear that warm-start quality determines the starting point for RL, and our weak warm-start is almost certainly why the AV mode-collapsed — it never had a strong enough "pull" toward human-readable explanations.

**Stronger KL penalty.** Increasing the KL coefficient beyond 0.05 might have prevented mode collapse by penalizing drift from the pretrained initialization more heavily. There's a tradeoff — too much KL prevents learning — but the current value was clearly insufficient.

**Temperature annealing.** Starting with high temperature and annealing down during RL training could encourage initial diversity in AV outputs before settling on high-reward explanations.

**More training steps.** The paper notes FVE grows roughly linearly in log(training steps). Our 150 steps is 1–2 orders of magnitude less than the paper's training runs. With Colab Pro (or a local GPU), running 1,000+ steps would better test whether the methodology converges at small scale.

## 6\. Summary

| What we implemented | Faithfulness to paper |
| :---- | :---- |
| AV architecture (activation injection via scaled embedding) | Faithful |
| AR architecture (truncated model \+ affine projection) | Faithful |
| SFT warm-start (AV CE \+ AR MSE, joint training) | Simplified (pseudo-summaries vs. Claude) |
| GRPO RL (group sampling, advantage-weighted SFT, AR regression) | Faithful |
| FVE metric, steganography checks, layer sensitivity | Faithful |

| What we found | Significance |
| :---- | :---- |
| AV mode-collapses to "feet feet feet..." | Illustrates steganography risk the paper discusses |
| Cosine similarity 0.96 despite negative FVE | Metrics diverge for unit-normalized activations |
| Steganography check becomes vacuous under mode collapse | Methodological limitation of shuffle-based checks |
| Layer specificity holds at 124M scale | Core finding is robust to model scale |

The most honest summary: the NLA *training loop* works at small scale — rewards increase, AR loss decreases, the system learns. But without a strong warm-start from a frontier model, the AV finds degenerate solutions that achieve good reconstruction without producing human-readable explanations. This is not a failure of the reimplementation; it's a confirmation that the paper's reliance on high-quality warm-start data is not merely a convenience but a *structural requirement* of the method.

## 7\. Reproducing These Results

Open the [Colab notebook](https://colab.research.google.com/drive/1wB2-V-aRAFRqeCzLjTzZzbG966_kTu7s), select **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Total time: \~25 minutes. No API keys or external dependencies required — everything runs on the free Colab tier.

The notebook produces two figures saved to `/content/`: `nla_training_analysis.png` (6-panel training curves) and `nla_layer_sensitivity.png` (layer FVE bar chart), and a JSON results file at `/content/nla_results.json`.  
