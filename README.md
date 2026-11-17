# Multilingual Text Continuation from Rendered Images

Kaggle course project for **CSC14005 (Introduction to Machine Learning)**, HCMUS.

The task is to predict the missing continuation of a paragraph in five languages, under one unusual rule: the model may not read the context as text, and its answer may not be submitted as text. The context has to be turned into an image, the continuation is predicted from that image, and the result is submitted as pixel coordinates of a rendered text image.

Final system: a fine-tuned **TrOCR** (`microsoft/trocr-base-printed`) that reads a rendered image of the end of the context and generates the continuation, which is then rendered back into a white-on-black text image and exported as `pixels.npz`.

## Contents

1. [Problem](#1-problem)
2. [Pipeline](#2-pipeline)
3. [Data and EDA](#3-data-and-eda)
4. [Text-to-image representation](#4-text-to-image-representation)
5. [Model and training](#5-model-and-training)
6. [Experiments](#6-experiments)
7. [Results](#7-results)
8. [Error analysis and limitations](#8-error-analysis-and-limitations)
9. [Repository structure](#9-repository-structure)
10. [How to run](#10-how-to-run)
11. [References](#11-references)

## 1. Problem

- **Input:** a `context` paragraph (about the first 80% of a text) in English, French, German, Italian or Spanish.
- **Output:** the missing continuation (`target`, about the last 20%).
- **Constraint:** the context must go through a non-text representation before it reaches the model, and the final answer is a set of foreground pixel coordinates, not a string.
- **Submission format:** `pixels.npz` (a structured array `pixels` with fields `sample_id`, `row_id`, `col_id`) plus `submission.csv` (`id`, `sample_id`).

## 2. Pipeline

```text
context text
   │  keep the last 160 characters
   ▼
render as white text on a black 384×384 canvas (bottom-anchored)
   │
   ▼
TrOCR  (ViT encoder frozen, top 4 decoder layers fine-tuned)
   │  decoder starts from a language prompt, e.g. "French: "
   ▼
continuation text
   │  clean prediction, render as a text image (white on black)
   ▼
foreground pixel coordinates (sample_id, row_id, col_id)  →  pixels.npz
```

## 3. Data and EDA

`train-3.csv` has 1,450 labelled rows with the columns `id`, `language`, `context`, `target`, `max_width`. The test set has 364 samples (`test.csv` is not included in this repository).

| Language | Train rows | Test samples |
|---|---:|---:|
| English | 725 | 182 |
| French | 163 | 50 |
| German | 191 | 46 |
| Italian | 191 | 38 |
| Spanish | 180 | 48 |

Findings from `EDA.ipynb`:

- **Lengths:** contexts average 211 characters (66 to 554); targets average 57 characters (16 to 156). The target is about 21% of the full text on average.
- **Split boundaries:** the context and target are cut in the middle of a word in 86.2% of samples, so the model has to handle a half-written word at the end of the context.
- **Leakage:** the start of the target does not appear in the context in any of the 1,450 samples.
- **Render width:** `max_width` ranges from 155 to 1586 pixels (mean 559). Many widths in the test set never appear in training, so the model cannot rely on memorised widths.
- **Rendered targets:** every target image is a single line of height 46 pixels. About 5.2% of pixels are foreground, which is about 1,364 white pixels per sample on average.
- **Determinism:** rendering the same text twice gives identical pixels, so the renderer is safe to use in training and inference.
- **Language imbalance:** English makes up about half of the training data. This matters later (see section 8).

## 4. Text-to-image representation

| Parameter | Value |
|---|---|
| Text kept from the context | last 160 characters |
| Image size | 384 × 384 (fixed canvas) |
| Font size | 18 |
| Padding | 12 |
| Colours | white text on black |
| Anchor | bottom |

An earlier version shrank the rendered image to fit a square. That made the text too small to read and was the first real error found in the project. The final version renders straight into a fixed canvas and never shrinks the text.

## 5. Model and training

| Item | Value |
|---|---|
| Pretrained model | `microsoft/trocr-base-printed` (`VisionEncoderDecoderModel`) |
| Encoder | frozen |
| Decoder | only the top 4 layers (layers 8 to 11) are fine-tuned |
| Trainable parameters | 117,089,280 of 333,921,792 (35.06%) |
| Optimiser | AdamW, learning rate 1e-5, weight decay 0.01 |
| Schedule | linear warm-up (10%) then cosine annealing |
| Batch size | 4, with gradient accumulation 2 |
| Epochs | up to 10, early stopping with patience 4 |
| Precision | bf16, gradient clipping 1.0 |
| Max target length | 128 tokens |

**Loss.** A weighted cross-entropy with three parts:

- **Prompt tokens:** the decoder starts from a short language prompt (for example `French: `). The prompt tokens are kept in the labels with weight 1.0 instead of being masked ("prompt supervision").
- **Prefix:** the first 12 continuation tokens get weight 3.0, to pull the model towards the true start of the continuation.
- **Tail:** the rest uses ordinary cross-entropy.

**Decoding.** `num_beams=2`, `max_new_tokens=72`, `min_new_tokens=8`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, `length_penalty=0.8`.

## 6. Experiments

Everything below was tried; only the rows marked "kept" are in the final system. The full log is in `PIPELINE_NOTES.md`.

| Variant | Idea | Outcome |
|---|---|---|
| VLM + LoRA | multimodal assistant-style continuation | Dropped. Output started with `Assistant:` and the model learned the wrong format. |
| TrOCR baseline | image of context → TrOCR → continuation | Works, but drifts to generic English. CER about 0.78 to 0.79; Spanish was the worst language. |
| TrOCR encoder + mT5-small decoder | encoder joined to `google/mt5-small` through a projection layer, two-phase training | An earlier version; its configuration is kept in `filelog/v1.py`. |
| Fixed-canvas representation | stop shrinking the text | **Kept.** Clear improvement. |
| Decoder frozen except grounding | freeze almost everything | Dropped. OCR junk such as `.com/survey`. |
| Top-4 decoder layers | fine-tune the top 4 layers only | **Kept.** Most stable setting. |
| Prefix-weighted loss | weight the first 12 tokens by 3 | **Kept.** `prefixCER` drops clearly. |
| Balanced language sampling | `WeightedRandomSampler` by language | Dropped. Less English bias, but CER about 0.800 and multilingual gibberish. |
| Stronger prompt wording | `Continue in Spanish:` | Dropped. Unstable and worse after 1 to 2 epochs. |
| Prompt supervision | keep prompt tokens in the labels | **Kept.** Best change at the end. |
| Candidate re-ranking | re-rank outputs by language consistency | Dropped. CER about 0.810, worse than the baseline. |

## 7. Results

Validation results (lower CER is better):

| Setting | CER |
|---|---:|
| Baseline before prompt supervision | about 0.785 |
| Balanced sampling | about 0.800 |
| Re-ranking | about 0.810 |
| **Prompt supervision + prefix weighting (final)** | **about 0.767** |

For the final checkpoint (best at epoch 2): CER 0.767, pixel F1 0.258, prefix CER 0.893.

On the test set the system produced non-empty output for all 364 samples: 307,785 foreground pixels in total (about 846 per sample) and 36.4 predicted characters on average (14 to 69). The public Kaggle score recorded for the final submission is **1.56**; see the competition page for how the score is defined.

Metrics tracked during development: `CER`, `prefixCER`, `pixelF1`, `repeat`, `lenRatio`, `short%`, `long%`, `unique`, `langCons` (does the prediction use words from the expected language) and `engOpen` (share of predictions opening with an English word such as `and`, `the`, `to`).

## 8. Error analysis and limitations

Failure patterns seen in the diagnostics:

- **English opener collapse:** predictions start with `and ...` even for non-English samples.
- **OCR junk:** fragments such as `.com/survey` or `FACE UP ON DETAILS`.
- **Multilingual gibberish:** mixed fragments from several languages.
- **Generic continuations:** plausible text that has nothing to do with the true continuation.

Limitations:

- The decoder still leans on a generic language prior, so it often opens with a generic English word.
- Language control is incomplete, and Spanish is the hardest language.
- A better `prefixCER` does not always lower the full-sequence CER, which tends to plateau.
- CER of about 0.77 means the model is far from reproducing the true continuation; the improvements are real but modest.

Ideas not tried for lack of time: a redesigned objective for the first continuation tokens, decoding that is explicitly constrained to the target language, and an architecture with a weaker language-model prior and stronger image-text grounding.

## 9. Repository structure

```text
├── PIPELINE_NOTES.md       Full development notes (Vietnamese): pipeline, experiments, metrics, errors
├── EDA.ipynb               Data analysis and design of the text-to-image renderer
├── trocr_pipeline.ipynb    Full pipeline: config, rendering, dataset, model, metrics, training, inference
├── train-3.csv             Labelled training data (1,450 rows)
├── data/pixels.npz         Final submission: foreground pixel coordinates
├── submission.csv          Submission index (id, sample_id)
├── v1.zip                  An earlier submission (pixels.npz and submission.csv)
└── filelog/v1.py           Configuration of the earlier TrOCR + mT5-small variant
```

`trocr_pipeline.ipynb` is organised in the same order as the pipeline: setup and imports, configuration, utilities, dataset and data loader, model, metrics, training loop, and inference with submission export.

## 10. How to run

Requirements: Python with `torch`, `transformers`, `accelerate`, `pillow`, `pandas` and `numpy`. The training configuration uses bf16, so a GPU with bf16 support is expected.

1. Put `train-3.csv` and the competition's `test.csv` next to the notebook (paths are set in `get_config()`). `test.csv` is not part of this repository.
2. Open `trocr_pipeline.ipynb`. The first code cell contains the `pip install` line, commented out for Kaggle and Colab.
3. Training: in the "Chạy training" cell, uncomment `history = train(config)` and run it. Checkpoints go to `outputs/`, and debug images and sample texts for each epoch go to `debug_outputs/`.
4. Inference: run `generate_submission()` in the last code cell. It loads `outputs/best_model` and writes the pixel-coordinate submission.

The renderer looks for a serif font in several locations (Windows and Linux paths), so the notebook runs on Windows, Linux and Kaggle. Rendering is deterministic, so the same text always gives the same pixels. The random seed is 42.

## 11. References

1. M. Li, T. Lv, J. Cui, Y. Lu, D. Florencio, C. Zhang, and F. Wei, "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 11, pp. 13094-13102, 2023.
2. Hugging Face, "Transformers Documentation." https://huggingface.co/docs/transformers/
3. Microsoft, "microsoft/trocr-base-printed," Hugging Face Model Hub. https://huggingface.co/microsoft/trocr-base-printed
4. Hugging Face, "Vision Encoder Decoder Models." https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder
