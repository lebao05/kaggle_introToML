# Pipeline Notes

## 0. Report Checklist

Bao cao PDF nen co cac muc sau:
- Student information
- Problem statement
- Proposed solution
- Data preprocessing
- Text-to-image representation
- Modeling approach
- Training setup
- Evaluation results
- Error analysis / failure cases
- Strengths and limitations
- Bibliography

## 0.1 Student Information

Can dien:
- Full name:
- Student ID:
- Course:
- Project title:

## 0.2 Problem Statement

Co the viet ngan gon nhu sau:

"The task requires predicting the missing continuation of a paragraph, but the prediction cannot be submitted as raw text. Instead, the context must first be transformed into a non-text representation, the continuation must be inferred from that representation, and the final output must be rendered as a text image and exported in pixel-coordinate format."

## 0.3 Final Proposed Solution

Mo ta ngan gon:

"We use a rendered-image representation of the tail of the context paragraph, then train a TrOCR-based encoder-decoder model to generate the continuation text. The predicted continuation is rendered back into a white-on-black text image and converted into pixel coordinates for submission."

## 1. Muc tieu bai toan

Bai toan yeu cau:
- nhan `context` text
- bien doi `context` sang non-text representation
- suy luan continuation
- render continuation thanh anh text
- xuat ket qua duoi dang pixel coordinates

Pipeline cuoi cung duoc giu lai:

`context text -> render tail context image -> TrOCR -> continuation text -> render -> pixels`

## 2. Pipeline cuoi cung duoc su dung

### Representation
- Khong dua raw text truc tiep vao model.
- Cat phan duoi cua `context` voi `context_tail_chars=160`.
- Render thanh anh trang tren nen den.
- Dua vao canvas co dinh `384x384`.
- Khong co resize lam nho chu theo kieu square-resize cu.

Thong so representation:
- `image_size = 384`
- `context_tail_chars = 160`
- `font_size = 18`
- `padding = 12`
- foreground: white text
- background: black
- render anchor: bottom

### Model
- Base model: `microsoft/trocr-base-printed`
- Encoder duoc freeze
- Decoder chi fine-tune top 4 layers
- Batch size: `4`
- LR: `1e-5`
- Decode:
  - `max_new_tokens=72`
  - `min_new_tokens=8`
  - `num_beams=2`
  - `repetition_penalty=1.2`
  - `no_repeat_ngram_size=3`

Thong so mo hinh va huan luyen:
- pretrained model: `microsoft/trocr-base-printed`
- trainable ratio: `117,089,280 / 333,921,792 = 35.06%`
- decoder tuning mode: `top_layers`
- tuned decoder layers: `[8, 9, 10, 11]`
- optimizer: `AdamW`
- weight decay: `0.01`
- scheduler: `Linear warmup + CosineAnnealing`
- warmup ratio: `0.10`
- gradient clipping: `1.0`
- mixed precision: `bf16=True`
- batch size: `4`
- grad accumulation: `2`
- effective micro-batch steps per optimizer step: `2`
- epochs planned: `10`
- early stop patience: `4`

### Loss
- Weighted loss cho phan dau continuation:
  - `prefix_token_count=12`
  - `prefix_loss_weight=3.0`
- Prompt tokens duoc dua vao labels va co supervision nhe:
  - `prompt_loss_weight=1.0`

Tom tat objective:
- prompt tokens duoc supervise thay vi mask hoan toan
- 12 token dau continuation duoc tang trong so
- phan con lai dung CE thong thuong

Y nghia:
- prompt supervision giup decoder hoc dieu kien ngon ngu tot hon
- prefix weighting giup model bam vao phan dau continuation

## 3. Nhung file chinh trong pipeline

- `config.py`: cau hinh train/infer
- `utils.py`: render text, clean prediction, helper
- `dataset.py`: tao dataset va labels
- `model.py`: build/load/save model
- `metrics.py`: CER, prefixCER, pixelF1, language diagnostics
- `train.py`: train loop, eval, debug logging
- `inference.py`: sinh du doan va xuat `pixels.npz`

## 4. Cac huong da thu va ket qua

Bang tong hop ngan gon:

| Variant | Main idea | Outcome |
|---|---|---|
| VLM + LoRA | multimodal assistant-style continuation | fail, wrong format |
| TrOCR basic | OCR seq2seq baseline | works but generic English drift |
| Fixed canvas representation | remove harmful shrinking | clear improvement |
| Grounding-only decoder | freeze too much | OCR junk / `.com` artifacts |
| Top-layer decoder tuning | tune top 4 decoder layers | best stable setting |
| Prefix-weighted loss | emphasize first continuation tokens | improves `prefixCER` |
| Balanced sampling | reduce English dominance | helps fairness, hurts overall CER |
| Stronger prompt wording | `Continue in ...` | unstable, rolled back |
| Prompt supervision | include prompt in labels | best final improvement |
| Candidate rerank | rerank generated outputs | worse than prompt-loss baseline |

### Huong 1: VLM/LoRA
Da thu huong VLM + LoRA cho continuation.

Van de:
- output bi `Assistant:`
- sinh cau hoi dap generic
- hoc sai format task

Ket luan:
- Khong phu hop bai nay.

### Huong 2: TrOCR baseline dau tien
Pipeline:

`context image -> TrOCR -> continuation`

Van de:
- output generic English continuation
- collapse ve opener `and`
- khong bam dung continuation that

Chi so:
- CER quanh `0.78-0.79`
- Spanish la ngon ngu te nhat

### Huong 3: Sua representation
Da bo kieu resize lam chu bi co.

Thay doi:
- render truc tiep vao canvas
- giu `rScale=(1.00,1.00)`

Tac dung:
- sua duoc loi representation that
- metric on dinh hon

Nhung:
- van con collapse decode

### Huong 4: Freeze decoder qua chat
Da thu chi train cac thanh phan grounding/cross-attention.

Van de:
- output OCR junk
- `.com/survey`
- `FACE UP ON DETAILS`

Ket luan:
- khoa decoder qua chat lam model mat kha nang continuation.

### Huong 5: Fine-tune top decoder layers
Da doi sang:
- freeze encoder
- fine-tune top 4 decoder layers

Day la huong on nhat trong cac thu nghiem.

### Huong 6: Prefix-weighted loss
Da them weighted loss cho 12 token dau continuation.

Tac dung:
- `prefixCER` giam ro
- phan dau continuation on hon

Nhung:
- model van mo dau bang `and`

### Huong 7: Balanced language sampling
Da thu WeightedRandomSampler de giam lech English.

Tac dung:
- giam English opener bias
- Spanish co luc cai thien

Nhung:
- CER tong the xau hon baseline tot nhat
- output chuyen sang multilingual gibberish

Ket luan:
- khong giu huong nay.

### Huong 8: Prompt wording manh hon
Da thu prompt:
- `Continue in Spanish:`

Tac dung:
- khong ben
- metric xau di sau 1-2 epoch

Ket luan:
- rollback.

### Huong 9: Prompt supervision
Da dua prompt tokens vao labels, khong mask hoan toan nhu truoc.

Tac dung:
- `prompt loss` giam rat ro
- best CER moi tot hon baseline cu

Day la sua doi hieu qua nhat o giai doan cuoi.

## 5. Best run dat duoc

Best run cuoi cung nen giu:
- representation moi
- top 4 decoder layers
- prefix weighted loss
- prompt supervision
- khong balanced sampling
- khong rerank

Ket qua tot nhat da ghi nhan:
- `CER ~ 0.767`
- tot hon baseline cu `~0.785`

Checkpoint tot nhat den tu run `prompt_loss`, around `epoch 2`.

## 5.1 Evaluation Metrics Used During Development

Metric da dung trong qua trinh phat trien:
- `CER` (Character Error Rate)
- `prefixCER`
- `pixelF1`
- `repeat`
- `lenRatio`
- `short%`
- `long%`
- `unique`
- `langCons`
- `firstTok`
- `engOpen`

Y nghia:
- `CER`: metric text-level chinh de theo doi chat luong continuation
- `prefixCER`: kiem tra phan dau continuation
- `pixelF1`: do muc do gan nhau trong image-space sau khi render
- `langCons`: muc do prediction khop voi marker ngon ngu mong doi
- `engOpen`: ty le prediction mo dau bang opener English nhu `and/the/to/...`

## 5.2 Best Validation Snapshot

Best snapshot quan trong nhat:
- run type: `prompt supervision + prefix weighting`
- best epoch: `epoch 2`
- `CER = 0.767`
- `pixF1 = 0.258`
- `prefixCER = 0.893`

So sanh voi baseline cu:
- old best `CER ~ 0.785`
- new best `CER ~ 0.767`
- cai thien tuy khong lon nhung la cai thien that va on dinh nhat trong cac bien the da thu

## 5.3 Kaggle Result Section

Can dien khi da co diem Kaggle:
- Public Kaggle score: `1.56`
- Private Kaggle score:
- Submission file used:
- Model checkpoint used:

Co the viet:

"The final Kaggle submission was generated from the best checkpoint of the prompt-supervised TrOCR variant. The submission was exported in the required pixel-coordinate format (`pixels.npz`) after rendering predicted continuation text into white-on-black text images."

## 5.4 Final Inference Statistics

Thong ke infer tren tap test:
- number of test samples: `364`
- language distribution:
  - English: `182`
  - French: `50`
  - Spanish: `48`
  - German: `46`
  - Italian: `38`
- total predicted pixels: `307,785`
- average pixels per sample: `846`
- average predicted characters: `36.4`
- character length range: `14..69`

Co the viet trong report:

"Inference completed successfully on all 364 test samples. The final system produced non-empty pixel-based outputs for the entire test set, with an average predicted length of 36.4 characters and 846 foreground pixels per sample. The resulting submission achieved a Kaggle score of 1.56."

## 5.5 Experiment Results Summary

Bang tong hop cac huong chinh da thu:

| Experiment | Main change | Best observed result | Outcome |
|---|---|---:|---|
| TrOCR baseline | fixed-canvas representation + top-4 decoder tuning + prefix loss | `CER ~ 0.785` | strong baseline |
| Balanced language sampling | WeightedRandomSampler by language | `CER ~ 0.800` | reduced English-opener bias but worse overall than baseline |
| Stronger prompt wording | `Continue in <language>:` | unstable, degraded after 1-2 epochs | rolled back |
| Prompt supervision | include prompt tokens in labels + prompt loss | `CER ~ 0.767` | best validation result |
| Candidate rerank | rerank candidates by language consistency / opener penalty | `CER ~ 0.810` | worse than prompt-supervision baseline |

## 5.6 Best Result Comparison

| Setting | CER | Notes |
|---|---:|---|
| Baseline before prompt supervision | `~0.785` | best around epoch 4 |
| Balanced sampling | `~0.800` | better language fairness, worse global CER |
| Prompt supervision | `~0.767` | best overall validation result |
| Rerank generation | `~0.810` | decode-side change did not help |

## 5.7 Suggested Evaluation Paragraph

Co the viet trong report nhu sau:

"Several model variants were evaluated during development. The strongest baseline, based on fixed-canvas rendering and top-layer decoder tuning, reached approximately 0.785 CER on validation samples. Adding prompt supervision further improved the best validation CER to approximately 0.767, which became the final selected configuration. Other modifications, such as balanced language sampling and decode-time candidate reranking, improved specific behaviors but did not outperform the prompt-supervised baseline in overall CER. The final Kaggle submission obtained a score of 1.56."

## 6. He thong debug da bo sung

Trong qua trinh train da in va luu:

### Sanity log
- `pixel_values shape`
- `image_size`
- `context_tail_chars`
- `prompt text`
- `prompt ids len`
- `decoder ids len`
- `label tokens`
- `orig render size`
- `render size`
- `resize ratio`
- `white density`

### Batch log
- `loss`
- `prompt`
- `prefix`
- `tail`
- `pTok`
- `lr`

### Eval log
- `CER`
- `prefixCER`
- `pixF1`
- `repeat`
- `lenRatio`
- `short%`
- `long%`
- `unique`
- `langCons`

### Decode diagnostics
- `next-token`
- `first-token mode`
- `firstTok` theo tung language
- `engOpen` theo tung language

### Debug artifacts
Moi epoch luu vao:
- `debug_outputs/train/epoch_xxx/`

gom:
- `context_*.png`
- `target_*.png`
- `pred_*.png`
- `sample_*.txt`

## 7. Loi chinh da duoc xac dinh bang so lieu

### Loi 1: Representation bi co chu
Bang chung:
- truoc khi sua, anh bi resize lam OCR rat yeu

Fix:
- fit vao canvas, khong co nho text vo ly

### Loi 2: Decoder bi LM prior lan at
Bang chung:
- output generic sentence
- `and...`
- `Assistant...`
- drift sang English

Fix da thu:
- top layer tuning
- prefix loss
- prompt supervision

### Loi 3: Language drift
Bang chung:
- `engOpen=1.00` tren nhieu non-English samples
- `firstTok=and` o epoch 2-4
- Spanish CER rat cao

Fix da thu:
- balanced sampling
- prompt wording manh hon
- prompt supervision

Ket qua:
- prompt supervision co tac dung tot nhat
- balanced sampling va rerank khong du tot de giu

### Loi 4: Generate collapse
Bang chung:
- same opener
- same generic clauses
- CER khong giam tuong ung voi prefixCER

Fix da thu:
- rerank theo language consistency

Ket qua:
- khong tot hon run prompt_loss
- da rollback

## 8. Nhung thay doi da rollback

Da thu nhung khong giu:
- balanced language sampling
- prompt `Continue in ...`
- bad words / blocker cho opener
- rerank candidate generation
- freeze decoder qua chat

Ly do:
- metric xau hon hoac khong on dinh

## 9. Phien ban cuoi cung nen nop

Nen nop phien ban:
- co prompt supervision
- khong rerank
- khong balanced sampling
- khong prompt wording manh hon

Neu can mo ta ngan gon trong report:
- final system is a TrOCR-based OCR-style continuation model operating on rendered context images
- the best variant used prompt supervision and prefix-weighted loss
- extensive diagnostics were added to analyze representation quality, language drift, and decoding collapse

## 9.1 Final Hyperparameter Table

| Hyperparameter | Value |
|---|---|
| model | `microsoft/trocr-base-printed` |
| image size | `384` |
| context tail chars | `160` |
| font size | `18` |
| padding | `12` |
| freeze encoder | `True` |
| decoder mode | `top_layers` |
| decoder train layers | `4` |
| batch size | `4` |
| grad accumulation | `2` |
| learning rate | `1e-5` |
| weight decay | `0.01` |
| warmup ratio | `0.10` |
| grad clip | `1.0` |
| bf16 | `True` |
| max target length | `128` |
| max new tokens | `72` |
| min new tokens | `8` |
| num beams | `2` |
| repetition penalty | `1.2` |
| no repeat ngram | `3` |
| length penalty | `0.8` |
| prefix token count | `12` |
| prefix loss weight | `3.0` |
| prompt loss weight | `1.0` |

## 9.2 Strengths

- Follows the competition requirement of using a non-text representation for the input.
- Uses a deterministic text-to-image rendering pipeline.
- Adds multiple diagnostics beyond raw loss.
- Improves over earlier baselines through representation fixes and prompt supervision.
- Produces valid rendered outputs that can be converted into `pixels.npz`.

## 9.3 Limitations

- Decoder still tends to fall back to generic English-like openers.
- Language control is incomplete, especially for Spanish.
- Even when `prefixCER` improves, full-sequence `CER` may plateau.
- The model still behaves partly like a language prior rather than a strongly grounded continuation model.

## 9.4 Failure Case Examples

Typical failure cases observed:
- English opener collapse:
  - prediction starts with `and ...` even for non-English samples
- OCR junk artifacts:
  - `.com/survey`
  - `FACE UP ON DETAILS`
- multilingual gibberish:
  - mixed fragments from multiple languages
- generic continuation:
  - semantically plausible but unrelated to the true continuation

## 9.5 Suggested Report Structure

Ban co the viet report theo khung sau:

1. Student information  
2. Problem description  
3. Data preprocessing and text-to-image representation  
4. Proposed model and training design  
5. Evaluation setup and metrics  
6. Experimental results  
7. Error analysis and limitations  
8. Conclusion  
9. References

## 9.6 References / Bibliography

Phan bibliography nen dung mot style thong nhat. O day de xuat dung IEEE.

### IEEE-style bibliography

[1] M. Li, T. Lv, J. Cui, Y. Lu, D. Florencio, C. Zhang, and F. Wei, “TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 11, pp. 13094-13102, 2023.

[2] Hugging Face, “Transformers Documentation,” *Hugging Face Docs*. [Online]. Available: https://huggingface.co/docs/transformers/. [Accessed: Apr. 10, 2026].

[3] Microsoft, “microsoft/trocr-base-printed,” *Hugging Face Model Hub*. [Online]. Available: https://huggingface.co/microsoft/trocr-base-printed. [Accessed: Apr. 10, 2026].

[4] Hugging Face, “Vision Encoder Decoder Models,” *Hugging Face Docs*. [Online]. Available: https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder. [Accessed: Apr. 10, 2026].

Neu trong report ban chi trich dan 2-3 nguon, co the giu [1]-[3]. Neu co nhac ro den kien truc VisionEncoderDecoder, nen giu them [4].

## 10. Limitation analysis

Nhung han che chinh:
- model van co xu huong collapse ve generic continuation
- language consistency van chua cao
- Spanish la language kho nhat
- output van co luc bi generic English opener
- decode con nhay vao pattern khong lien quan den target that

## 11. Huong cai thien neu con thoi gian

Neu con them thoi gian sau deadline, nen uu tien:
- redesign objective cho token dau continuation
- decode constrained by language more explicitly
- architecture it generative-LM-prior hon
- stronger image-text grounding thay vi continuation LM behavior

## 12. Tom tat ngan gon

Trong tat ca cac thu nghiem:
- sua representation la can thiet
- top-layer decoder tuning la hop ly nhat
- prefix loss co ich
- prompt supervision la buoc cai thien tot nhat cuoi cung
- balanced sampling, rerank, prompt wording manh hon deu khong vuot duoc baseline tot nhat

Best practical checkpoint:
- run `prompt_loss`
- best around `epoch 2`
- `CER ~ 0.767`
