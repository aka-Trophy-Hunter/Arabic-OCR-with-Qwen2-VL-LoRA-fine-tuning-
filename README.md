# Arabic OCR: LoRA Fine-Tuning of Qwen2-VL-2B (Small-Scale Study)

A learning project: supervised fine-tuning (LoRA) of Qwen2-VL-2B-Instruct for Arabic OCR,
using data and the OCR prompt from [KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench)
(Heakl et al., ACL 2025). **This is not an official KITAB-Bench result.**

## Setup
- **Model:** Qwen2-VL-2B-Instruct, fp16 on a Tesla T4 (Google Colab)
- **Method:** LoRA (r=16, alpha=32, dropout 0.05) on attention and MLP projections
  (18.5M trainable params, 0.83% of the model); loss computed on answer tokens only
- **Train:** 300 samples from `ahmedheakl/arocrbench_synthesizear`, 1 epoch (38 steps),
  lr 1e-4, effective batch size 8
- **Test:** 30 held-out samples from the same subset (in-domain) and 30 samples from
  `ahmedheakl/arocrbench_hindawi` (out-of-domain, book pages)
- **Prompt:** the same one KITAB-Bench uses
- **Metrics:** CER / WER (`jiwer`) after KITAB-Bench's Arabic text normalization

## Results

| | CER | WER |
|---|---|---|
| Base, in-domain | 0.349 | 0.589 |
| LoRA, in-domain | **0.043** | **0.167** |
| Base, out-of-domain | 0.873 | 0.950 |
| LoRA, out-of-domain | **0.691** | **0.840** |

Training loss fell from 0.87 to 0.47 over 38 steps.

| Step | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|---|---|---|---|---|---|---|---|
| Loss | 0.87 | 0.75 | 0.76 | 0.67 | 0.61 | 0.56 | 0.47 |

## Failure cases (worst 3 of 30 in-domain samples)
1. **Dot-dependent letters confused:** *نظاما → بظامما*, *حراريا → خرازيا*. Letters that
   differ only in dots (ن/ب, ح/خ, ر/ز) get swapped.
2. **Repeated or inserted letters:** *الفلور → الفللور*, *تهاجم → تهااجم*. Connected
   cursive strokes are sometimes read as extra letters.
3. **Plausible but wrong words:** *الاشعه → شفه*, *هذا السياق → فها الساف*. The model
   produces Arabic-looking text that is not what the image shows.

## Limitations
- Tiny train/test sets, a single run and seed; treat results as indicative only
- The in-domain test is synthetic text matching the training distribution, so the large
  gain is partly distribution match
- Out-of-domain performance is still weak; generation was capped at 512 new tokens,
  which may truncate long pages
- KITAB-Bench evaluates 7B models; this uses 2B, so numbers are **not comparable**
  with the paper

## Next steps
- Repeat with several seeds and report mean and spread
- Train on more and more varied data (real scans, handwriting)
- Hold out a validation set and tune epochs and LoRA rank
- Try the 7B model and the full KITAB-Bench OCR split for a comparison closer to the paper
  
## Reproduce
Open `notebook.ipynb` on a Colab T4. The adapter is in `qwen2vl-arabic-ocr-lora/`.
The LoRA adapter is on Hugging Face: [base_model: HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/moneebaa/qwen2vl-arabic-ocr-lora/tree/main)

## Credits
KITAB-Bench (MBZUAI-Oryx), Qwen2-VL (Alibaba), Hugging Face Transformers and PEFT.
