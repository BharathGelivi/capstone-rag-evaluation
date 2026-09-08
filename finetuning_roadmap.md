# Fine-Tuning Roadmap: NLI Models to LLMs
## Learning Plan for X-RAG Legal Domain Adaptation

---

## The Big Picture First

Before the roadmap, understand where NLI fine-tuning sits in the landscape:

```
Fine-Tuning Universe
├── Encoder-only models (BERT, DeBERTa, RoBERTa)
│   ├── Classification tasks ← NLI lives here
│   ├── Token labeling (NER)
│   └── Question Answering (extractive)
│
├── Decoder-only models (GPT, Llama, Qwen)   ← "LLM fine-tuning"
│   ├── Instruction following (SFT)
│   ├── Chat (RLHF/DPO)
│   └── Domain adaptation
│
└── Encoder-Decoder (T5, BART)
    ├── Summarization
    └── Translation
```

**NLI fine-tuning = fine-tuning an encoder model for 3-class classification.**
**LLM fine-tuning = fine-tuning a decoder model to follow instructions.**

They share the same core mechanics but differ in architecture and task head.

---

## Is NLI Fine-Tuning Similar to LLM Fine-Tuning?

| Aspect | NLI Fine-Tuning | LLM Fine-Tuning (SFT) |
|---|---|---|
| **Architecture** | Encoder (BERT/DeBERTa) | Decoder (Llama/Qwen) |
| **Task head** | Linear classifier (3 outputs) | Language model head (vocab outputs) |
| **Loss function** | Cross-entropy (3 classes) | Cross-entropy (next token prediction) |
| **Data format** | (premise, hypothesis, label) triples | (instruction, response) pairs |
| **Training framework** | HuggingFace Trainer | HuggingFace Trainer / TRL |
| **GPU requirement** | Low (3B params model needs ~6GB) | High (7B model needs 16GB+, with LoRA ~8GB) |
| **Training time** | Hours on a single GPU | Days (full fine-tune) / Hours (LoRA) |
| **Core concept** | Same — update weights via backprop | Same |

**The mechanics are identical. The model architecture and data format differ.**

---

## Phase 0 — Prerequisites (Week 1–2)

### What You Must Know Before Starting

**0.1 Python + PyTorch Basics**
```python
# You need to be comfortable with:
import torch
tensor = torch.tensor([[1.0, 2.0, 3.0]])
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

**0.2 What a Transformer Actually Does (Conceptual)**

```
Input: ["The sky", "is blue"]  ← two sentences (premise, hypothesis)
         ↓
Tokenizer → token IDs → embeddings
         ↓
Transformer layers (attention, feed-forward) × 12 layers
         ↓
[CLS] token hidden state → shape (768,)
         ↓
Linear layer → shape (3,)   [entailment, neutral, contradiction]
         ↓
Softmax → probabilities
         ↓
Predicted class
```

The key intuition: the `[CLS]` token acts as a "summary" of the entire input pair. The classifier head reads this summary and predicts the relationship.

**0.3 What Fine-Tuning Means**

```
Pre-trained DeBERTa
(trained on Wikipedia, books — general language)
         ↓
You take its weights as a starting point
         ↓
Continue training on YOUR data
(legal claims + legal evidence + your labels)
         ↓
Weights shift to understand legal language patterns
```

You're not training from scratch. You're redirecting existing language knowledge toward your specific task.

**Resources for Phase 0:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — read this, no code needed
- [PyTorch 60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Andrej Karpathy's "Neural Networks: Zero to Hero" (YouTube) — optional but gold

---

## Phase 1 — Understanding NLI as a Task (Week 2–3)

### 1.1 What NLI Datasets Look Like

The classic NLI datasets you'll encounter:

**SNLI (Stanford NLI) — 570k examples:**
```json
{
  "premise": "A man is playing guitar on stage.",
  "hypothesis": "A musician is performing.",
  "label": "entailment"
}
{
  "premise": "A man is playing guitar on stage.",
  "hypothesis": "A woman is sleeping.",
  "label": "contradiction"
}
{
  "premise": "A man is playing guitar on stage.",
  "hypothesis": "The concert is sold out.",
  "label": "neutral"
}
```

**MultiNLI — 433k examples across 10 genres (news, fiction, government...):**
Contains a "government" genre with legal/formal language — closest to your use case.

**Your domain-specific dataset will look like:**
```json
{
  "premise": "Section 103 prescribes death or imprisonment for life for murder.",
  "hypothesis": "Murder is punishable by a maximum of 10 years imprisonment.",
  "label": "contradiction"
}
{
  "premise": "Section 399 deals with counterfeiting currency.",
  "hypothesis": "Section 399 relates to currency counterfeiting offences.",
  "label": "entailment"
}
```

### 1.2 The Three Label Meanings (for your project specifically)

| Label | In General NLI | In Your RAG Verifier |
|---|---|---|
| **Entailment** | Premise logically implies hypothesis | Retrieved chunk SUPPORTS the claim |
| **Neutral** | Premise doesn't confirm or deny | Retrieved chunk is UNRELATED to claim |
| **Contradiction** | Premise logically opposes hypothesis | Retrieved chunk CONTRADICTS the claim |

### 1.3 Exercise: Run Inference on an Existing Model

```python
from transformers import pipeline

nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-large",
    device=-1  # CPU
)

result = nli(
    "Section 103 prescribes death or life imprisonment for murder.",
    text_pair="Murder is punishable by a maximum of 10 years.",
    top_k=None
)
print(result)
# [{'label': 'contradiction', 'score': 0.92},
#  {'label': 'neutral', 'score': 0.05},
#  {'label': 'entailment', 'score': 0.03}]
```

Do this with 10 examples from your legal corpus. Note where it gets it wrong — these errors become your training data motivation.

---

## Phase 2 — The HuggingFace Trainer (Week 3–5)

This is the core skill. Everything else builds on this.

### 2.1 The Standard Fine-Tuning Loop

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch

# ── Step 1: Load model + tokenizer ──────────────────────────────────────────
model_name = "cross-encoder/nli-deberta-v3-small"  # start small!

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,          # entailment / neutral / contradiction
    ignore_mismatched_sizes=True
)

# ── Step 2: Prepare your dataset ────────────────────────────────────────────
# Your data: list of dicts with premise, hypothesis, label
raw_data = [
    {"premise": "...", "hypothesis": "...", "label": 0},  # 0=entailment
    {"premise": "...", "hypothesis": "...", "label": 1},  # 1=neutral
    {"premise": "...", "hypothesis": "...", "label": 2},  # 2=contradiction
]

dataset = Dataset.from_list(raw_data)

# ── Step 3: Tokenization ─────────────────────────────────────────────────────
def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.train_test_split(test_size=0.2)  # 80/20 split

# ── Step 4: Metrics ───────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import f1_score, classification_report

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_per_class": f1_score(labels, preds, average=None).tolist()
    }

# ── Step 5: Training config ───────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./legal_nli_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=10,
    report_to="none"   # set to "wandb" when you want experiment tracking
)

# ── Step 6: Train ─────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./legal_nli_final")
```

### 2.2 Key Hyperparameters You Must Understand

| Parameter | Typical Value | What It Does |
|---|---|---|
| `learning_rate` | 1e-5 to 5e-5 | How fast weights update — too high = training collapses |
| `num_train_epochs` | 3–5 | How many full passes over data |
| `per_device_train_batch_size` | 8–32 | Samples per GPU step — limited by RAM |
| `weight_decay` | 0.01 | L2 regularization — prevents overfitting |
| `warmup_steps` | 0–500 | Slowly ramp up learning rate — stabilizes early training |
| `max_length` | 512 | Token truncation limit — critical for NLI pairs |

### 2.3 What Loss Looks Like During Training

```
Epoch 1: train_loss=1.09  eval_f1=0.41   ← random chance is 0.33 (3 classes)
Epoch 2: train_loss=0.65  eval_f1=0.71   ← model learning
Epoch 3: train_loss=0.31  eval_f1=0.83   ← good
Epoch 4: train_loss=0.15  eval_f1=0.81   ← overfitting starting (eval drops)
Epoch 5: train_loss=0.08  eval_f1=0.79   ← definitely overfitting
```

`load_best_model_at_end=True` saves you — it loads the epoch 3 weights automatically.

---

## Phase 3 — Building Your Legal NLI Dataset (Week 5–7)

This is the hard part and the real research contribution.

### 3.1 Data Creation Strategy

You need (premise, hypothesis, label) triples from **your actual corpus** (BNS, BNSS, BSA).

**Source 1: SNLI-style generation from your chunks**

Take a chunk from your corpus and write claims for it:
```
Chunk: "Section 103. Punishment for murder.—
Whoever commits murder shall be punished with death, or 
imprisonment for life, and shall also be liable to fine."

Entailment claim: "Section 103 provides for capital punishment in murder cases."
Neutral claim:    "Murder rates in India have changed since 2020."
Contradiction:    "Section 103 limits punishment for murder to 5 years imprisonment."
```

**Source 2: Use your existing eval dataset**

Your 40 evaluation questions contain gold answers vs. generated answers. Each verified claim in your `artifacts/` directory is already labeled (SUPPORTED/UNSUPPORTED/CONTRADICTED). This is already NLI training data — you just need to convert the format.

**Source 3: Synthetic generation via LLM**

```python
# Prompt to Groq/LLM:
prompt = f"""
Given this legal text:
{chunk_text}

Generate:
1. One ENTAILMENT claim (a fact directly stated in the text)
2. One NEUTRAL claim (about a related but unconfirmed topic)  
3. One CONTRADICTION claim (a fact that directly opposes the text)

Format as JSON: [{{"claim": "...", "label": "entailment"}}, ...]
"""
```

Then **manually verify** each LLM-generated example before using it as training data. Never trust LLM labels blindly.

### 3.2 Target Dataset Size

| Phase | Size | Purpose |
|---|---|---|
| Pilot | 50 examples | Verify training loop works |
| Small | 200 examples | Initial model, compare vs. base |
| Research | 500+ examples | Publishable fine-tuned model |

For your 4-month deadline: **200 examples is achievable and sufficient for a comparison in the paper.**

### 3.3 Label Distribution

Balance your labels:
```
Entailment:    34% (68 examples)
Neutral:       33% (66 examples)
Contradiction: 33% (66 examples)
Total:         200 examples
```

An imbalanced dataset (e.g., 80% entailment, 10% each for neutral/contradiction) will produce a model that rarely predicts contradiction — exactly your current problem.

---

## Phase 4 — Evaluating Your Fine-Tuned Model (Week 7–8)

### 4.1 Metrics You Must Report

```python
from sklearn.metrics import classification_report

# Per-class F1 is more informative than accuracy for NLI
print(classification_report(
    y_true, y_pred,
    target_names=["entailment", "neutral", "contradiction"]
))

# Output:
#               precision  recall  f1-score  support
# entailment       0.89     0.91     0.90      40
# neutral          0.78     0.74     0.76      40
# contradiction    0.85     0.87     0.86      40
# macro avg        0.84     0.84     0.84     120
```

**For your paper, report:**
- F1-macro (average across all 3 classes)
- F1-contradiction specifically (this is your broken class right now)
- Compare base model vs. your fine-tuned model on the same test set

### 4.2 Threshold Calibration After Fine-Tuning

After fine-tuning, the model's raw scores will be better calibrated. Re-run your calibration script:

```bash
python -m scripts.calibrate_thresholds
```

Plot precision-recall curves for each threshold value and pick the one that maximizes F1 on your labeled examples. **This is the justification for whatever threshold you end up using in the paper** — no more magic numbers.

---

## Phase 5 — LLM Fine-Tuning (Week 9–12)

Now you understand encoder fine-tuning. LLM fine-tuning (decoder) is the same loop but with different architecture, bigger models, and LoRA to make it memory-efficient.

### 5.1 The Core Difference: Causal Language Modeling

```
NLI Training:
Input:  [CLS] premise [SEP] hypothesis [SEP]
Output: class label (0, 1, 2)
Loss:   CrossEntropy(predicted_class, true_class)

LLM Training (SFT):
Input:  [INST] What is the punishment for murder? [/INST]
Output: Section 103 prescribes death or life imprisonment.
Loss:   CrossEntropy(predicted_next_token, actual_next_token)
         averaged across all output tokens
```

For LLMs, the loss is computed over every output token — the model learns to predict each word given all previous words.

### 5.2 Why Full Fine-Tuning LLMs Is Hard

A 7B parameter model (Llama, Qwen) requires:
```
7B params × 4 bytes (float32) = 28 GB just to STORE the model
+ gradients = another 28 GB
+ optimizer states (Adam) = another 56 GB
Total: ~112 GB VRAM — you don't have this
```

### 5.3 LoRA — The Solution That Makes It Practical

**LoRA (Low-Rank Adaptation):** Instead of updating all 7B weights, you add small "adapter" matrices (A and B) to each attention layer and only train those.

```
Original weight matrix W (frozen)
     +
Low-rank adapter:  A × B   (trained)
     ↓
Effective weight = W + A×B

A: (d × r)    r = rank, e.g., 16
B: (r × d)
Parameters in A×B: 2 × d × r = 2 × 4096 × 16 = 131,072

Compare to full W:  d × d = 4096 × 4096 = 16,777,216

LoRA trains 0.78% of the parameters of a full fine-tune.
Memory needed: ~8–12 GB with 4-bit quantization (feasible on free Google Colab)
```

### 5.4 LoRA Fine-Tuning Code (TRL Library)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ── 4-bit quantization (QLoRA) — fits on free Colab T4 ──────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank
    lora_alpha=32,                 # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]  # which layers to adapt
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 6,815,744 || all params: 7,248,578,560 || trainable%: 0.09

# ── Data format for instruction tuning ───────────────────────────────────────
def format_prompt(example):
    return {
        "text": f"<|im_start|>system\nYou are a legal expert.\n<|im_end|>\n"
                f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{example['answer']}\n<|im_end|>"
    }

dataset = Dataset.from_list(your_qa_pairs).map(format_prompt)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./legal_llm",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        learning_rate=2e-4,
        max_seq_length=1024,
    )
)
trainer.train()
```

### 5.5 Other Fine-Tuning Types You Should Know

| Type | What It Does | When to Use |
|---|---|---|
| **SFT** (Supervised Fine-Tuning) | Teach the model to follow instructions | Domain adaptation, your use case |
| **RLHF** | Train using human preference feedback | Alignment (OpenAI uses this) |
| **DPO** (Direct Preference Optimization) | Simpler version of RLHF | When you have (good, bad) response pairs |
| **QLoRA** | LoRA + 4-bit quantization | Memory-constrained training (your situation) |
| **Adapter layers** | Small bottleneck layers instead of LoRA | Older technique, LoRA is now preferred |

---

## Your Personal Roadmap (4-Month Timeline)

```
MONTH 1: Foundations + NLI Basics
├── Week 1-2: Read "Illustrated Transformer", run inference on NLI models
├── Week 3:   Understand HuggingFace Trainer loop with SNLI dataset
└── Week 4:   Fine-tune deberta-v3-small on SNLI subset (100 examples)
              Goal: see training loop work end-to-end

MONTH 2: Build Your Legal NLI
├── Week 5-6: Create 200 labeled legal (premise, hypothesis, label) triples
│             from your BNS/BNSS/BSA corpus
├── Week 7:   Fine-tune on your legal data, evaluate vs. base model
└── Week 8:   Run calibrate_thresholds.py with new model
              Swap into X-RAG, re-run 40-example evaluation
              RESULT: Paper now has "Legal NLI" contribution

MONTH 3: LLM Fine-Tuning
├── Week 9-10:  Study LoRA/QLoRA, run Colab tutorial
├── Week 11:    Fine-tune Qwen2.5-7B on your 40 Q&A pairs (small experiment)
└── Week 12:    Compare: base Qwen vs. fine-tuned Qwen on your eval set
               RESULT: Understand the full fine-tuning spectrum

MONTH 4: Paper + Integration
├── Week 13-14: Fix all critical bugs, run full 40-example evaluation
├── Week 15:    Write Results section with both NLI + evaluation findings
└── Week 16:    Final paper editing, abstract, related works
```

---

## Tools & Resources

### Must-Have Tools
| Tool | Purpose |
|---|---|
| `transformers` | Load/train any HuggingFace model |
| `datasets` | Dataset loading and processing |
| `peft` | LoRA implementation |
| `trl` | SFT/DPO training utilities |
| `sklearn` | Metrics (F1, confusion matrix) |
| `wandb` | Experiment tracking (free tier) |

### Learning Resources (in order)

1. **[HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)** — Free, covers tokenizers, fine-tuning, everything. Do chapters 1–3.
2. **[HuggingFace PEFT docs](https://huggingface.co/docs/peft)** — LoRA tutorial specifically.
3. **Sebastian Raschka's "LLMs from Scratch"** (book/GitHub) — Best deep-dive on LLM architecture.
4. **Andrej Karpathy's "Let's build GPT"** (YouTube) — Understand what's actually happening in transformers.

### Free GPU Resources
| Platform | GPU | Limit | Use For |
|---|---|---|---|
| Google Colab (free) | T4 (16GB) | ~10 hrs/day | NLI fine-tuning (small models) |
| Colab Pro | A100 (40GB) | Paid ~$10/mo | LLM fine-tuning with QLoRA |
| Kaggle Notebooks | T4 ×2 | 30 hrs/week | NLI fine-tuning |
| HuggingFace Spaces | Varies | Free tier | Hosting inference |

---

## Key Insight: What Actually Happens During Fine-Tuning

```
Before fine-tuning (general DeBERTa):
"Section 399" → model has no idea what this is

During fine-tuning (seeing your data 3× epochs):
Backpropagation adjusts weights so that:
- "Section 399" near "currency counterfeiting" → entailment with "currency-related"
- "Section 103" near "death penalty" → entailment with "capital punishment"  
- "Section 103" near "10 years maximum" → contradiction

After fine-tuning:
The model has restructured its internal representations
to better capture legal entailment patterns
```

The weights don't memorize your 200 examples — they generalize the **pattern** that legal text uses to express entailment/contradiction, making the model better on new legal examples it has never seen.
