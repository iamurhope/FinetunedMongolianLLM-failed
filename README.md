# 🇲🇳 Fine-tuned Mongolian LLM (Llama-2-7B + LoRA + Unsloth)

**Llama-2-7B суурь загварыг монгол хэлээр instruction дагах, асуулт-хариулт зэрэг NLP даалгаварт зориулж Parameter-efficient (PEFT) аргаар LoRA болон Unsloth ашиглан fine-tune хийсэн төсөл.**

---

## ✨ Онцлог

- 🧠 **Суурь загвар:** Llama-2-7B (Meta AI)
- ⚙️ **Сургалтын арга:** LoRA + 4-bit Quantization
- 🚀 **Framework:** Unsloth (Colab optimized)
- 📚 **Dataset формат:** Alpaca (`instruction`, `input`, `output`)
- 🎯 **Зорилго:** Монгол хэлний LLM-н заавар дагах болон QA чанарыг сайжруулах

---

## ⚙️ Загварын тохиргоо

```python
model_name = "meta-llama/Llama-2-7b-hf"
max_seq_length = 2048
lora_config = {
  "r": 16,
  "alpha": 16,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "load_in_4bit": True
}
```

---

## 🚀 Сургалтын параметрүүд

```python
training_args = {
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 5,
  "num_train_epochs": 6,
  "learning_rate": 2e-4,
  "optimizer": "adamw_8bit",
  "weight_decay": 0.01,
  "lr_scheduler_type": "linear",
  "save_steps": 100,
  "save_total_limit": 3
}
```

---

## 📦 Суурилуулах (Google Colab)

```bash
pip install -q -U transformers accelerate peft bitsandbytes datasets huggingface_hub
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft
```

### Google Drive холбох:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧠 Dataset жишээ

```json
{
  "instruction": "Хиймэл оюун ухаан гэж юу вэ?",
  "input": null,
  "output": "Хиймэл оюун ухаан бол компьютерийн шинжлэх ухааны салбар бөгөөд өгөгдлөөс суралцаж, логик шийдвэр гаргах чадвартай ухаалаг систем юм."
}
```

---

## ⚠️ Өгөгдлийн сангийн чанарын том асуудал

| Асуудал | Тайлбар |
|---|---|
| ❌ `input` = float64 | Текст байх ёстой талбарт санамсаргүй **тоо** орсон |
| ❌ Авто орчуулга | Англи→Монгол орчуулга **дүрмийн болон утгын их алдаатай** |
| ❌ Wrong facts | Түүх, газарзүй, шинжлэх ухааны **баримт худал/зөрчилтэй** |
| ❌ Mixed languages | Датад **монгол, англи, хятад** холигдсон |
| ❌ Hallucination их | Хариултууд логик ба контекст **тасарсан, хий үзэгдэлтэй** |

### Нөлөө:
```
👉 Суурь Llama-2-ын мэдлэг муу датад хэт overfit хийснээс  
   “Catastrophic Forgetting + Bad Overfitting” үүсэж, буруу хариулт өгдөг болсон.
```

---

## 🤖 Inference ашиглах код

```python
from unsloth import FastLanguageModel

# Загварыг 4-bit-ээр ачаалах
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/your/model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Inference горимд шилжүүлэх
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Prompt-ийг токенчилж GPU-руу дамжуулах
inputs = tokenizer([
    alpaca_prompt.format("Хиймэл оюун ухаан гэж юу вэ?", "")
], return_tensors="pt").to("cuda")

# Текст үүсгэх
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.batch_decode(outputs)[0])
```

---
> **Тэмдэглэл:** Энэ бол сургалтын төсөл бөгөөд цэвэр dataset + validation + testing-ээр  
> үргэлжлүүлэн сайжруулбал production-д бэлэн болгох боломжтой 🚀🇲🇳
