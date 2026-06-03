# Own-Custom-LLM-Model
Create and Train Own Custom LLM Model

## Load ML Model
from google.colab import drive
drive.mount('/content/drive')

## ML Model with TP
import tensorflow as tf

model_path = "/content/drive/MyDrive/my_model.h5"
model = tf.keras.models.load_model(model_path)

model.summary()

## If you get errors while loading
model = tf.keras.models.load_model("my_model.h5", compile=False)

## If the model has custom layers or custom functions
model = tf.keras.models.load_model(
    "my_model.h5",
    custom_objects={
        "CustomLayerName": CustomLayerName
    }
)

# Qwen2.5-Coder Retrain on Colab/Jupiter notebook

## Then check GPU:

!nvidia-smi

## Install required packages
!pip install -U transformers accelerate datasets peft bitsandbytes trl safetensors

## Load Qwen2.5-Coder-1.5B-Instruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

## Test the model
prompt = "Create a Python function to add two numbers."

messages = [
    {
        "role": "system",
        "content": "You are a helpful coding assistant."
    },
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.2,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True
)

print(response)

## Create a training dataset
from datasets import Dataset

training_data = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional coding assistant."
            },
            {
                "role": "user",
                "content": "Create a Python function to add two numbers."
            },
            {
                "role": "assistant",
                "content": """```python
def add_numbers(a, b):
    return a + b
```"""
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional coding assistant."
            },
            {
                "role": "user",
                "content": "Create a JavaScript function to validate an email address."
            },
            {
                "role": "assistant",
                "content": """```javascript
function isValidEmail(email) {
  const pattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return pattern.test(email);
}
```"""
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional coding assistant."
            },
            {
                "role": "user",
                "content": "Create a PHP function to connect to MySQL using PDO."
            },
            {
                "role": "assistant",
                "content": """```php
function connectDatabase($host, $dbName, $username, $password) {
    try {
        $dsn = "mysql:host=$host;dbname=$dbName;charset=utf8mb4";
        return new PDO($dsn, $username, $password, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
        ]);
    } catch (PDOException $e) {
        throw new Exception("Database connection failed: " . $e->getMessage());
    }
}
```"""
            }
        ]
    }
]

dataset = Dataset.from_list(training_data)

print(dataset)

## Fine-tune using QLoRA
import gc
import torch

del model
gc.collect()
torch.cuda.empty_cache()

## Load model in 4-bit
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

## Configure LoRA adapter
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

## Train with SFTTrainer
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen2_5_coder_1_5b_lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    max_length=1024,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    eos_token="<|im_end|>"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

trainer.train()

## Save the tuned adapter
trainer.save_model("./qwen2_5_coder_1_5b_lora")
tokenizer.save_pretrained("./qwen2_5_coder_1_5b_lora")

## To save it to Google Drive
from google.colab import drive
drive.mount('/content/drive')

trainer.save_model("/content/drive/MyDrive/qwen2_5_coder_1_5b_lora")
tokenizer.save_pretrained("/content/drive/MyDrive/qwen2_5_coder_1_5b_lora")

## Use your fine-tuned model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "./qwen2_5_coder_1_5b_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

## Recommended training settings for Colab
### For T4 GPU
per_device_train_batch_size=1
gradient_accumulation_steps=4
max_length=512 or 1024
num_train_epochs=2 or 3
learning_rate=2e-4

### For better GPU like A100
per_device_train_batch_size=2
gradient_accumulation_steps=8
max_length=2048
num_train_epochs=3
learning_rate=1e-4

## clear memory
import gc
import torch

try:
    del model
except:
    pass

gc.collect()
torch.cuda.empty_cache()

## Load model using FP16, not BF16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # IMPORTANT: use float16
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16               # IMPORTANT: use float16
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

## LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

## Use FP16 training and disable BF16
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen2_5_coder_1_5b_lora",

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,

    max_length=1024,

    fp16=True,      # IMPORTANT
    bf16=False,     # IMPORTANT

    optim="paged_adamw_8bit",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

trainer.train()

## check your GPU
import torch

print(torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

## If it says: BF16 supported: False
fp16=True
bf16=False

# Start

## Install packages
!pip install -U transformers accelerate datasets peft bitsandbytes trl safetensors

## Create sample dataset
from datasets import Dataset

training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a professional coding assistant."},
            {"role": "user", "content": "Create a Python function to add two numbers."},
            {"role": "assistant", "content": "```python\ndef add_numbers(a, b):\n    return a + b\n```"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a professional coding assistant."},
            {"role": "user", "content": "Create a JavaScript function to validate an email address."},
            {"role": "assistant", "content": "```javascript\nfunction isValidEmail(email) {\n  const pattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n  return pattern.test(email);\n}\n```"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a professional coding assistant."},
            {"role": "user", "content": "Create a PHP PDO MySQL database connection function."},
            {"role": "assistant", "content": "```php\nfunction connectDatabase($host, $dbName, $username, $password) {\n    $dsn = \"mysql:host=$host;dbname=$dbName;charset=utf8mb4\";\n    return new PDO($dsn, $username, $password, [\n        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION\n    ]);\n}\n```"}
        ]
    }
]

dataset = Dataset.from_list(training_data)
print(dataset)

## Load Qwen in 4-bit and manually attach LoRA
import os
import gc
import torch

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

model = get_peft_model(model, peft_config)

# Important: force trainable LoRA parameters to float32
# This avoids BF16 gradient scaler errors in Colab.
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

## Train without FP16/BF16 mixed precision
fp16=False
bf16=False

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen2_5_coder_1_5b_lora",

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    logging_steps=1,

    save_steps=50,
    save_total_limit=2,

    max_length=1024,

    fp16=False,
    bf16=False,

    optim="paged_adamw_8bit",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()

## Save trained adapter
trainer.save_model("./qwen2_5_coder_1_5b_lora")
tokenizer.save_pretrained("./qwen2_5_coder_1_5b_lora")

## Change local model folder name
trainer.save_model("./qwen2_5_coder_1_5b_lora")
tokenizer.save_pretrained("./qwen2_5_coder_1_5b_lora")

## Use own name
my_model_name = "./dhananja-coding-assistant"

trainer.save_model(my_model_name)
tokenizer.save_pretrained(my_model_name)

## Save to Google Drive with your model name
from google.colab import drive
drive.mount('/content/drive')

my_model_name = "/content/drive/MyDrive/dhananja-coding-assistant"

trainer.save_model(my_model_name)
tokenizer.save_pretrained(my_model_name)

## Use tuned model from local folder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "/content/drive/MyDrive/dhananja-coding-assistant"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

model.eval()

## Test your tuned coding model
messages = [
    {
        "role": "system",
        "content": "You are Dhananja Coding Assistant, a professional full-stack coding assistant."
    },
    {
        "role": "user",
        "content": "Create a Node.js Express login API with JWT authentication."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=700,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True
)

print(response)

## Upload your model adapter to Hugging Face with custom name
from huggingface_hub import login

login()

## Then upload your adapter folder
from huggingface_hub import create_repo, upload_folder

repo_id = "YOUR_HF_USERNAME/dhananja-coding-assistant"

create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)

upload_folder(
    folder_path="/content/drive/MyDrive/dhananja-coding-assistant",
    repo_id=repo_id,
    repo_type="model"
)

## Hugging Face Hub supports uploading local folders to a model repository
repo_id = "aksd-dhananja/dhananja-coding-assistant"

## Use your uploaded Hugging Face adapter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_id = "YOUR_HF_USERNAME/dhananja-coding-assistant"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_id
)

model.eval()

# Retrain/Tune

## Install packages in Colab
!pip install -U transformers accelerate datasets peft bitsandbytes trl safetensors huggingface_hub

## Then
from huggingface_hub import login

login(token="hf_TexAbFEcmJdgUFguNOrLGQMfrZJLvAzXsRVX")

## Load The Stack with streaming
from datasets import load_dataset

stack_stream = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    split="train",
    streaming=True,
    token=True
)

sample = next(iter(stack_stream))
print(sample.keys())
print(sample["content"][:1000])

## The dataset supports language-specific loading like
data_dir="data/python"
data_dir="data/javascript"
data_dir="data/php"
data_dir="data/sql"
data_dir="data/typescript"
data_dir="data/html"
data_dir="data/css"

## Create a small clean training dataset
from datasets import Dataset

ALLOWED_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc"
}

def license_ok(sample):
    licenses = sample.get("licenses", [])

    if isinstance(licenses, str):
        licenses = [licenses]

    licenses = [lic.lower() for lic in licenses]

    return any(lic in ALLOWED_LICENSES for lic in licenses)

def quality_ok(sample):
    content = sample.get("content", "")

    if not content:
        return False

    # Avoid very small or very large files for Colab testing
    if len(content) < 200:
        return False

    if len(content) > 8000:
        return False

    # Avoid minified / unreadable files
    if sample.get("max_line_length", 0) > 300:
        return False

    if sample.get("avg_line_length", 0) > 120:
        return False

    return True

def format_sample(sample):
    lang = sample.get("lang", "code")
    content = sample["content"]

    return {
        "text": f"""<|fim_prefix|># Language: {lang}
# Continue the following source code.

{content}<|fim_suffix|>"""
    }

def build_stack_dataset(language="python", max_samples=2000):
    stream = load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{language}",
        split="train",
        streaming=True,
        token=True
    )

    rows = []

    for sample in stream:
        if not quality_ok(sample):
            continue

        if not license_ok(sample):
            continue

        rows.append(format_sample(sample))

        if len(rows) >= max_samples:
            break

    return Dataset.from_list(rows)

train_dataset = build_stack_dataset("python", max_samples=2000)

print(train_dataset)
print(train_dataset[0]["text"][:1000])

## Load Qwen2.5-Coder with QLoRA
import os
import torch
import gc

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

## Add LoRA adapter
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

model = get_peft_model(model, peft_config)

# Avoid BF16/FP16 gradient issues in Colab
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

## Train on The Stack dataset
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen_stack_python_lora",

    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,
    logging_steps=10,

    save_steps=200,
    save_total_limit=2,

    max_length=1024,
    packing=True,

    fp16=False,
    bf16=False,

    optim="paged_adamw_8bit",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

trainer.train()

## For a Colab T4 GPU, keep
max_length=512 or 1024
per_device_train_batch_size=1
gradient_accumulation_steps=4
num_train_epochs=1

## Save your trained adapter on Google Drive
from google.colab import drive
drive.mount("/content/drive")

trainer.save_model("/content/drive/MyDrive/qwen_stack_python_lora")
tokenizer.save_pretrained("/content/drive/MyDrive/qwen_stack_python_lora")

## Use trained Model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "/content/drive/MyDrive/qwen_stack_python_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

## Test it
messages = [
    {
        "role": "system",
        "content": "You are a professional coding assistant."
    },
    {
        "role": "user",
        "content": "Create a Python Flask API for user login with JWT."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=700,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True
)

print(response)

# Train JavaScript

## Install packages
!pip install -U transformers accelerate datasets peft bitsandbytes trl safetensors huggingface_hub

## Login to Hugging Face
from huggingface_hub import login, whoami

login(token="hVX")
print(whoami())

## Test JavaScript streaming dataset
from datasets import load_dataset

js_stream = load_dataset(
    "bigcode/the-stack",
    data_dir="data/javascript",
    split="train",
    streaming=True,
    token=True
)

sample = next(iter(js_stream))

print(sample.keys())
print(sample["content"][:1000])

## Full JavaScript QLoRA training code prepare dataset
from datasets import load_dataset

def quality_filter(sample):
    content = sample.get("content", "")

    if not content:
        return False

    # Avoid tiny files
    if len(content) < 200:
        return False

    # Avoid very large files in Colab
    if len(content) > 12000:
        return False

    # Avoid minified JavaScript
    if sample.get("max_line_length", 0) > 500:
        return False

    if sample.get("avg_line_length", 0) > 180:
        return False

    return True


def format_js_sample(sample):
    content = sample["content"]

    return {
        "text": f"""// JavaScript source code
// Continue or complete the following code.

{content}"""
    }


js_stream = load_dataset(
    "bigcode/the-stack",
    data_dir="data/javascript",
    split="train",
    streaming=True,
    token=True
)

train_dataset = (
    js_stream
    .shuffle(buffer_size=10_000, seed=42)
    .filter(quality_filter)
    .map(format_js_sample)
)

first = next(iter(train_dataset))
print(first["text"][:1000])

## Load Qwen2.5-Coder with QLoRA
import os
import torch

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

## Attach LoRA adapter
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

model = get_peft_model(model, peft_config)

# Avoid BF16 gradient issues in Colab
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

## Train on full JavaScript stream
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen_javascript_full_lora",

    # Streaming dataset needs max_steps
    max_steps=1000,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,
    warmup_steps=50,

    logging_steps=10,
    save_steps=250,
    save_total_limit=2,

    max_length=1024,
    packing=True,

    fp16=False,
    bf16=False,

    optim="paged_adamw_8bit",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

trainer.train()

## Save to Google Drive:
from google.colab import drive
drive.mount("/content/drive")

trainer.save_model("/content/drive/MyDrive/qwen_javascript_full_lora")
tokenizer.save_pretrained("/content/drive/MyDrive/qwen_javascript_full_lora")

## Use the trained JavaScript model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "/content/drive/MyDrive/qwen_javascript_full_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

model.eval()

## Test
messages = [
    {
        "role": "system",
        "content": "You are a professional JavaScript and Node.js coding assistant."
    },
    {
        "role": "user",
        "content": "Create a Node.js Express API with login, JWT authentication, and MySQL user validation."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=800,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[-1]:],
    skip_special_tokens=True
)

print(response)

## Best method: continue from checkpoint
import os, glob

output_dir = "./qwen_javascript_full_lora"

checkpoints = sorted(
    glob.glob(os.path.join(output_dir, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1])
)

print(checkpoints)
print("Latest checkpoint:", checkpoints[-1] if checkpoints else "No checkpoint found")

## Continue Process
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen_javascript_full_lora",

    # Continue until 2000 total steps
    max_steps=2000,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,
    warmup_steps=50,

    logging_steps=10,
    save_steps=250,
    save_total_limit=3,

    max_length=1024,
    packing=True,

    fp16=False,
    bf16=False,

    optim="paged_adamw_8bit",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

trainer.train(resume_from_checkpoint=True)

## 
