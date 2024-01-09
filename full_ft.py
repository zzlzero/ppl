import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

# 加载预训练的 GPT-2 模型和分词器
device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# 这里我们使用 HuggingFace 的 TextDataset 和 DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

train_dataset = load_dataset("titles.txt", tokenizer)
data_collator = load_data_collator(tokenizer)





# 将句子列表转换为模型输入的格式




def train(model, tokenizer, train_dataset, data_collator, output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):


    tokenizer.save_pretrained(output_dir)
      

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
    )
      
    trainer.train()
    trainer.save_model()

overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 20
save_steps = 500
output_dir = './full_ft/epoch={}'.format(num_train_epochs)

train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=data_collator,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)
# 保存微调后的模型
model.save_pretrained("full_fine_tuned_gpt2")