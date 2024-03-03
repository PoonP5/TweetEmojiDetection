from datetime import datetime
import os, sys
import shutil
import gc

import torch
import transformers
import pandas as pd
import datasets
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from trl import SFTTrainer

from label_map import id_to_emoji

use_clearml = False
if use_clearml:
    from clearml import Task

    task = Task.init(project_name='emoji', task_name='mistral train', reuse_last_task_id=False)
    task.add_tags("mock")
    logger = task.get_logger()

# Get the current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

do_train = True
do_test = False
do_not_eval = False
only_eval = False
only_save = False
resume_from_full_finetuned_model = False
resume = False
merge_adapter_to_full_model = False  # specify resume path
save_only_adapter = True

num_epoch = 1
train_size = 0.8

# only for debugging purpose
# for actual experiment, set to -1
num_train = 10
# eval every epoch with 200 samples
num_eval = 200

gradient_accumulation_steps = 1
per_device_train_batch_size = 60

max_seq_length = 1000

learning_rate = 2e-4
logging_steps = 0.25
device = "cuda"


############ Read data in to datasets #####################

def read_text(text_path):
    # Read Tweet data and return a list of each tweet
    # text_path = "data/us_test.text"
    text_list = []
    with open(text_path, "r") as text_file:
        for line in text_file:
            line = line.strip()
            text_list.append(line)

    print(len(text_list))
    return text_list


def read_label(label_path):
    # label_path = "data/us_test.labels"
    label_list = []
    with open(label_path, "r") as label_file:
        for line in label_file:
            line = line.strip()
            line = int(line)
            label_list.append(line)

    print(len(label_list))
    print(label_list[:2])

    print("set", set(label_list))
    return label_list


train_text_list = read_text("data/us_test.text")
train_label_list = read_label("data/us_test.labels")

train_data_dict = {'TITLE': train_text_list, 'CATEGORY': train_label_list, 'ENCODE_CAT': train_label_list}
train_df = pd.DataFrame(data=train_data_dict)

train_df_ = train_df.sample(frac=train_size, random_state=200)
eval_df = train_df.drop(train_df_.index).reset_index(drop=True)
train_df = train_df_.reset_index(drop=True)
print("train size", train_df.shape)
print("test size", eval_df.shape)


######################## Create input prompt ################

labels = []

for i, emoji in id_to_emoji.items():
    labels.append(emoji)

def emoji_gen(df, split, num_samples=-1):

    instruction = f"""Predict an emoji of the given text. 
Possible answers are {labels}."""

    c = 0
    for index in range(len(df.index)):
        c +=1
        if c == num_samples:
            break
        title = str(df.TITLE[index])
        title = " ".join(title.split())
        label = df.ENCODE_CAT[index]

        label_emoji = id_to_emoji[label]

        if split == "train":
            prompt = f"""[INST]{instruction}[/INST]
### Input: {title}
### Answer: {label_emoji}"""

        else:  # input when evaluating
            prompt = f"""[INST]{instruction}[/INST]
### Input: {title}
### Answer:"""

        sample = {"idx": index,
                  "prompt": prompt,
                  "label": label,
                  "output": label_emoji,
                  "ori_id": index
                  }

        yield sample


# check input prompt
train_gen = emoji_gen(train_df, split="train")
for i in train_gen:
    print(i["prompt"])
    break

print("====" * 10)

train_gen = emoji_gen(eval_df, split="eval")
for i in train_gen:
    print(i["prompt"])
    break

################################ Load model ##############
mistral_finetune_dir = "mistral_finetune"
resume_full_model_path = "mistral_Biomed_2024-01-25_22-35-20_1/adapter"
base_model_for_adapter = "mistralai/Mistral-7B-Instruct-v0.1"
resume_model = os.path.join(mistral_finetune_dir, "mistral_Biomed_2024-01-26_07-58-17_7")

new_model_to_save = f"mistral_Biomed_{formatted_datetime}"

if use_clearml:
    task.set_user_properties(num_epoch=num_epoch,
                             gradient_accumulation_steps=gradient_accumulation_steps,
                             per_device_train_batch_size=per_device_train_batch_size,
                             resume=resume,
                             resume_full=resume_from_full_finetuned_model,
                             learning_rate=learning_rate,
                             logging_steps=logging_steps,
                             resume_model=resume_model,
                             resume_full_model=resume_full_model_path,
                             )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

home_directory = os.path.expanduser("~")
directory_path = os.path.join(home_directory, '../../media/data/')
model_path = "mistralai/Mistral-7B-Instruct-v0.1"
if use_clearml:
    task.set_user_properties(model_path=model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    add_eos_token=True,
    trust_remote_code=True,
    cache_dir=directory_path,
    local_files_only=True
)

tokenizer.padding = "longest"
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model = None

if resume_from_full_finetuned_model:
    model_path = os.path.join(directory_path, mistral_finetune_dir, resume_full_model_path)
    print("model_path", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 # cache_dir=directory_path,
                                                 local_files_only=True,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 return_dict=True,
                                                 # torch_dtype=torch.float16,
                                                 # load_in_4bit=False,
                                                 low_cpu_mem_usage=True
                                                 )

if merge_adapter_to_full_model:

    if base_model_for_adapter == "mistralai/Mistral-7B-Instruct-v0.1":
        model_path = base_model_for_adapter
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     cache_dir=directory_path,
                                                     local_files_only=True,
                                                     quantization_config=bnb_config,
                                                     device_map={"": 0},
                                                     return_dict=True)

    else:
        model_path = os.path.join(directory_path, mistral_finetune_dir, base_model_for_adapter)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     # cache_dir=directory_path,
                                                     local_files_only=True,
                                                     quantization_config=bnb_config,
                                                     device_map={"": 0},
                                                     return_dict=True,
                                                     torch_dtype=torch.float16,
                                                     # load_in_4bit=False,
                                                     low_cpu_mem_usage=True
                                                     )

    resume_model_path = os.path.join(directory_path, resume_model)
    # merged_model = PeftModel.from_pretrained(base_model, new_model_path)
    new_model_path = os.path.join(directory_path, mistral_finetune_dir, f"{new_model_to_save}_merged")

    merged_model = PeftModel.from_pretrained(model, resume_model_path)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(new_model_path, safe_serialization=True)  # safe_serialization=True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(new_model_path)
    print("saved tokenizer...")
    task.set_user_properties(new_model_path=new_model_path)
    print(new_model_path)
    sys.exit()

if resume:
    print("resume....")

    resume_model_path = os.path.join(directory_path, resume_model)

    config = PeftConfig.from_pretrained(resume_model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    if config.base_model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.1":
        model_path = base_model_for_adapter
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     cache_dir=directory_path,
                                                     local_files_only=True,
                                                     quantization_config=bnb_config,
                                                     device_map={"": 0},
                                                     return_dict=True)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    model = PeftModel.from_pretrained(
        model,
        resume_model_path,
        is_trainable=False,
    ).to(device)
    model = prepare_model_for_kbit_training(model)

if model is None:
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 cache_dir=directory_path,
                                                 local_files_only=True,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 return_dict=True)


def create_text_row(instruction, output, input):
    if output == "":
        # text_row = f"""<s>[INST] {instruction} here are the inputs {input} [/INST] \\n"""
        text_row = f"""[INST]{instruction}[/INST]

### {input}

### Answer:"""
    else:
        # text_row = f"""<s>[INST] {instruction} here are the inputs {input} [/INST] \\n {output} </s>"""
        text_row = f"""[INST]{instruction}[/INST]

### {input}

### Answer: [{output}]</s>"""
    return text_row


id2label = id_to_emoji
label2id = {v: k for k, v in id2label.items()}

if do_train:
    train_dataset = datasets.Dataset.from_generator(emoji_gen, gen_kwargs={"df": train_df, "split": "train", "num_samples": num_train})
    train_dataset = train_dataset.shuffle(seed=1234)
    train_data = train_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

if do_train or only_eval:
    eval_dataset = datasets.Dataset.from_generator(emoji_gen, gen_kwargs={"df": eval_df, "split": "eval", "num_samples": num_eval})

if do_train:
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


if do_train:
    modules = find_all_linear_names(model)
    print(modules)

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=modules, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        dataset_text_field="prompt",
        peft_config=lora_config,
        max_seq_length=max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0.03,
            num_train_epochs=1,
            max_steps=-1,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            output_dir=f"{directory_path}/outputs_mistral",
            optim="paged_adamw_8bit",
            save_strategy="no",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


def text_to_pred(text):
    for label_id, label_emoji in id_to_emoji.items():

        if label_emoji in text.lower():
            return label2id[label_emoji]
    return 1


def eval(model, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              cache_dir=directory_path,
                                              local_files_only=True, )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    total_count = 0
    correct = 0
    model.eval()

    preds = []
    wrong_pred_input = []
    wrong_pred_label = []
    for sample in dataset:
        prompt = sample["prompt"]
        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        total_count += 1

        model_inputs = encodeds.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=5, do_sample=True,
                                       pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids)

        pred_text = (decoded[0][-15:])
        print(pred_text)
        pred = text_to_pred(pred_text)
        ref = sample["label"]
        pred_name = id2label[pred]
        preds.append((sample["ori_id"], pred_name))
        print("pred:", pred)
        print("ref:", ref)
        if pred == ref:
            correct += 1
        elif ref != -1:
            wrong_pred_input.append(sample["idx"])
            wrong_pred_label.append(id2label[ref])
        print("===" * 10)

    print("correct:", correct)
    print("total:", total_count)
    acc = correct / total_count
    print("acc:", correct / total_count)

    return acc, preds, wrong_pred_input, wrong_pred_label


def save_model(epoch):
    print("save model")
    new_model_path = os.path.join(directory_path, mistral_finetune_dir, f"{new_model_to_save}_{epoch}")
    if use_clearml:
        task.set_user_properties(new_model_to_save=new_model_path)
    if os.path.exists(new_model_path):
        shutil.rmtree(new_model_path)
        print(f"Folder '{new_model_path}' removed.")

    trainer.model.save_pretrained(new_model_path)

    if not save_only_adapter:
        # this should be the model we resume not base model
        print("model_path", model_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          cache_dir=directory_path,
                                                          local_files_only=True,
                                                          # quantization_config=bnb_config,
                                                          device_map={"": 0},
                                                          return_dict=True,
                                                          low_cpu_mem_usage=True,
                                                          torch_dtype=torch.float16, )
        merged_model = PeftModel.from_pretrained(base_model, new_model_path)
        merged_model = merged_model.merge_and_unload()
        print("new_model_path", new_model_path)

        merged_model.save_pretrained(new_model_path, safe_serialization=True)  # safe_serialization=True
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.save_pretrained(new_model_path)
        print("saved tokenizer...")
        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        print("done..")


last_epoch = 0

if do_train:

    try:
        for i in range(num_epoch):
            trainer.train()
            loss = pd.DataFrame(trainer.state.log_history)
            if use_clearml:
                logger.report_table(title='loss', series=f'epoch={i}', iteration=i, table_plot=loss)

            if not do_not_eval:
                acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

                df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
                if use_clearml:
                    logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={i}', iteration=i,
                                        table_plot=df_eval)

                    logger.report_scalar("eval", "acc", iteration=i, value=acc)
            print("epoch", i)
            last_epoch = i
            save_model(last_epoch)


    except KeyboardInterrupt:
        print("Interrupt!!")
        # eval
        acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

        df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
        if use_clearml:
            logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={0}', iteration=0,
                                table_plot=df_eval)

            logger.report_scalar("eval", "acc", iteration=0, value=acc)

if only_eval:
    acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

    df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
    if use_clearml:
        logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={0}', iteration=0,
                            table_plot=df_eval)

        logger.report_scalar("eval", "acc", iteration=0, value=acc)

print(formatted_datetime)
sys.exit()
