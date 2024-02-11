# CUDA_VISIBLE_DEVICES=1 python mistral2.py


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from clearml import Task
import datasets

task = Task.init(project_name='emoji', task_name='mistral train', reuse_last_task_id=False)
task.add_tags("mock")
logger = task.get_logger()
from datetime import datetime
import os

# Get the current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

do_train = True
# only_eval = True
do_test = False  # do test when interrupt
do_not_eval = False
only_eval = False #not do_train
only_save = False
resume_from_full_finetuned_model = False
resume = False
merge_adapter_to_full_model = False  # specify resume path
save_only_adapter= True

num_epoch = 10
train_num = 1#""
test_num = 0
test_id_start = 0
test_id_end = 1
test_split = f"test[{test_id_start}:{test_id_end}]"
eval_num = ""
new_data_path = None #"list_single_300.json"

trial_token_len = -1
gradient_accumulation_steps = 1
per_device_train_batch_size = 60

max_seq_length = 1000
# ori 2e-4
#
learning_rate = 2e-4
logging_steps = 0.25
device = "cuda"
#########


import pandas as pd

def read_text(text_path):
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

if do_test:
    test_text_list = read_text("data/us_trial.text")
    test_label_list = read_label("data/us_trial.labels")

    test_data_dict = {'TITLE': test_text_list, 'CATEGORY': test_label_list, 'ENCODE_CAT': test_label_list}
    test_df = pd.DataFrame(data=test_data_dict)
from label_map import id_to_emoji

train_size = 0.8
test_size = 0.01

train_df_ = train_df.sample(frac=train_size, random_state=200)
#test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)

eval_df = train_df.drop(train_df_.index).reset_index(drop=True)


train_df = train_df_.reset_index(drop=True)
print("train size", train_df.shape)
print("test size", eval_df.shape)

if do_test:
    test_df = test_df.sample(frac=test_size, random_state=200)
    #test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

def emoji_gen(df, split="train"):
    labels = []

    for i, emoji in id_to_emoji.items():
        labels.append(emoji)
        #labels.append((i, emoji))

    instruction = f"""Predict an emoji of the given text. 
Possible answers are {labels}."""

    for index in range(len(df.index)):
        title = str(df.TITLE[index])
        title = " ".join(title.split())
        label = df.ENCODE_CAT[index]

        label_emoji = id_to_emoji[label]

        if split == "train":
            prompt = f"""[INST]{instruction}[/INST]
### Input: {title}
### Answer: {label_emoji}"""

        else:
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


train_gen = emoji_gen(train_df, split="train")
for i in train_gen:
    print(i["prompt"])
    break

print("===="*10)


train_gen = emoji_gen(eval_df, split="eval")
for i in train_gen:
    print(i["prompt"])
    break


#######
mistral_finetune_dir = "mistral_finetune"

# merged model mistralai-Instruct-Finetune-Biomed_2024-01-24_18-05-59_333
# Instruct-Finetune-Biomed_2024-01-24_20-23-52_0 acc73
# mistralai-Instruct-Finetune-Biomed_2024-01-24_20-16-04_merged  (high acc)
# mistralai-Instruct-Finetune-Biomed_2024-01-24_20-23-52_0

# mistralai-Instruct-Finetune-Biomed_acc69_full_work
# mistralai-Instruct-Finetune-Biomed_2024-01-24_23-26-29_0

# try saving this: mistralai-Instruct-Finetune-Biomed_2024-01-25_04-15-13_0  (can not save)

# mistralai-Instruct-Finetune-Biomed_2024-01-24_23-26-29_0 (can not save)
# mistralai-Instruct-Finetune-Biomed_2024-01-25_05-36-49_0 (saved from 69, works and can be saved)

#mistral_Biomed_2024-01-25_22-35-20_0
#mistralai-Instruct-Finetune-Biomed_2024-01-25_12-06-32_2
resume_full_model_path = "mistral_Biomed_2024-01-25_22-35-20_1/adapter"
base_model_for_adapter = "mistralai/Mistral-7B-Instruct-v0.1"
resume_model = os.path.join(mistral_finetune_dir, "mistral_Biomed_2024-01-26_07-58-17_7")

new_model_to_save = f"mistral_Biomed_{formatted_datetime}"
# resume_model = os.path.join(mistral_finetune_dir,"mistralai-Instruct-Finetune-Biomed_2024-01-23_20-28-14_11")

# mistralai-Instruct-Finetune-Biomed_2024-01-23_20-28-14_0

# resume_model = os.path.join(mistral_finetune_dir,"mistralai-Instruct-Finetune-Biomed_2024-01-24_02-20-48_one_epoch_0")
# resume_model = os.path.join(mistral_finetune_dir, f"{new_model_to_save}_{5}")
task.set_user_properties(num_epoch=num_epoch,
                         num_train=train_num,
                         num_test=test_num,
                         num_eval=eval_num,
                         trial_token_len=trial_token_len,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         per_device_train_batch_size=per_device_train_batch_size,
                         resume=resume,
                         resume_full=resume_from_full_finetuned_model,
                         learning_rate=learning_rate,
                         logging_steps=logging_steps,
                         resume_model=resume_model,
                         resume_full_model=resume_full_model_path,
                         # new_model_to_save= new_model_to_save
                         )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

import os

home_directory = os.path.expanduser("~")
directory_path = os.path.join(home_directory, '../../media/data/')
model_path = "mistralai/Mistral-7B-Instruct-v0.1"
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
                                                 #cache_dir=directory_path,
                                                 local_files_only=True,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 return_dict=True,
                                                 #torch_dtype=torch.float16,
                                                 # load_in_4bit=False,
                                                 low_cpu_mem_usage=True
                                                 )

from peft import LoraConfig, PeftModel

if merge_adapter_to_full_model:

    if base_model_for_adapter == "mistralai/Mistral-7B-Instruct-v0.1":
        model_path = base_model_for_adapter
        # ori
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     cache_dir=directory_path,
                                                     local_files_only=True,
                                                     quantization_config=bnb_config,
                                                     device_map={"": 0},
                                                     return_dict=True)

    else:
        model_path = os.path.join(directory_path, mistral_finetune_dir, base_model_for_adapter)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                             #cache_dir=directory_path,
                                             local_files_only=True,
                                             quantization_config=bnb_config,
                                             device_map={"": 0},
                                             return_dict=True,
                                             torch_dtype=torch.float16,
                                             # load_in_4bit=False,
                                             low_cpu_mem_usage=True
                                             )

    # base_model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                                   cache_dir=directory_path,
    #                                                   local_files_only=True,
    #                                                   device_map={"": 0},
    #                                                   return_dict=True,
    #                                                   low_cpu_mem_usage=True,
    #                                                   torch_dtype=torch.float16)

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
    ###

    # resume_model_path = os.path.join(directory_path, resume_model)
    # #offload_folder
    # model = PeftModel.from_pretrained(model, resume_model_path)
    # model = model.merge_and_unload()
    # print("...save model ...")
    # model.save_pretrained(new_model_path)  ## works
    # #new_model_path = os.path.join(directory_path, mistral_finetune_dir, f"{new_model_to_save}_{333}")
    # tokenizer.save_pretrained(new_model_path)
    # print("...save tokenizer ...")

    task.set_user_properties(new_model_path=new_model_path)
    print(new_model_path)
    import sys

    sys.exit()

if resume:
    print("resume....")
    from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training

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
        # ori
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

    # not work
    if False:
        if base_model_for_adapter == "mistralai/Mistral-7B-Instruct-v0.1":
            model_path = base_model_for_adapter
            # ori
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         cache_dir=directory_path,
                                                         local_files_only=True,
                                                         quantization_config=bnb_config,
                                                         device_map={"": 0},
                                                         return_dict=True)

        else:
            model_path = os.path.join(directory_path, mistral_finetune_dir, base_model_for_adapter)
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 #cache_dir=directory_path,
                                                 local_files_only=True,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 return_dict=True,
                                                 #torch_dtype=torch.float16,
                                                 # load_in_4bit=False,
                                                 low_cpu_mem_usage=True
                                                 )

        resume_model_path = os.path.join(directory_path, resume_model)
        # offload_folder
        model = PeftModel.from_pretrained(model, resume_model_path, is_trainable=True)
        model = model.merge_and_unload()
        print("....peft or not", type(
            model))  # new_model_path = os.path.join(directory_path, mistral_finetune_dir, f"{new_model_to_save}_{333}")

        # model.save_pretrained(new_model_path) ## works  # transformers.models.mistral.modeling_mistral.MistralForCausalLM

if model is None:
    # ori
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 cache_dir=directory_path,
                                                 local_files_only=True,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 return_dict=True)

from datasets import load_dataset


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
    train_dataset = datasets.Dataset.from_generator(emoji_gen, gen_kwargs={"df": train_df})
    # train_dataset = datasets.Dataset.from_generator(train_gen)
    train_dataset = train_dataset.shuffle(seed=1234)  # Shuffle dataset here

    print("len train + new", len(train_dataset))
    task.set_user_properties(len_train=len(train_dataset))

if do_train or only_eval:
    eval_dataset = datasets.Dataset.from_generator(emoji_gen, gen_kwargs={"df": eval_df, "split": "eval"})
    task.set_user_properties(len_eval=len(eval_dataset))

if do_test:
    test_dataset = datasets.Dataset.from_generator(emoji_gen, gen_kwargs={"df": test_df, "split": "eval"})
    task.set_user_properties(len_test=len(test_dataset))


# dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")

# df = dataset.to_pandas()
# print(df.head(10))




# add the "prompt" column in the dataset
# text_column = [generate_prompt(data_point) for data_point in dataset]
# dataset = dataset.add_column("prompt", text_column)

if do_train:
    train_data = train_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
# eval_data = eval_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

# train_data = train_dataset.map(tokenize_prompt, batched=True)
# eval_data = eval_dataset.map(tokenize_prompt, batched=True)


# test_data = eval_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

# dataset = dataset.train_test_split(test_size=0.2)
# train_data = dataset["train"]
# test_data = dataset["test"]


# print(test_data)


from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
if do_train:
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

# print(model)

import bitsandbytes as bnb


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

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=modules, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM")

    # MistralForCausalLM' object has no attribute 'get_nb_trainable_parameters'

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")

    # new code using SFTTrainer
    import transformers

    from trl import SFTTrainer

    # torch.cuda.empty_cache()

    # for i in train_data:
    #     print(i["prompt"])
    #     logger.report_text(i["prompt"])
    #     break


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=eval_data,
        # compute_metrics=compute_metrics,
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
            # evaluation_strategy="epoch", ##
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

# for batch in trainer.get_eval_dataloader(eval_data):
#     break

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


# trainer.train()

def text_to_pred(text):

    for label_id, label_emoji in id_to_emoji.items():

        if label_emoji in text.lower():
            return label2id[label_emoji]
    return 1




from torch.utils.data import DataLoader


def eval(model, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              cache_dir=directory_path,
                                              local_files_only=True, )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # todo

    # eval_dataloader = DataLoader(eval_dataset, batch_size=5 ,shuffle=False)
    total_count = 0
    correct = 0
    model.eval()

    preds = []
    wrong_pred_input = []
    wrong_pred_label = []
    for sample in dataset:
        prompt = sample["prompt"]
        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # print(encodeds)
        total_count += 1

        model_inputs = encodeds.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=5, do_sample=True,
                                       pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids)
        # logits = model(generated_ids).logits.to("cpu")
        # print("...logits", logits)
        # logits = generated_ids.scores.to("cpu") # not work
        # print(type(logits))
        # print(logits.shape)
        # print("...logits", logits)

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


import os

import pandas as pd
# Create a folder with the formatted datetime as its name
# folder_name = f"predictions_{formatted_datetime}"
# os.makedirs(folder_name)
import shutil
import gc


def save_model(epoch):
    # save model
    print("save model")
    # import shutil
    #
    # new_model = os.path.join(directory_path, new_model_to_save)
    # task.set_user_properties(new_model=new_model)
    #
    # if os.path.exists(new_model):
    #     # Remove the folder
    #     shutil.rmtree(new_model)
    #     print(f"Folder '{new_model}' removed.")

    new_model_path = os.path.join(directory_path, mistral_finetune_dir, f"{new_model_to_save}_{epoch}")
    task.set_user_properties(new_model_to_save=new_model_path)
    if os.path.exists(new_model_path):
        # Remove the folder
        shutil.rmtree(new_model_path)
        print(f"Folder '{new_model_path}' removed.")

    trainer.model.save_pretrained(new_model_path)

    if not save_only_adapter:
        # model.save_pretrained(new_model_path)

        # base_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, return_dict=True,
        #     torch_dtype=torch.float16, device_map={"": 0}, )

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
        # base_model = None

        # merged_model = merged_model.merge() #_and_unload()

        # Save the merged model
        print("new_model_path", new_model_path)

        merged_model.save_pretrained(new_model_path, safe_serialization=True)  # safe_serialization=True
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.save_pretrained(new_model_path)
        print("saved tokenizer...")
        # merged_model = None
        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        print("done..")


last_epoch = 0

from torch.utils.data import ConcatDataset

if only_save:
    save_model(333)

if do_train:

    try:
        for i in range(num_epoch):
            trainer.train()
            loss = pd.DataFrame(trainer.state.log_history)
            logger.report_table(title='loss', series=f'epoch={i}', iteration=i, table_plot=loss)

            if not do_not_eval:
                # frames.append(loss)
                acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

                df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
                logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={i}', iteration=i,
                                    table_plot=df_eval)

                logger.report_scalar("eval", "acc", iteration=i, value=acc)
            print("epoch", i)
            last_epoch = i
            #save_model(last_epoch)


    except KeyboardInterrupt:
        print("Interrupt!!")
        # eval
        acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

        df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
        logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={0}', iteration=0,
                            table_plot=df_eval)

        logger.report_scalar("eval", "acc", iteration=0, value=acc)

if only_eval:
    acc, _, wrong_pred_input, wrong_pred_label = eval(model=model, dataset=eval_dataset)

    df_eval = pd.DataFrame({'idx': wrong_pred_input, 'label': wrong_pred_label})
    logger.report_table(title='wrongly predicted samples from dev', series=f'epoch={0}', iteration=0,
                        table_plot=df_eval)

    logger.report_scalar("eval", "acc", iteration=0, value=acc)

if do_test:
    _, preds, _, _ = eval(model=model, dataset=test_dataset)

    # Use zip to transpose the list of tuples
    ids, pred, text = list(zip(*preds))

    df_eval = pd.DataFrame({'idx': ids, 'pred': pred, 'text': text})
    logger.report_table(title='prediction on test set', series=f'epoch={0}', iteration=0, table_plot=df_eval)


print(formatted_datetime)
import sys

sys.exit()
