import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, PretrainedConfig
from label_map import id_to_emoji

from torch import cuda

import torch
torch.manual_seed(0)
device = torch.device('cuda:0') if cuda.is_available() else 'cpu'

verbose = False
train_size = 0.8
test_size = 1


def read_text(text_path):
    #text_path = "data/us_test.text"
    text_list = []
    with open(text_path, "r") as text_file:
        for line in text_file:
            line = line.strip()
            text_list.append(line)
    
    print(len(text_list))
    return text_list


def read_label(label_path):

    #label_path = "data/us_test.labels"
    label_list = []
    with open(label_path, "r") as label_file:
        for line in label_file:
            line = line.strip()
            line = int(line)
            label_list.append(line)
    
    print(len(label_list))
    print(label_list[:2])
    
    print("set",set(label_list))
    return label_list




train_text_list = read_text("data/us_test.text")
train_label_list = read_label("data/us_test.labels")

train_data_dict = {'TITLE': train_text_list, 'CATEGORY': train_label_list, 'ENCODE_CAT': train_label_list}
train_df = pd.DataFrame(data=train_data_dict)

train_df_ = train_df.sample(frac=train_size, random_state=200)
test_df = train_df.drop(train_df_.index).reset_index(drop=True)
train_df = train_df_.reset_index(drop=True)

# test_text_list = read_text("data/us_trial.text")
# test_label_list = read_label("data/us_trial.labels")
#
# test_data_dict = {'TITLE': test_text_list, 'CATEGORY': test_label_list, 'ENCODE_CAT': test_label_list}
# test_df = pd.DataFrame(data=test_data_dict)

########




import argparse



parser = argparse.ArgumentParser(description='Set TRAIN_BATCH_SIZE to 56')

# Add argument for TRAIN_BATCH_SIZE
parser.add_argument('--batch_size', type=int, default=56, help='Training batch size (default: 56)')

args = parser.parse_args()

# Defining some key variables that will be used later on in the training
MAX_LEN = 300
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.batch_size
EPOCHS = 15
LEARNING_RATE = 1e-05


import os

home_directory = os.path.expanduser("~")
directory_path = os.path.join(home_directory, '../../media/data/models')
model_name = "bert-base-multilingual-cased"

model_path = os.path.join(directory_path, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
bert_feature_size = model_config.hidden_size
# distilbert-base-cased
num_classes = 20


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(title, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', return_token_type_ids=True, truncation=True)

        # inputs = self.tokenizer.encode_plus(title, None, add_special_tokens=True, max_length=None,
        #     padding='longest', return_token_type_ids=True, truncation=False)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long),
                'text': title
                }

    def __len__(self):
        return self.len


# Creating the dataset and dataloader for the neural network


# train_size = 0.1
# train_dataset = train_df.sample(frac=train_size, random_state=200)
# #test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
# train_dataset = train_dataset.reset_index(drop=True)

#train_size = 0.01
# train_df = train_df.sample(frac=train_size, random_state=200)
# #test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
# train_df = train_df.reset_index(drop=True)
#
# #test_size = 0.01
# test_df = test_df.sample(frac=test_size, random_state=200)
# #test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
# test_df = test_df.reset_index(drop=True)

# # print("FULL Dataset: {}".format(train_df.shape))
# print("TRAIN Dataset: {}".format(train_dataset.shape))
# print("TEST Dataset: {}".format(test_dataset.shape))
###



training_set = Triage(train_df, tokenizer, MAX_LEN)
testing_set = Triage(test_df, tokenizer, MAX_LEN)

print("train size", len(training_set))
print("test size", len(testing_set))

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output
# for the model.

class BERTClass(torch.nn.Module):
    def __init__(self, num_classes, model_name):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_path,local_files_only=True)
        # self.pre_classifier = torch.nn.Linear(bert_feature_size, 768)
        # self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(bert_feature_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=False) #
        #hidden_state = output_1[0]
        #pooler = hidden_state[:, 0] # torch.Size([batch, 768])

        # print(len(output_1)) # 3
        # print(output_1.hidden_states[-1].shape) # (batch, seqlen, embed dim)
        # pooler = output_1.hidden_states[-1][:,0,:] # get cls from last hidden state
        # print(pooler[0][:10])
        pooler = output_1.pooler_output
        #print(pooler[0][:10])
        #print("....")
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.dropout(pooler)

        output = self.classifier(pooler)
        return output


class BERTClass(torch.nn.Module):
    def __init__(self, num_classes, model_name):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_path,local_files_only=True)
        self.pre_classifier = torch.nn.Linear(bert_feature_size, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids,
                           attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0] # torch.Size([batch, 768])
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


model = BERTClass(num_classes, model_name)
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# Function to calcuate the accuracy of the model

def calculate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model

import sys, os

print(id_to_emoji)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        # ids  = data['ids']
        # print(len(data['ids']))
        # sys.exit()
        # print(data['ids'])
        # print(data['targets'].shape)

        # ids = torch.randn((4, 512)).long().to(device)
        # mask = torch.randn((4, 512)).long().to(device)
        # targets = torch.arange(20)[:4].long().to(device)

        # print(ids.shape)

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)


        big_idx = big_idx.to("cpu")
        targets = targets.to("cpu")
        if verbose:
            for i in range(len(big_idx)):
                # print(big_idx[i].item())
                print("text:", data["text"][i])
                print("prediction:", id_to_emoji[big_idx[i].item()])
                print("label:", id_to_emoji[targets[i].item()])
                print("....")

        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


for epoch in range(EPOCHS):
    train(epoch)

# Saving the files for re-use



# output_model_file = './models/pytorch_distilbert_news.bin'
# output_vocab_file = './models/vocab_distilbert_news.bin'
#
# model_to_save = model
# torch.save(model_to_save, output_model_file)
# tokenizer.save_vocabulary(output_vocab_file)

# print('All files saved')
# print('This tutorial is completed')


def valid(model, testing_loader):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0  # Initialize tr_loss
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()  # Update tr_loss with the current loss
            big_val, big_idx = torch.max(outputs.data, dim=1)

            big_idx = big_idx.to("cpu")
            targets = targets.to("cpu")
            if verbose:
                for i in range(len(big_idx)):
                    # print(big_idx[i].item())
                    print("text:", data["text"][i])
                    print("prediction:", id_to_emoji[big_idx[i].item()])
                    print("label:", id_to_emoji[targets[i].item()])
                    print("....")

            n_correct += calculate_accu(big_idx, targets)  # Fix the typo

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu


print('Evaluation....')

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)
