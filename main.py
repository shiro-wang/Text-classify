import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from preprocessing import preprocess_agnews, AGNewsDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from save_result import save_result, save_logger


# BATCH_SIZE = 16
# EPOCH = 1
# LR = 0.0001
# MAX_LENGTH = 512
parser = argparse.ArgumentParser()
# parser.add_argument("--data_name", type=str, default="agnews")
parser.add_argument("--use_agnews_title", action="store_true")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--random", type=int, default=74)

args = parser.parse_args()

if args.data_name == "agnews":
    preprocessing_func = preprocess_agnews

train_text, train_label, val_text, val_label = preprocessing_func(
    data_name=args.data_name,
    data_type="train",
    use_agnews_title=args.use_agnews_title,
    random_state=args.random,
)
test_text, test_label = preprocessing_func(
    args.data_name,
    data_type="test",
    use_agnews_title=args.use_agnews_title,
)
num_labels = len(set(train_label))

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_encodings = tokenizer(
    train_text, truncation=True, padding=True, max_length=args.max_length
)
val_encodings = tokenizer(
    val_text, truncation=True, padding=True, max_length=args.max_length
)
test_encodings = tokenizer(
    test_text, truncation=True, padding=True, max_length=args.max_length
)

train_dataset = AGNewsDataset(train_encodings, train_label)
val_dataset = AGNewsDataset(val_encodings, val_label)
test_dataset = AGNewsDataset(test_encodings, test_label)


train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=args.test_batch_size, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

train_history = []
val_history = []
for epoch in tqdm(range(args.num_epoch)):
    train_loss = 0.
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        # input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # labels = batch['labels'].to(device)
        # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        train_history.append(loss.item())
        train_loss += loss.item()
        optimizer.step()
    
    val_loss = 0.
    model.eval()
    for batch in tqdm(val_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        val_history.append(loss.item())
        val_loss += loss.item()

    print(f"[epoch {epoch+1}] train loss: {train_loss/len(train_loader)}, valid loss: {val_loss/len(val_loader)}")

model.eval()
with torch.no_grad():
    y_preds = []
    for batch in tqdm(test_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        # input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # labels = batch['labels'].to(device)
        # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs[1], 1)
        y_preds.extend(predicted.tolist())
save_result(args, "Training loss", train_history)
acc = accuracy_score(test_label, y_preds)
save_logger(args, acc)
# print(f"Acc: {accuracy_score(test_label, y_preds)}")

