import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def preprocess_agnews(
    data_type: str = "train",
    use_agnews_title: bool = False,
    train_size: float = 0.8,
    random_state: int = 42,
):
    datas = pd.read_csv("./dataset/{data_type}.csv")
    datas["Class Index"] = datas["Class Index"] - 1

    if data_type == "train":
        # Split train data into train and validation
        train_data, val_data = train_test_split(
            datas, train_size=train_size, random_state=random_state
        )

        train_label = train_data["Class Index"].tolist()
        val_label = val_data["Class Index"].tolist()

        if use_agnews_title:
            train_text = train_data["Title"] + " " + train_data["Description"]
            val_text = val_data["Title"] + " " + val_data["Description"]
            train_text = train_text.tolist()
            val_text = val_text.tolist()
        else:
            train_text = train_data["Description"].tolist()
            val_text = val_data["Description"].tolist()

        return train_text, train_label, val_text, val_label

    else:
        test_label = datas["Class Index"].tolist()
        if use_agnews_title:
            test_text = datas["Title"] + " " + datas["Description"]
            test_text = test_text.tolist()
        else:
            test_text = datas["Description"].tolist()

        return test_text, test_label
    
class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    