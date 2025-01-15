import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset


class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path, usecols=["Sentence", "Sentiment"])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"positive": 1.0, "negative": 0.0, "neutral": 0.5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["Sentence"]
        sentiment = row["Sentiment"]
        label = self.label_map[sentiment]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train():
    # logger.info("Training sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )
    model.classifier = nn.Linear(
        model.config.hidden_size, 1
    )  # Single output for regression
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model loaded on {device}.")
    model.to(device)
    # logger.info(f"Model loaded on {device}.")

    # Load dataset
    print("Loading dataset...")
    dataset = SentimentDataset("data/raw/data.csv", tokenizer)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print("Dataloaders for train, validation, and test initialized.")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(3):  # Example for 3 epochs
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)  # Single output for regression
            loss = criterion(logits, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training complete.")

    # Evaluation
    model.eval()
    mse_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            mse_loss += criterion(logits, labels.float()).item()

    print(f"Test MSE Loss: {mse_loss / len(test_loader):.4f}")
    # save model
    model.save_pretrained("models/sentiment_model_finetuned")


if __name__ == "__main__":
    train()
