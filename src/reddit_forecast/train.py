import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Ensure the text is valid
        text = row["text"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError(f"Invalid text at index {index}: {text}")

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label = 1 if row["sentiment"] == "POSITIVE" else 0
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label,
        }



def train():
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Load dataset and dataloader
    print("Loading dataset...")
    dataset = SentimentDataset("models/processed_posts_with_sentiment.csv", tokenizer)
    print(f"Dataset loaded with {len(dataset)} samples.")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("Dataloader initialized.")

    # Define loss function and optimizer
    print("Initializing optimizer and loss function...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    epochs = 3
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}...")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    model.save_pretrained("models/fine_tuned_distilbert")
    tokenizer.save_pretrained("models/fine_tuned_distilbert")
    print("Model and tokenizer saved!")



if __name__ == "__main__":
    train()
