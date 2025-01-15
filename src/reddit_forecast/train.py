import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("Training sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded on {device}.")

    # Load dataset and dataloader
    logger.info("Loading dataset and initializing dataloader...")
    dataset = SentimentDataset("models/processed_posts_with_sentiment.csv", tokenizer)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    logger.info("Dataloader initialized.")

    # Define loss function and optimizer
    logger.info("Defining loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    # Training loop
    epochs = 3
    accumulation_steps = 2  # Accumulate gradients over multiple steps
    logger.info(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"Processing batch {batch_idx + 1}...")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps  # Normalize loss for accumulation

            scaler.scale(loss).backward()

            # Perform optimizer step after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Save the fine-tuned model
    logger.info("Saving fine-tuned model and tokenizer...")
    model.save_pretrained("models/fine_tuned_distilbert")
    tokenizer.save_pretrained("models/fine_tuned_distilbert")
    logger.info("Model and tokenizer saved!")


if __name__ == "__main__":
    train()
