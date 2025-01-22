import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pytorch_lightning as pl
import optuna
from optuna.storages import RDBStorage
import pandas as pd
import functools
import hydra
import os


def seed_randoms(cfg):
    seed = cfg.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, cfg):
        max_length = cfg.train.max_length
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }



class SentimentRegressionModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, l2):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.regression_head = nn.Linear(
            self.model.config.hidden_size, 1
        )  # Single output
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.l2 = l2

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
        logits = self.regression_head(pooled_output)
        return logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        return optimizer


def objective(trial, train_dataset, val_dataset, cfg):
    # Set random seed for reproducibility
    seed_randoms(cfg)

    # Define hyperparameter search space
    model_name = "distilbert-base-uncased"

    # Hyperparameters to tune
    learning_rate = trial.suggest_float(
        "learning_rate", cfg.train.lr_min, cfg.train.lr_max, log=True
    )
    l2 = trial.suggest_float("l2", cfg.train.l2_min, cfg.train.l2_max, log=True)
    batch_size = trial.suggest_categorical("batch_size", [cfg.train.batch_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = SentimentRegressionModel(model_name, learning_rate, l2)

    # Trainer for Optuna
    trainer = pl.Trainer(
        max_epochs=cfg.train.optuna_epoch,
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices=1 if torch.cuda.is_available() else 0,
        logger=False,
        enable_checkpointing=False,
    )

    # Train the model and return the validation loss
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_loss"].item()


@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg):
    # Set up persistent storage for Optuna study
    storage = RDBStorage("sqlite:///optuna_study.db")
    study = optuna.create_study(
        storage=storage,
        study_name="models/sentiment_tuning",
        direction="minimize",
        load_if_exists=True,
    )

    model_name = "distilbert-base-uncased"
    dataset_path = cfg.train.data_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = SentimentDataset(dataset_path, tokenizer, cfg)
    train_size = int(cfg.train.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Use functools.partial to pass arguments
    partial_objective = functools.partial(
        objective, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg
    )

    # Run optimization
    study.optimize(
        partial_objective, n_trials=cfg.train.n_trials
    )  # Number of trials for tuning

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Train the final model with the best hyperparameters
    best_params = study.best_trial.params

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataset = SentimentDataset(dataset_path, tokenizer)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=int(best_params["batch_size"]), shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=int(best_params["batch_size"]))

    final_model = SentimentRegressionModel(
        model_name, learning_rate=best_params["learning_rate"], l2=best_params["l2"]
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.final_epoch,  # Train longer for the final model
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(final_model, train_loader, val_loader)
    final_model.model.save_pretrained("models/sentiment_model_finetuned")
    final_model.tokenizer.save_pretrained("models/sentiment_model_finetuned")


if __name__ == "__main__":
    main()
