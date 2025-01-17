import torch
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


def evaluate(model, test_loader, device, output_path="evaluation_metrics"):
    """Evaluate the model and save evaluation metrics."""
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(logits.cpu().numpy())

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)

    print(f"Evaluation Results - MSE: {mse:.4f}, R^2: {r2:.4f}")

    # Save metrics to a file
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.txt"

    with open(metrics_file, "w") as f:
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"R-squared: {r2:.4f}\n")

    return all_labels, all_predictions
