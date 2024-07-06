from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
from typing import Tuple
from data import CarDataSet, split_df_labels
from model import (
    ResNetWithDualFC,
    EfficientNetWithDualFC,
    MobileNetWithDualFC,
    ViTWithDualFC,
)

writer = SummaryWriter("runs/")
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df, test_df = split_df_labels()

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = CarDataSet(
    car_df=train_df, images_dir="imgs", transform=train_transform
)
val_dataset = CarDataSet(car_df=val_df, images_dir="imgs", transform=val_test_transform)
test_dataset = CarDataSet(
    car_df=test_df, images_dir="imgs", transform=val_test_transform
)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE // 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE // 4,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0  # To accumulate the loss over the epoch
    n_total_steps = len(train_loader)
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 20 == 0:
            logging.info(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{n_total_steps}], Train loss: {loss.item():.4f}"
            )
            writer.add_scalar(
                "training loss granular",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
    average_loss = running_loss / len(train_loader)
    logging.info(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average train loss: {average_loss:.4f}"
    )
    writer.add_scalar("training loss", average_loss, epoch)
    return average_loss, epoch


def validate(
    val_loader: DataLoader, model: nn.Module, criterion: nn.Module, epoch: int
) -> Tuple[float, int]:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(val_loader)}], Val loss: {loss.item():.4f}"
                )
    average_loss = val_loss / len(val_loader)
    logging.info(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average validation loss: {average_loss:.4f}"
    )
    writer.add_scalar("validation loss", average_loss, epoch)
    return average_loss, epoch


def test(test_loader: DataLoader, model: nn.Module, metric_func: nn.Module) -> float:
    model.eval()
    metric_score = 0.0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            current_metric_score = metric_func(y_pred, y)
            metric_score += current_metric_score.item()
            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Batch [{batch_idx+1}/{len(test_loader)}], MSE score: {current_metric_score:.4f}"
                )
    average_metric_score = metric_score / len(test_loader)
    logging.info(f"Test average metric score: {average_metric_score:.4f}")
    return average_metric_score


def get_model(base_model_name: str) -> nn.Module:
    if base_model_name == "resnet18":
        return ResNetWithDualFC(freeze_backbone=True).to(DEVICE)
    elif base_model_name == "resnet50":
        return ResNetWithDualFC(base="resnet50", freeze_backbone=True).to(DEVICE)
    elif base_model_name == "resnet152":
        return ResNetWithDualFC(base="resnet152", freeze_backbone=True).to(DEVICE)
    elif base_model_name == "efficientnet":
        return EfficientNetWithDualFC(freeze_backbone=True).to(DEVICE)
    elif base_model_name == "mobilenet":
        return MobileNetWithDualFC(freeze_backbone=True).to(DEVICE)
    elif base_model_name == "vit":
        return ViTWithDualFC(freeze_backbone=True).to(DEVICE)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model on car images")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Base model for representation learning backbone (resnet18, resnet50, resnet152)",
    )
    args = parser.parse_args()
    base_model_name = args.model
    model = get_model(base_model_name)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    criterion = torch.nn.MSELoss(reduction="mean")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    lowest_val_loss = float("inf")
    best_epoch = 0
    logging.basicConfig = logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/log_{base_model_name}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(
        f"{'-' * 20} Base Mode for Representation Learning is {base_model_name} {'-' * 20}"
    )
    logging.info(f"{'-' * 20} Starting training {'-' * 20}")
    for epoch in range(NUM_EPOCHS):
        train_loss, epoch = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, epoch = validate(val_loader, model, criterion, epoch)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_epoch = epoch
            PATH = f"weights/best_{base_model_name}.pth"
            torch.save(model.state_dict(), PATH)
        lr_scheduler.step()
    logging.info(
        f"Best epoch {best_epoch+1}, Lowest validation loss: {lowest_val_loss:.4f}"
    )
    logging.info(f"{'-' * 20} Finished training {'-' * 20}")
    logging.info(f"{'-' * 20} Testing Time {'-' * 20}")
    model.load_state_dict(torch.load(PATH))
    test(test_loader, model, criterion)
    writer.close()


if __name__ == "__main__":
    main()
