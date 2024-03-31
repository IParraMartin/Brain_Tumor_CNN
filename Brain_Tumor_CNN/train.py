import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import BrainTumorDataset
from model import TumorCNN
import torchvision.transforms as transforms

DATA_PATH = 'BRAIN_TUMOR_DATASET_PATH'
CSV_FILE = 'LABELS_CSV_PATH'
EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
SAVE_PATH = 'MODEL_SAVE_PATH'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader

def train_one_epoch(model, train_dataloader, test_dataloader, criterion, optimizer, device):
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        y_hat = model(inputs)
        loss = criterion(y_hat, targets)
        total_train_loss += loss.item()

        _, predicted = torch.max(y_hat.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy = 100 * correct_train / total_train
    train_loss = total_train_loss / len(train_dataloader)

    model.eval()
    total_test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            y_hat_test = model(inputs)
            test_loss = criterion(y_hat_test, targets)
            total_test_loss += test_loss.item()

            _, test_predicted = torch.max(y_hat_test.data, 1)
            total_test += targets.size(0)
            correct_test += (test_predicted == targets).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_loss = total_test_loss / len(test_dataloader)

    return train_loss, train_accuracy, test_loss, test_accuracy


def train(epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")

        (total_train_loss,
         train_accuracy,
         total_test_loss,
         test_accuracy) = train_one_epoch(model=model,
                                          train_dataloader=train_dataloader,
                                          test_dataloader=test_dataloader,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          device=device)

        print(f"Train L: {total_train_loss:.3f} - Train Acc: {train_accuracy:.3f} || Test L: {total_test_loss:.3f} - Test Acc: {test_accuracy:.3f}")
        print("--------------------------------------------------------------------------------------------------------------------")

    print("Training finished.")


if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.2484], [0.2341])
    ])

    dataset = BrainTumorDataset(csv_file=CSV_FILE,
                                root_dir=DATA_PATH,
                                transform=image_transform)
    train_size = 200
    test_size = 53
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = get_dataloader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = get_dataloader(test_set, batch_size=BATCH_SIZE)

    cnn_model = TumorCNN(in_channels=1, num_filters=8, classes=2).to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    train(
        epochs=EPOCHS,
        model= cnn_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=loss_criterion,
        optimizer=optim,
        device=device
    )

    torch.save(cnn_model.state_dict(), SAVE_PATH)
