import torch
import matplotlib.pyplot as plt
import sys
import torch
import tqdm

# Model training function
def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        (label, ids, length) = batch
        label = label.to(device).float()
        ids = ids.to(device)
        length = length.to(device)
        
        prediction = model(ids, length)
        loss = criterion(prediction.squeeze(), label)
        accuracy = get_accuracy(prediction.squeeze(), label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

# Model evaluation function
def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            (label, ids, length) = batch
            label = label.to(device).float()
            ids = ids.to(device)
            length = length.to(device)
            
            prediction = model(ids, length)
            loss = criterion(prediction.squeeze(), label)
            accuracy = get_accuracy(prediction.squeeze(), label.float())
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

# Accuracy calculation for binary classification
def get_accuracy(prediction, label, threshold=0.5):
    predicted_classes = (prediction >= threshold).long()  # Binary classification using threshold
    correct_predictions = predicted_classes.eq(label).sum().item()
    accuracy = correct_predictions / len(label)
    return torch.tensor(accuracy)  # Ensure accuracy is returned as a tensor

def collate_batch(batch):
    label_list, text_list, length_list = [], [], []

    for (text, label, length) in batch:
        text_list.append(text)
        label_list.append(label)
        length_list.append(length)

    label_list = torch.stack(label_list, dim=0)
    text_list = torch.stack(text_list, dim=0)
    length_list = torch.tensor(length_list, dtype=torch.int64)

    return label_list, text_list, length_list


def plot_losses(train_losses, valid_losses, output_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f'{output_dir}/loss_epoch_plot.png')
    plt.close()