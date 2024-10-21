import torch
import matplotlib.pyplot as plt
import sys
import tqdm

# Model training function
def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    
    for (encoded_sent1, encoded_sent2), scores, (length1, length2) in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        encoded_sent1 = {k: v.to(device) for k, v in encoded_sent1.items()}
        encoded_sent2 = {k: v.to(device) for k, v in encoded_sent2.items()}
        scores = scores.to(device)
        
        optimizer.zero_grad()
        lengths = torch.min(length1, length2).to(device)  # Use the shorter of the two for packing
        prediction = model(encoded_sent1['input_ids'], encoded_sent2['input_ids'], lengths)
        loss = criterion(prediction.squeeze(), scores)
        accuracy = get_accuracy(prediction.squeeze(), scores.float())
        
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
        for (encoded_sent1, encoded_sent2), scores, (length1, length2) in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            encoded_sent1 = {k: v.to(device) for k, v in encoded_sent1.items()}
            encoded_sent2 = {k: v.to(device) for k, v in encoded_sent2.items()}
            scores = scores.to(device)
            
            lengths = torch.min(length1, length2).to(device)  # Use the shorter of the two for packing
            prediction = model(encoded_sent1['input_ids'], encoded_sent2['input_ids'], lengths)
            loss = criterion(prediction.squeeze(), scores)
            accuracy = get_accuracy(prediction.squeeze(), scores.float())
            
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            
    return epoch_losses, epoch_accs


# Accuracy calculation for regression (using similarity threshold)
def get_accuracy(prediction, score, threshold=0.5):
    diff = torch.abs(prediction - score)
    correct = (diff < threshold).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

# Plot training and validation losses
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
