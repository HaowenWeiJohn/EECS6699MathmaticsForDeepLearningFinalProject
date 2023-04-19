import torch

def test(model, test_loader, criterion, device):
    # set the model to evaluation mode
    model.eval()

    # initialize variables for computing performance metrics
    test_loss = 0
    correct = 0
    total = 0

    # turn off gradients for efficiency
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # update performance metrics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(axis=1)).sum().item()

    # compute average loss and accuracy
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / total

    return avg_loss, accuracy