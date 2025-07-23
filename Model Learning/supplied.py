import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, random_split

def load_mnist_training_validation(percentage_validation, batch_size):

    # Define transformations (convert images to tensors and normalize them)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])
    
    # Load full training dataset
    full_train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    
    # Define train-validation split (e.g., 90% train, 10% validation)
    percentage_training = 1-percentage_validation
    train_size = int(percentage_training * len(full_train_dataset))  # 90% of the data
    val_size = len(full_train_dataset) - train_size  # Remaining 10% for validation
    print(f"{train_size} images for training  and {val_size} images for validation")
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_mnist_test(batch_size=64):
    # Define transformations (convert images to tensors and normalize them)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])

    # Load test dataset
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def train_model(model, train_loader, test_loader, num_epochs, optimizer,loss_fn,device):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            # Move data to the correct device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = loss_fn(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print the training statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Validation loop (optional, to check model performance on validation data)
        model.eval()  # Set the model to evaluation mode
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

    print("Training complete.")
    return model