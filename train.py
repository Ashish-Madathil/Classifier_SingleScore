import torch
import torch.optim as optim
from Classifier import Classifier
import torch.nn as nn
from load_data import train_loader, val_loader

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 100



# Initialize Model and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_classes = 5
true_positives = torch.zeros(num_classes)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    true_positives.fill_(0)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            for i in range(num_classes):
                true_positives[i] += (predicted == labels).logical_and(predicted == i).sum().item()
                total_per_class[i] += (labels == i).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100. * correct / len(val_loader.dataset):.2f}%")
    for i in range(num_classes):
        print(f"Class {i} Recall: {true_positives[i] / total_per_class[i]:.2f}")
        print(f"Class {i} True Positives: {true_positives[i]}/{total_per_class[i]}")

# Save model
torch.save(model.state_dict(), f'model.pth')
print('Finished Training. Model Saved')    
