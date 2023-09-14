
# External imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import csv
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import matplotlib.pyplot as plt
# import sys
import logging

# Local imports
from models import *
from stratified_sampling import split_indices
from embryo_dataset import EmbryoDataset


# Hyperparameters
learning_rate = 0.00001
lower_lr = 0.000001
batch_size = 64
num_epochs = 300
# NUM_EPOCHS_BEFORE_FINE_TUNING = 150
losses = []

# log_file = open("training_log.txt", "w")
# sys.stdout = log_file

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler("training_log_ModifiedResnet50_1409_3.txt"),
                              logging.StreamHandler()])  # StreamHandler is for console
logger = logging.getLogger()

#Assuming these are required transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
    transforms.ToTensor(),
])

full_dataset = EmbryoDataset(txt_path="ed4_as_target.txt", transform=transform)
train_indices, val_indices, test_indices = split_indices(len(full_dataset), train_pct=0.6, val_pct=0.2, seed=42, stratify=full_dataset.label_list)
logger.info(f"{len(train_indices)}, {len(val_indices)}, {len(test_indices)}")

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=test_sampler)



# Initialize Model and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50Classifier(num_classes=5).to(device)

# # Freeze all layers
# for param in model.resnet50.parameters():
#     param.requires_grad = False

# # Unfreeze the last layer/classifier layer (fully connected layer)
# for param in model.resnet50.fc.parameters():
#     param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min')  # 'min' indicates reducing LR when a quantity (val loss in our case) stops decreasing
# scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))

num_classes = 5
true_positives = torch.zeros(num_classes)
valloss =[]
val_acc = []

# Early Stopping
patience = 50  # Number of epochs to wait for improvement before stopping
best_valid_loss = float('inf')
best_valid_epoch = 0

best_val_loss = float('inf')
epochs_no_improve = 0
patience_epochs = 10
fine_tuning_started = False # Flag to indicate if fine-tuning has started

# Training Loop
for epoch in range(num_epochs):
    model.train()
    tr_loss = 0
    for images, labels in train_loader:
        # print(images.size(), labels.size())
        images, labels = images.to(device), labels.long().to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        tr_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tr_loss /= len(train_loader)  #average loss per batch for the epoch
    losses.append(tr_loss)
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    true_positives.fill_(0)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            for i in range(num_classes):
                true_positives[i] += ((predicted == labels) & (predicted == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_indices)

    scheduler.step(val_loss)
    # scheduler.step()

    valloss.append(val_loss)
    val_acc.append(accuracy)

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     epochs_no_improve = 0
    # else:
    #     epochs_no_improve += 1

    # # If patience is reached and fine-tuning hasn't started, start fine-tuning
    # if epochs_no_improve == patience_epochs and not fine_tuning_started:
    #     print("Starting fine-tuning...")
    #     # Unfreeze all parameters of the model
    #     for param in model.resnet50.parameters():
    #         param.requires_grad = True

    #     optimizer = optim.Adam(model.parameters(), lr=lower_lr, weight_decay=1e-4)
    #     # Set flag to prevent multiple starts
    #     fine_tuning_started = True
    #     epochs_no_improve = 0  # reset for the next phase

    # Early Stopping
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_epoch = epoch
        # Save the model checkpoint whenever validation loss improves
        torch.save(model.state_dict(), 'model_ModifiedResnet50_1409_3.pth')
    if epoch - best_valid_epoch >= patience:
        logger.info(f"Validation loss hasn't improved for {patience} epochs. Stopping training.")
        break

    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100. * correct / len(val_indices):.2f}%")
    for i in range(num_classes):
        logger.info(f"Class {i} Recall: {true_positives[i] / total_per_class[i]:.2f} True Positives: {true_positives[i]}/{total_per_class[i]}")

# Save model
torch.save(model.state_dict(), f'model_ModifiedResnet50_1409_3.pth')
logger.info('Finished Training. Model Saved')
# Graph it out!
plt.plot(losses)
plt.ylabel("Training Loss")
plt.xlabel('Epoch')
plt.title("Training Loss over Epochs")
plt.savefig("training_loss_plot_ModifiedResnet50_1409_3.png", dpi=300)
plt.clf()

plt.plot(valloss)
plt.ylabel("Validation Loss")
plt.xlabel('Epoch')
plt.title("Validation Loss over Epochs")
plt.savefig("val_loss_plot_ModifiedResnet50_1409_3.png", dpi=300)
plt.clf()

plt.plot(val_acc)
plt.ylabel("Validation Accuracy")
plt.xlabel('Epoch')
plt.title("Validation Accuracy over Epochs")
plt.savefig("val_accuracy_plot_ModifiedResnet50_1409_3.png", dpi=300)



#INFERENCE

def infer_and_write_results(model, dataloader, indices, dataset, device, csv_filename='results_ModifiedResnet50_1409_3.csv'):
    """
    Use the trained model to make predictions on the dataloader and save results in a CSV.
    """
    model.eval()
    correct = 0
    # Open a CSV file to write results
    with open(csv_filename, 'w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(["Image_Path", "Predicted_Class", "Ground_Truth"])

        # Iterate over dataloader and write results to CSV
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Make predictions
                outputs = model(images)
                _, predicted = outputs.max(1)

                # Assuming dataloader returns batches of data in sequential order from dataset
                for idx, (pred, gt) in enumerate(zip(predicted, labels)):
                    img_path = dataset.img_list[dataloader.sampler.indices[idx]]
                    csvwriter.writerow([img_path, pred.item(), gt.item()])
                    # Correct or not
                    if pred.item() == gt.item():
                        correct +=1
    accuracy = 100. * correct / len(indices)
    logger.info(f'Accuracy on test data: {accuracy:.2f}% ({correct}/{len(indices)})')


    logger.info(f'We got {correct} correct!')


# Load trained model
model_path = "model_ModifiedResnet50_1409_3.pth"
# model = Classifier().to(device)
model = ResNet50Classifier(num_classes=5).to(device)
# model = GoogLeNetClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load(model_path))

# Call the function
infer_and_write_results(model, test_loader, test_indices, full_dataset, device, 'results_ModifiedResnet50_1409_3.csv')
# log_file.close()
