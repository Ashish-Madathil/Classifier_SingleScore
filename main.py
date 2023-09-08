
# External imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import csv
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import matplotlib.pyplot as plt

# Local imports
# from Classifier import Classifier
from resnet_18 import ResNet18Classifier
from stratified_sampling import split_indices
from embryo_dataset import EmbryoDataset

# # Global Variables (can be moved to a configuration file or passed as arguments later)
# MODEL_PATH = "model.pth"
# CSV_FILENAME = 'results.csv'
# TRAIN_PCT = 0.6
# VAL_PCT = 0.2
# LEARNING_RATE = 0.001
# BATCH_SIZE = 32
# NUM_EPOCHS = 100

# Hyperparameters
learning_rate = 0.00001
batch_size = 32
num_epochs = 300
losses = []

#Assuming these are required transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

full_dataset = EmbryoDataset(txt_path="ed4_as_target.txt", transform=transform)
train_indices, val_indices, test_indices = split_indices(len(full_dataset), train_pct=0.6, val_pct=0.2, seed=42, stratify=full_dataset.label_list)
print(len(train_indices), len(val_indices), len(test_indices))

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=test_sampler)

# for images, labels in train_loader:
#     print(images.size(), labels.size())
#     break

# Initialize Model and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Classifier().to(device)
model = ResNet18Classifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_classes = 5
true_positives = torch.zeros(num_classes)


# # Small subset of the dataset for testing
# subset_data = torch.utils.data.Subset(full_dataset, indices=range(32))
# loader = DataLoader(subset_data, batch_size=32)

# for images, labels in loader:
#     outputs = model(images.to(device))
#     print(outputs.size(), labels.size())

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        # print(images.size(), labels.size())
        images, labels = images.to(device), labels.long().to(device)

        # Forward pass
        outputs = model(images)
        # print('OUTPUTS : ', outputs.size())
        # print('LABELS : ', labels.size(), labels)
        loss = criterion(outputs, labels)
        losses.append(loss.detach().cpu().numpy())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    valloss =[]
    correct = 0
    true_positives.fill_(0)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            valloss.append(val_loss)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            for i in range(num_classes):
                true_positives[i] += (predicted == labels).logical_and(predicted == i).sum().item()
                total_per_class[i] += (labels == i).sum().item()

    # val_loss /= len(val_loader.dataset)
    # accuracy = 100. * correct / len(val_loader.dataset)
    val_loss /= len(val_indices)
    # valloss.append(val_loss)
    accuracy = 100. * correct / len(val_indices)
    # print('Accuracy : ', accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100. * correct / len(val_indices):.2f}%")
    for i in range(num_classes):
        print(f"Class {i} Recall: {true_positives[i] / total_per_class[i]:.2f} True Positives: {true_positives[i]}/{total_per_class[i]}")
        # print(f"Class {i} True Positives: {true_positives[i]}/{total_per_class[i]}")

# Save model
torch.save(model.state_dict(), f'model.pth')
print('Finished Training. Model Saved')
# Graph it out!
avg_losses = [sum(losses[i:i+46])/46 for i in range(0, 13800, 46)] # 9200 = 300 epochs * 46 batches
plt.plot(range(num_epochs), avg_losses)
plt.ylabel("Average Loss")
plt.xlabel('Epoch')
plt.title("Training Loss over Epochs")

# Save the figure as a PNG file
plt.savefig("training_loss_plot.png", dpi=300)
plt.clf()
avg_vallosses = [sum(valloss[i:i+46])/46 for i in range(0, 13800, 46)]
plt.plot(range(num_epochs), avg_vallosses)
plt.ylabel("Average Validation Loss")
plt.xlabel('Epoch')
plt.title("Validation Loss over Epochs")

# Save the figure as a PNG file
plt.savefig("val_loss_plot.png", dpi=300)



#INFERENCE

def infer_and_write_results(model, dataloader, dataset, device, csv_filename='results.csv'):
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

    print(f'We got {correct} correct!')


# Load trained model
model_path = "model.pth"
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

# Call the function
infer_and_write_results(model, test_loader, full_dataset, device, 'results.csv')