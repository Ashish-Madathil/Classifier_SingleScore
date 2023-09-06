import torch
from torchvision.transforms import transforms
from PIL import Image
import csv

#Assuming these are required transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    # Add batch dimension and send to device
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    with torch.no_grad():
        output = model(input)
        _, predicted = output.max(1)
    # index = output.data.cpu().numpy().argmax()
    return predicted.item()

# Load trained model
model_path = "model.pth"
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Score images and write results to CSV
image_paths = ['list_of_image_paths']  # Modify this to have paths of images you want to predict

with open('results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Image_Path", "Predicted_Class"])
    
    for image_path in image_paths:
        predicted_class = predict_image(image_path, model, device)
        csvwriter.writerow([image_path, predicted_class])
