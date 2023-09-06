import torch
from torchvision.transforms import transforms
from PIL import Image
import csv
from Classifier import Classifier
from load_data import test_loader, full_dataset

#Assuming these are required transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# def predict_image(image_path, model, device):
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image)
#     # Add batch dimension and send to device
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = image_tensor.to(device)
#     with torch.no_grad():
#         output = model(input)
#         _, predicted = output.max(1)
#     # index = output.data.cpu().numpy().argmax()
#     return predicted.item()

def infer_and_write_results(model, dataloader, dataset, device, csv_filename='results.csv'):
    """
    Use the trained model to make predictions on the dataloader and save results in a CSV.
    """
    model.eval()

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


# Load trained model
model_path = "model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

# Call the function
infer_and_write_results(model, test_loader, full_dataset, device, 'results.csv')

# # Score images and write results to CSV
# image_paths = ['list_of_image_paths']  # Modify this to have paths of images you want to predict

# with open('results.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(["Image_Path", "Predicted_Class"])
    
#     for image_path in image_paths:
#         predicted_class = predict_image(image_path, model, device)
#         csvwriter.writerow([image_path, predicted_class])
