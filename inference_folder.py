from torchvision import transforms
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
import os

height = 100
width = 100
model_name = 'model-resnet18-2020.pth'

def process_image(image_path):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Replace with actual size
        transforms.ToTensor(),
        # Add any other transformations used during training
    ])
    input_image = Image.open(image_path)
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

folder_path = './test'
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

model = models.resnet18(pretrained=True)
num_classes = 2  # Replace with your number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_name))
model.eval()

for image_path in image_paths:
    image_tensor = process_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        prob_predicted_class = probabilities[0][predicted_class].item()
        print(f"{image_path} - Predicted class: {predicted_class.item()}, Probability: {prob_predicted_class:.4f}")