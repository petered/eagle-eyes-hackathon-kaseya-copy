from torchvision import transforms
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
import os

height = 100
width = 100
model_name = 'model-resnet50-1005.pth'

def process_image(image_path):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop(224),  # Replace with actual size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = Image.open(image_path)
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

folder_path = './test'
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

model = models.resnet50(pretrained=True)
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