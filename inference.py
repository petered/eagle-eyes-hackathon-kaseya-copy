from torchvision import transforms
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
import os

height = 100
width = 100

#folder_path = './test'
#image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_path = './test/p3.png'

def process_image(image_path):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Replace with actual size
        transforms.ToTensor(),
        # Add any other transformations used during training
    ])
    # Load the image and apply transformations
    input_image = Image.open(image_path)
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

image_tensor = process_image(image_path)

model = models.resnet18(pretrained=True)  # Replace with your model class
# Modify the last layer of the model
num_classes = 2 # replace with the number of classes in your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model-restnet18-4010.pth'))  # Load the saved model
model.eval()  # Set the model to evaluation mode



with torch.no_grad():  # No need to track gradients during inference
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

prob_predicted_class = probabilities[0][predicted_class].item()
print(f"Predicted class: {predicted_class.item()}, Probability: {prob_predicted_class:.4f}")

