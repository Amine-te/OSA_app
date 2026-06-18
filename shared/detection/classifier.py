import torch
from PIL import Image

def classify_crop(cropped_image, cnn_transform, cnn_model, class_names, device):
    """Classify a cropped image using the CNN classifier"""
    pil_image = Image.fromarray(cropped_image)
    input_tensor = cnn_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

    return predicted_class, confidence_score
