import os
import torch
import torchvision.transforms as transforms

def get_device():
    """Detect dynamic optimal device for PyTorch execution."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEFAULT_ASSIGNMENT_PARAMS = {
    'spatial_context_weight': 0.5,
    'proximity_weight': 0.25,
    'scarcity_weight': 0.15,
    'pattern_weight': 0.1,
    'confidence_weight': 0.05,
    'clustering_eps': 80,
    'min_cluster_size': 2,
    'max_assignment_distance': 200,
    'spatial_context_threshold': 100,
    'neighbor_alignment_tolerance': 50,
    'isolation_distance_threshold': 100,
    'min_assignment_confidence': 0.3,
    'high_confidence_threshold': 0.5
}

DEFAULT_CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
