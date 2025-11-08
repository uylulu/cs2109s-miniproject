from torch import nn
import torch
from torchvision import transforms
from PIL import Image


def some(num_classes: int) -> nn.Module:
    res = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.LeakyReLU(0.1),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.LeakyReLU(0.1),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4096, 64),
        nn.LeakyReLU(0.1),
        nn.Linear(64, num_classes),
    )
    return res


def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a TorchScript model loaded from an embedded, base64-encoded compressed blob.
    Self-contained: no need for the original Python class.

    Args:
        device: Where to map the model (e.g., "cpu", "cuda", "cuda:0").
        dtype: Optional dtype to convert parameters/buffers to (e.g., "float32", "float16").
    """
    model = some(21)
    model.load_state_dict(
        torch.load(
            "/home/dotronguy/Desktop/cs2109s-miniproject/image_classification/image_models/new_model",
            map_location="cpu",
        )
    )
    model.eval()
    return model


def get_augmentations():
    return transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.5,
                saturation=0.3,
                hue=0.02,
            ),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ]
    )


class ImageClassify:
    def __init__(self):
        self.label_mapping: list[str] = [
            "boots",
            "box",
            "coin",
            "dragon",
            "exit",
            "floor",
            "gem",
            "ghost",
            "human",
            "key",
            "lava",
            "locked",
            "metalbox",
            "opened",
            "portal",
            "robot",
            "shield",
            "sleeping",
            "spike",
            "wall",
            "wolf",
        ]
        self.transform = get_augmentations()
        self.model = get_model()
        self.model.eval()

    def predict(self, x: Image.Image) -> str:
        x = self.transform(x)
        x = x.unsqueeze(0)
        # expects the tensor to have size 3, 58, 58
        self.model.eval()
        pred = torch.argmax(self.model(x), dim=1)

        return self.label_mapping[int(pred.item())]
