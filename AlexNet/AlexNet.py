import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5,padding=2), 
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    def forward(self, x):
        x = self.net(x)  
        x = F.softmax(x, dim=1)
        return x
        



random_data = torch.rand((2, 3, 224, 224))
model = AlexNet(10)
x = model(random_data)
print(x)