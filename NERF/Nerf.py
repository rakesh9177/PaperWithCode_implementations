import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image


class MLP(nn.Module):
    def __init__(self, input_xyz=3, input_d=3 , hidden_size=256) -> None:
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(input_xyz, hidden_size),
                                  nn.ReLU()])
        for i in range(7):
            self.fc.extend([nn.Linear(hidden_size, hidden_size),
                            nn.ReLU()])
        self.sigma_layer = nn.Linear(hidden_size, 1)

        self.d = nn.Sequential(
            nn.Linear(input_d, hidden_size),  # Fewer layers for d
            nn.ReLU()
        )
        self.combine_layer = nn.Sequential(
            nn.Linear(hidden_size+hidden_size, 128),  # Concatenate x and d
            nn.ReLU()
        )
        self.rgb_layer = nn.Sequential(nn.Linear(128, 3),
                                        nn.ReLU())

    def forward(self,x, d):
        for layer in self.fc:
            x = layer(x)
        sigma = self.sigma_layer(x)
        d = self.d(d)
        d = torch.concat((x, d), dim=1)
        d = self.combine_layer(d)
        d = self.rgb_layer(d)
        return sigma, d
    
def render_image(sigma, color, volume_shape = (64, 64, 64), delta=0.01):
        N=1
        sigma = sigma.detach().numpy()
        color = color.detach().numpy()
        rendered_image = np.ones(volume_shape + (3,))
        x, y, z = np.meshgrid(np.arange(volume_shape[0]), np.arange(volume_shape[1]), np.arange(volume_shape[2]))
        for i in range(N):
            Ti = np.exp(-np.cumsum(np.sum(sigma[:i]) * delta))  # Calculate Ti
            C_r = Ti * (1 - np.exp(-sigma[i] * delta)) * color[i]  # Calculate C(r) for this component
            rendered_image[x, y, z] += C_r  # Accumulate C(r)
        rendered_image = np.clip(rendered_image, 0, 1)
        plt.imshow(rendered_image[:, :, volume_shape[2] // 2, :])
        plt.show()

image = Image.open("/Users/rakeshrathod/Desktop/PaperWithCode_implementations/NERF/image_277.jpg")
print(image)
transform = transforms.ToTensor()
xyz = transform(image)
input_d = torch.rand((2, 3))

model = MLP(input_xyz=3, input_d=3)
sigma, d = model(xyz, input_d)
print(sigma)
print(d)

render_image(sigma, d)

