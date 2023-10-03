import numpy as np
import matplotlib.pyplot as plt

# Define volume dimensions
volume_shape = (64, 64, 64)

# Create a synthetic volume dataset with random data
volume_data = np.random.rand(*volume_shape)

# Define properties for a single component (you can have multiple components)
sigma_i = 0.2  # Volume density
delta_i = 0.01  # Differential distance
color_i = np.array([1.0, 0.0, 0.0])  # Color (red)

# Initialize the final rendered image
rendered_image = np.ones(volume_shape + (3,))

# Perform volume rendering using the equation and accumulate contributions
for z in range(volume_shape[2]):
    for y in range(volume_shape[1]):
        for x in range(volume_shape[0]):
            # Compute C(r) for the current voxel using the equation
            Ti = np.exp(-np.cumsum(sigma_i * delta_i))
            C_r = np.sum(Ti * (1 - np.exp(-sigma_i * delta_i)) * color_i)
            
            # Assign the computed color to the rendered image
            rendered_image[x, y, z] = C_r * color_i

# Display the rendered image (slice along a plane)
plt.imshow(rendered_image[:, :, volume_shape[2] // 2, :])
plt.show()
