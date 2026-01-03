import cv2
import matplotlib.pyplot as plt
import os

# Configuration
input_image_path = os.path.join('data', 'from_jeremy', 'images', 'm-t0627-00538-00634.jpg')
output_image_path = os.path.join('data', 'processed', 'preprocessed_image.png')

# Read image
image = cv2.imread(input_image_path, 0)
if image is None:
    print("Error: Could not load image. Check the path.")
    exit()

# Process image
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# Save result
os.makedirs('data/processed', exist_ok=True)
cv2.imwrite(output_image_path, binary_image)
print(f"Success! Image saved to: {output_image_path}")

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.show()