import json
import os
import cv2

# with open("/home/alex/U-Net-VSR/data/raw/test_images_lr", "r") as f:
image_dir = "/home/alex/U-Net-VSR/data/raw/test_images"
for img in os.listdir(image_dir):
    image = cv2.imread(os.path.join(image_dir, img))
    image = cv2.resize(image, (1280//4, 720//4), cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("/home/alex/U-Net-VSR/data/raw/test_images_lr", img), image)