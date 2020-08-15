#!/usr/bin/env python3
import os
from model import SemanticSegmentation

images = [f for f in os.listdir('./images/')
          if os.path.splitext(f)[-1] == '.jpg']  # collect all images from images folder

# run model for every image
MODEL = SemanticSegmentation()
for image_name in images:
    MODEL.run_segmentation_visualization('./images/' + image_name)