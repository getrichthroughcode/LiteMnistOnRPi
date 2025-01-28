from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np


image_size = (1000, 1000)
output_dir = "../data/mnist_like_data"
Font_path = "/Work/training/Fonts/DejaVuMathTeXGyre.ttf"
digits = list(range(10))  
num_samples = 10  

os.makedirs(output_dir, exist_ok=True)
for digit in digits:
    os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)


for digit in digits:
    for i in range(num_samples):
        img = Image.new("L", image_size, color=0) 
        draw = ImageDraw.Draw(img)
        
        
        font_size = 425 
        font = ImageFont.truetype(Font_path, font_size)  
        x, y = random.randint(250,700), random.randint(250, 700) 
        draw.text((x, y), str(digit), fill=255, font=font)

        
        img = img.rotate(random.uniform(-15, 15)) 
        img = img.resize(image_size) 

        
        img.save(os.path.join(output_dir, str(digit), f"{digit}_{i}.png"))

print(f"Dataset generated in {output_dir}")