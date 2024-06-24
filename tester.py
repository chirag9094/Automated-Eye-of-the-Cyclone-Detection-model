from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

def display_bounding_box(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2, confidence, class_index = boxes
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    image.show()

def main():
    model = YOLO('best.pt')
    image_path = r'C:\Users\Chirag C\vit\docs\cyclone\data\images\train\2014101121203300V.ppi.jpg'  
    results = model(image_path)[0]
    bounding_box_coordinates1 =  results.boxes.data.tolist()
    bounding_box_coordinates1 = torch.tensor(bounding_box_coordinates1[0])
    display_bounding_box(image_path, bounding_box_coordinates1)

if __name__ == '__main__':
    main()
