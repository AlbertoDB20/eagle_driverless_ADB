import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from IPython.display import Image, clear_output
from PIL import Image
import glob

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Load the YOLOv5 model (replace 'best.pt' with your model file if you have a custom-trained one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_cones(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 model on the image
    results = model(img_rgb)

    # Extract predictions (bounding boxes, confidence scores, and class names)
    detections = results.pandas().xyxy[0]  # Extracting DataFrame with results

    # Filter for only cone detections if classes are known
    # If 'cone' is labeled with a specific class name in your model, replace 'cone_class_name' with that
    cone_class_name = 'cone'  # Modify if the class name in your dataset is different
    cone_detections = detections[detections['name'] == cone_class_name]

    # Display the results
    for index, row in cone_detections.iterrows():
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']

        # Draw bounding box and label on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{cone_class_name}: {confidence:.2f}"
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the resulting image with bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Return bounding box information
    return cone_detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']]

# Test the function with an example image
image_path = "correct_image.png"  # Replace with the path to your image
bounding_boxes = detect_cones(image_path)

# Print bounding boxes and confidence scores
print("Bounding Boxes for Cones:")
print(bounding_boxes)