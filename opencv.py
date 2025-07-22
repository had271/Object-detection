import cv2
from ultralytics import YOLO


# Load YOLO pre-trained model
model = YOLO('yolo11n.pt')

#start the device camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Couldn't open the camera")
    exit()

# Function to process frames for object detection
def process_frame(frame):
    results = model(frame)   
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = float(box.conf[0].numpy())
            class_detected_number = int(box.cls[0])
            class_detected_name = result.names[class_detected_number]

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # text label
            text = f'{class_detected_name} ({confidence:.2f}%)'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 0, 255), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read frame")
        break

    processed_frame = process_frame(frame)
    cv2.imshow("YOLO Object Detection", processed_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()