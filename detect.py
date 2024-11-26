import cv2
import torch
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load

# Load pre-trained YOLO model
# Replace 'path_to_yolo_model' with the actual path to your YOLO model
yolo_model = attempt_load('path_to_yolo_model', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
yolo_model.eval()

# Define class labels
class_labels = ['helmet', 'head', 'person']

# Function to perform object detection using YOLO
def detect_objects_yolo(frame):
    img = torch.from_numpy(frame).to(yolo_model.device).float() / 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)

    pred = yolo_model(img)[0]
    pred = non_max_suppression(pred['xyxy'].cpu(), conf_thres=0.5, iou_thres=0.5)[0]

    for det in pred:
        label = int(det[5])
        class_name = class_labels[label]
        box = det[:4].int().tolist()

        # Draw bounding box on the frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Load video file
# Replace 'path_to_input_video' with the actual path to your input video file
video_capture = cv2.VideoCapture('path_to_input_video')

# Get video properties
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
# Replace 'output_video.mp4' with the desired output video file name
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform object detection using YOLO
    detected_frame_yolo = detect_objects_yolo(frame)

    # Display the frame with annotations
    cv2.imshow('YOLO Detection', detected_frame_yolo)

    # Write the frame to the output video
    output_video.write(detected_frame_yolo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
