from ultralytics import YOLO
import torch
import cv2

# Kiểm tra xem có GPU không
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load mô hình YOLOv8 cho Pose Estimation (pretrained)
model = YOLO('yolov8n-pose.pt')
model.to(device)

# Đọc video từ webcam
cap = cv2.VideoCapture('2human.mp4')

while True:
    print("="*50)
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame , (0, 0), fx=0.3, fy=0.3)
    # Chạy mô hình YOLOv8 trên từng frame để phát hiện pose
    results = model(frame, device=device)

    # Lấy danh sách các kết quả phát hiện
    detections = results[0]

    # Kiểm tra xem có người nào được phát hiện không
    if detections:
        for i, pose in enumerate(detections.keypoints):
            print(f"Person {i + 1}:")
            for keypoint in pose:
                print("*******************************")
                print(keypoint.xy)
                # x, y, confidence = keypoint
                # print(f"Keypoint: (x={x}, y={y}, confidence={confidence})")
                print("*******************************")
            print("-" * 20)

    # Trực quan hóa kết quả phát hiện pose
    annotated_frame = detections.plot()

    # Hiển thị frame đã được annotate
    cv2.imshow('Pose Estimation', annotated_frame)

    # Thoát chương trình khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("=" * 50)
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
