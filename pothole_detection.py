import subprocess
import os
import cv2
from ultralytics import YOLO

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load the YOLOv8 model
model = YOLO("weights/best.pt")

ffmpeg_path = "ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"

# Function to detect potholes in an image
def detect_from_image(image_path, model):
    image = cv2.imread(image_path) #read the image file 
    if image is None:
        print("Error: Cannot read the image.")
        return

    results = model(image)
    for result in results:
        annotated_image = result.plot()
        output_path = "results/image_result.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

# Function to detect potholes in a video
def detect_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (frame_width, frame_height)

    output_video_path = "results/video_result.avi"
    result_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    print(f"Processing video... Output will be saved to {output_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        results = model(frame)
        for result in results:
            annotated_frame = result.plot()
            result_writer.write(annotated_frame)

    cap.release()
    result_writer.release()
    print(f"Video processing complete. Output saved to {output_video_path}")

    # Convert the output video to MP4 using FFmpeg
    output_mp4_path = "results/processed.mp4"
    subprocess.run([ffmpeg_path, "-i", output_video_path, "-vcodec", "libx264", output_mp4_path, "-y"])
    print(f"Video conversion complete. Saved to {output_mp4_path}")

# Example usage (this will be called in your Streamlit app)
if __name__ == '__main__':
    # Test detection from image
    detect_from_image('uploads/image.jpg', model)

    # Test detection from video
    detect_from_video('uploads/video.mp4', model)
