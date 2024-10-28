import cv2
import os
import numpy as np


def extract_frames_from_video(
    video_path: str, output_folder: str = "uploads", max_frames: int = None
) -> list:
    """
    Extract frames from a video and save them as images in the specified folder.

    Args:
    video_path (str): Path to the input video file.
    output_folder (str): Folder to save extracted frames. Defaults to "uploads".
    max_frames (int, optional): Maximum number of frames to extract. If None, extract all frames.

    Returns:
    list: List of paths to saved frame images.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if max_frames is not None and frame_count >= max_frames:
            break  # Reached maximum number of frames

        # Generate filename for each frame
        frame_filename = f"frame_{frame_count}.png"
        frame_path = os.path.join(output_folder, frame_filename)

        # Save the frame as an image
        cv2.imwrite(frame_path, frame)
        saved_frames.append(frame_path)

        frame_count += 1

        # Optional: Print progress
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count} frames...")

    # Release the video capture object
    cap.release()

    print(f"Total frames extracted: {frame_count}")
    return saved_frames


# Example usage
if __name__ == "__main__":
    video_path = "../input_video.mp4"
    output_folder = "../uploads"
    max_frames = None  # Set to None to extract all frames

    extracted_frames = extract_frames_from_video(video_path, output_folder, max_frames)
    print(f"Frames saved in {output_folder}")
    print(f"First few frames: {extracted_frames[:5]}")
