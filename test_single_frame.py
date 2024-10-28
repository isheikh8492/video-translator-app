import cv2
from video_text_translator import VideoTextTranslator


def test_single_frame(input_image_path: str, output_image_path: str):
    try:
        translator = VideoTextTranslator(
            batch_size=1
        )  # batch_size doesn't matter for single frame
        
        # Process the single frame
        processed_frame = translator.process_single_frame(input_image_path)

        if processed_frame is not None:
            # Save the processed frame
            cv2.imwrite(output_image_path, processed_frame)
            print(f"Processed image saved to {output_image_path}")
        else:
            print("Failed to process the frame.")

    except Exception as e:
        print(f"Error in test_single_frame: {str(e)}")


if __name__ == "__main__":
    # Path to the input frame
    input_path = "uploads/frame_157.png"  # Adjust the filename as needed
    output_path = "output/processed_frame_157.png"

    # Run the test
    test_single_frame(input_path, output_path)
