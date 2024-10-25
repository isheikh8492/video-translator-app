import cv2
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image, ImageFont, ImageDraw
from typing import List, Tuple, Dict
import easyocr
import logging
import os
from collections import deque
from dataclasses import dataclass


@dataclass
class FrameBatch:
    frames: List[np.ndarray]
    indices: List[int]
    texts: List[List[Tuple]] = None
    translations: List[List[str]] = None


class VideoTextTranslator:

    def __init__(self, batch_size: int = 8):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size

        self.reader = easyocr.Reader(["en"])
        self.translator_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-es"
        )
        self.translator_tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-es"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator_model.to(self.device)

    def get_font_path(self):
        """Get the appropriate font path based on the operating system."""
        if os.name == "nt":  # Windows
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return path
        else:  # Linux/Mac
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return path

        # Fallback to default
        return None

    def create_text_image(self, text: str, original_region: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Create an image containing the text with proper dimensioning using updated Pillow methods.
        """
        try:
            # Create base image with original dimensions
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Get font path
            font_path = self.get_font_path()

            # Start with a small font size
            font_size = 12
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

            # Calculate font size to match original text scale
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top

            width_ratio = width / text_width
            height_ratio = height / text_height
            scale_factor = min(width_ratio, height_ratio) * 0.8

            new_font_size = int(font_size * scale_factor)
            font = ImageFont.truetype(font_path, new_font_size) if font_path else ImageFont.load_default()

            # Get text position for centering
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            x = (width - text_width) // 2
            y = (height - text_height) // 2

            # Draw text in white to create mask
            draw.text((x, y), text, font=font, fill=(255, 255, 255))

            # Convert to numpy array
            text_mask = np.array(img)[:, :, 3]

            # Create output image
            output = np.zeros_like(original_region)

            # Apply original styling using the text mask
            for c in range(3):  # RGB channels
                output[:, :, c] = cv2.bitwise_and(
                    original_region[:, :, c],
                    original_region[:, :, c],
                    mask=text_mask
                )

            return output

        except Exception as e:
            self.logger.error(f"Error creating styled text image: {str(e)}")
            return original_region

    def detect_text_batch(self, batch: FrameBatch) -> List[List[Tuple]]:
        """Detect text in a batch of frames."""
        all_results = []
        try:
            # Process all frames in the batch at once with EasyOCR
            # Note: EasyOCR's readtext already handles batching internally
            frames_array = np.array(batch.frames)
            results = self.reader.readtext(frames_array)

            # Process results for each frame
            for frame_idx in range(len(batch.frames)):
                frame_height, frame_width = batch.frames[frame_idx].shape[:2]
                frame_results = []

                for box, text, conf in results[frame_idx]:
                    if conf < 0.5:
                        continue

                    # Convert box points to integer coordinates
                    box = np.array(box).astype(np.int32)

                    # Get bounding box coordinates
                    x_min = max(0, min(box[:, 0]))
                    y_min = max(0, min(box[:, 1]))
                    x_max = min(frame_width, max(box[:, 0]))
                    y_max = min(frame_height, max(box[:, 1]))

                    if x_min >= x_max or y_min >= y_max:
                        continue

                    frame_results.append(((x_min, y_min, x_max, y_max), text, conf))

                all_results.append(frame_results)

        except Exception as e:
            self.logger.error(f"Error in batch text detection: {str(e)}")
            # Return empty results for all frames in case of error
            all_results = [[] for _ in range(len(batch.frames))]

        return all_results

    def translate_batch(self, texts_batch: List[List[str]]) -> List[List[str]]:
        """Translate multiple texts in batch."""
        try:
            # Flatten all texts into a single list
            flat_texts = [
                text for frame_texts in texts_batch for text, _ in frame_texts
            ]

            if not flat_texts:
                return [[]] * len(texts_batch)

            # Prepare inputs for batch translation
            inputs = self.translator_tokenizer(
                flat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate translations
            with torch.no_grad():
                outputs = self.translator_model.generate(**inputs)

            # Decode translations
            translations = self.translator_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Unflatten translations back to original batch structure
            result = []
            idx = 0
            for frame_texts in texts_batch:
                frame_translations = []
                for _ in frame_texts:
                    if idx < len(translations):
                        frame_translations.append(translations[idx])
                    idx += 1
                result.append(frame_translations)

            return result

        except Exception as e:
            self.logger.error(f"Batch translation error: {str(e)}")
            return [[]] * len(texts_batch)

    def process_frame_batch(self, batch: FrameBatch) -> List[np.ndarray]:
        """Process a batch of frames."""
        try:
            # Detect text in all frames
            text_regions = self.detect_text_batch(batch)

            # Extract all texts for translation
            texts_to_translate = [
                [(text, coords) for coords, text, _ in frame_regions]
                for frame_regions in text_regions
            ]

            # Translate all texts
            translated_texts = self.translate_batch(texts_to_translate)

            # Process each frame with its translations
            processed_frames = []
            for frame_idx, frame in enumerate(batch.frames):
                processed_frame = frame.copy()
                frame_translations = translated_texts[frame_idx]
                frame_regions = texts_to_translate[frame_idx]

                for (coords, _), translated_text in zip(
                    frame_regions, frame_translations
                ):
                    x_min, y_min, x_max, y_max = coords
                    region_width = x_max - x_min
                    region_height = y_max - y_min

                    if region_width <= 0 or region_height <= 0:
                        continue

                    # Create and apply translated text image
                    text_image = self.create_text_image(
                        translated_text, region_width, region_height
                    )

                    if text_image.shape[:2] != (region_height, region_width):
                        text_image = cv2.resize(
                            text_image, (region_width, region_height)
                        )

                    processed_frame[y_min:y_max, x_min:x_max] = text_image

                processed_frames.append(processed_frame)

            return processed_frames

        except Exception as e:
            self.logger.error(f"Error processing frame batch: {str(e)}")
            return batch.frames  # Return original frames in case of error

    def process_video(self, input_path: str, output_path: str):
        """Process video with batch frame processing."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open input video")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError("Could not create output video")

            frame_count = 0
            batch_frames = []
            batch_indices = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in the last batch
                    if batch_frames:
                        batch = FrameBatch(frames=batch_frames, indices=batch_indices)
                        processed_frames = self.process_frame_batch(batch)
                        for processed_frame in processed_frames:
                            out.write(processed_frame)
                    break

                batch_frames.append(frame)
                batch_indices.append(frame_count)
                frame_count += 1

                # Process batch when it reaches the specified size
                if len(batch_frames) == self.batch_size:
                    batch = FrameBatch(frames=batch_frames, indices=batch_indices)
                    processed_frames = self.process_frame_batch(batch)

                    for processed_frame in processed_frames:
                        out.write(processed_frame)

                    # Clear batch
                    batch_frames = []
                    batch_indices = []

                # Update progress
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processing progress: {progress:.2f}%")

            cap.release()
            out.release()
            self.logger.info("Video processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            if "cap" in locals():
                cap.release()
            if "out" in locals():
                out.release()
            raise

    def detect_text_regions(self, frame: np.ndarray) -> List[Tuple]:
        """Detect text regions in the frame with proper coordinate handling."""
        try:
            results = self.reader.readtext(frame)
            processed_results = []
            frame_height, frame_width = frame.shape[:2]

            for box, text, conf in results:
                if conf < 0.5:
                    continue

                # Convert box points to integer coordinates
                box = np.array(box).astype(np.int32)

                # Get bounding box coordinates
                x_min = max(0, min(box[:, 0]))
                y_min = max(0, min(box[:, 1]))
                x_max = min(frame_width, max(box[:, 0]))
                y_max = min(frame_height, max(box[:, 1]))

                # Ensure box has valid dimensions
                if x_min >= x_max or y_min >= y_max:
                    continue

                processed_results.append(((x_min, y_min, x_max, y_max), text, conf))

            return processed_results

        except Exception as e:
            self.logger.error(f"Error in text detection: {str(e)}")
            return []

    def translate_text(self, text: str) -> str:
        """Translate text from English to Spanish with error handling."""
        try:
            inputs = self.translator_tokenizer(text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.translator_model.generate(**inputs)
            translated = self.translator_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return translated

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return text

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with proper dimension handling."""
        try:
            processed_frame = frame.copy()
            text_regions = self.detect_text_regions(frame)

            for coords, text, conf in text_regions:
                x_min, y_min, x_max, y_max = coords
                region_width = x_max - x_min
                region_height = y_max - y_min

                if region_width <= 0 or region_height <= 0:
                    continue

                # Extract original region with styling
                original_region = processed_frame[y_min:y_max, x_min:x_max]

                # Translate text
                translated_text = self.translate_text(text)

                # Create new text image with preserved styling
                text_image = self.create_text_image(
                    translated_text,
                    original_region,
                    region_width,
                    region_height
                )

                # Apply the styled text image to the frame
                processed_frame[y_min:y_max, x_min:x_max] = text_image

            return processed_frame

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def process_video(self, input_path: str, output_path: str):
        """Process the entire video with proper error handling."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open input video")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError("Could not create output video")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = self.process_frame(frame)
                out.write(processed_frame)

                # Update progress
                frame_count += 1
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processing progress: {progress:.2f}%")

            cap.release()
            out.release()
            self.logger.info("Video processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            # Cleanup
            if "cap" in locals():
                cap.release()
            if "out" in locals():
                out.release()
            raise


def main():
    try:
        translator = VideoTextTranslator(batch_size=8)
        translator.process_video("input_video.mp4", "output_video.mp4")
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
