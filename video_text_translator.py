import cv2
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Tuple
import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import easyocr


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

        # Load STEFANN models
        custom_objects = {
            "VarianceScaling": tf.keras.initializers.VarianceScaling,
            "Model": tf.keras.Model,
        }

        def load_model_with_custom_objects(model_json_path, weights_path):
            with open(model_json_path, "r") as json_file:
                model_json = json_file.read()
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = model_from_json(model_json)
                model.load_weights(weights_path)
            return model

        self.fannet = load_model_with_custom_objects(
            "models/fannet.json", "models/fannet_weights.h5"
        )
        self.colornet = load_model_with_custom_objects(
            "models/colornet.json", "models/colornet_weights.h5"
        )

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

    def create_text_image(
        self, text: str, original_region: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        try:
            output = np.zeros((height, width, 3), dtype=np.uint8)
            x_offset = 0
            total_char_width = 0

            style_tensor = self.region_to_tensor(
                original_region
            )  # Shape: (1, 64, 64, 1)
            style_rgb = np.repeat(style_tensor, 3, axis=-1)  # Convert grayscale to RGB

            # First pass: calculate total width of all characters
            for char in text:
                char_tensor = self.char_to_tensor(char)
                fannet_output = self.fannet.predict([style_tensor, char_tensor])[0]
                fannet_output = np.expand_dims(fannet_output, axis=0)
                colornet_output = self.colornet.predict([style_rgb, fannet_output])[0]
                total_char_width += colornet_output.shape[1]

            # Calculate overall scale factor
            scale = min(width / total_char_width, height / colornet_output.shape[0])

            for char in text:
                char_tensor = self.char_to_tensor(char)
                fannet_output = self.fannet.predict(
                    {"input_1": style_tensor, "input_2": char_tensor}
                )[0]
                fannet_output = np.expand_dims(fannet_output, axis=0)
                colornet_output = self.colornet.predict(
                    {"input_1": style_rgb, "input_2": fannet_output}
                )[0]

                char_height, char_width = colornet_output.shape[:2]
                new_width = int(char_width * scale)
                new_height = int(char_height * scale)

                if new_width <= 0 or new_height <= 0:
                    self.logger.warning(
                        f"Invalid character size after scaling: {new_width}x{new_height}"
                    )
                    continue

                char_image = cv2.resize(
                    colornet_output,
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                y_offset = (height - new_height) // 2
                if x_offset + new_width > width:
                    self.logger.warning("Text exceeds image width. Truncating.")
                    break

                output[
                    y_offset : y_offset + new_height, x_offset : x_offset + new_width
                ] = char_image
                x_offset += new_width

            return output

        except Exception as e:
            self.logger.error(f"Error creating styled text image: {str(e)}")
            self.logger.exception("Exception details:")
            return original_region

    def char_to_tensor(self, char: str) -> np.ndarray:
        # Keep the original alphabet for FANNET compatibility
        fannet_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Extended alphabet for character recognition
        extended_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ^"

        onehot = [0.0] * len(fannet_alphabet)

        char = char.upper()
        if char in fannet_alphabet:
            onehot[fannet_alphabet.index(char)] = 1.0
        elif char in extended_alphabet:
            # For characters in extended alphabet but not in FANNET alphabet,
            # use a default representation (e.g., last letter 'Z')
            onehot[-1] = 1.0
            self.logger.info(
                f"Character '{char}' mapped to default FANNET representation."
            )
        else:
            self.logger.warning(
                f"Character '{char}' not found in extended alphabet. Using default."
            )
            onehot[-1] = 1.0  # Use 'Z' as default

        return np.array(onehot).reshape(1, 26, 1)  # Shape: (1, 26, 1)

    def region_to_tensor(self, region: np.ndarray) -> np.ndarray:
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_region, (64, 64))
        return np.expand_dims(resized, axis=(0, -1)) / 255.0  # Shape: (None, 64, 64, 1)

    def detect_text_batch(self, batch: FrameBatch) -> List[List[dict]]:
        all_results = []
        try:
            self.logger.info(f"Processing batch of {len(batch.frames)} frames")
            for frame_idx, frame in enumerate(batch.frames):
                self.logger.info(f"Processing frame {frame_idx}")
                frame_results = self.detect_text_easyocr_from_frame(frame)
                all_results.append(frame_results)
                self.logger.debug(
                    f"Frame {frame_idx}: detected {len(frame_results) if frame_results else 0} text regions"
                )

        except Exception as e:
            self.logger.error(f"Error in batch text detection: {str(e)}")
            self.logger.exception("Exception details:")
            all_results = [[] for _ in range(len(batch.frames))]

        return all_results

    def detect_text_easyocr_from_frame(self, frame):
        try:
            results = self.reader.readtext(frame)

            if not results:
                self.logger.info("No text detected in frame")
                return []

            self.logger.info(f"Detected {len(results)} text elements.")

            text_info = []
            for i, (bbox, text, conf) in enumerate(results, 1):
                text_info.append(
                    {"description": text, "bounding_box": bbox, "confidence": conf}
                )

                self.logger.debug(f"Text {i}: '{text}' (Confidence: {conf:.2f})")

            return text_info

        except Exception as e:
            self.logger.error(f"Error in EasyOCR text detection: {str(e)}")
            return []

    def preprocess_frame(self, frame):
        if frame.ndim == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3:  # BGR (OpenCV default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def translate_batch(
        self, texts_batch: List[List[Tuple[str, List[Tuple[float, float]]]]]
    ) -> List[List[str]]:
        """Translate multiple texts in batch."""
        try:
            flat_texts = [
                text for frame_texts in texts_batch for text, _ in frame_texts
            ]

            if not flat_texts:
                return [[]] * len(texts_batch)

            inputs = self.translator_tokenizer(
                flat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.translator_model.generate(**inputs)

            translations = self.translator_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

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
            text_regions = self.detect_text_batch(batch)
            texts_to_translate = [
                [(info["description"], info["bounding_box"]) for info in frame_regions]
                for frame_regions in text_regions
            ]
            translated_texts = self.translate_batch(texts_to_translate)

            processed_frames = []
            for frame_idx, frame in enumerate(batch.frames):
                try:
                    processed_frame = frame.copy()
                    frame_translations = translated_texts[frame_idx]
                    frame_regions = text_regions[frame_idx]

                    self.logger.debug(
                        f"Processing frame {frame_idx} with {len(frame_regions)} regions"
                    )

                    for region_info, translated_text in zip(
                        frame_regions, frame_translations
                    ):
                        try:
                            bounding_box = region_info["bounding_box"]
                            x_min, y_min = map(
                                int, min(bounding_box, key=lambda p: p[0] + p[1])
                            )
                            x_max, y_max = map(
                                int, max(bounding_box, key=lambda p: p[0] + p[1])
                            )

                            region_width = x_max - x_min
                            region_height = y_max - y_min

                            if region_width <= 0 or region_height <= 0:
                                self.logger.warning(
                                    f"Invalid region dimensions in frame {frame_idx}: {bounding_box}"
                                )
                                continue

                            original_region = processed_frame[y_min:y_max, x_min:x_max]
                            text_image = self.create_text_image(
                                translated_text,
                                original_region,
                                region_width,
                                region_height,
                            )
                            processed_frame[y_min:y_max, x_min:x_max] = text_image
                        except Exception as e:
                            self.logger.error(
                                f"Error processing region in frame {frame_idx}: {str(e)}"
                            )

                    processed_frames.append(processed_frame)
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                    processed_frames.append(frame)

            return processed_frames

        except Exception as e:
            self.logger.error(f"Error processing frame batch: {str(e)}")
            self.logger.exception("Exception details:")
            return batch.frames

    def process_video(self, input_path: str, output_path: str):
        """Process video with batch frame processing."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open input video")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                    if batch_frames:
                        batch = FrameBatch(frames=batch_frames, indices=batch_indices)
                        processed_frames = self.process_frame_batch(batch)
                        for processed_frame in processed_frames:
                            out.write(processed_frame)
                    break

                batch_frames.append(frame)
                batch_indices.append(frame_count)
                frame_count += 1

                if len(batch_frames) == self.batch_size:
                    batch = FrameBatch(frames=batch_frames, indices=batch_indices)
                    processed_frames = self.process_frame_batch(batch)

                    for processed_frame in processed_frames:
                        out.write(processed_frame)

                    batch_frames = []
                    batch_indices = []

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

                box = np.array(box).astype(np.int32)
                x_min = max(0, min(box[:, 0]))
                y_min = max(0, min(box[:, 1]))
                x_max = min(frame_width, max(box[:, 0]))
                y_max = min(frame_height, max(box[:, 1]))

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

                original_region = processed_frame[y_min:y_max, x_min:x_max]
                translated_text = self.translate_text(text)
                text_image = self.create_text_image(
                    translated_text, original_region, region_width, region_height
                )
                processed_frame[y_min:y_max, x_min:x_max] = text_image

            return processed_frame

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def process_single_frame(self, frame_path: str) -> np.ndarray:
        """Process a single frame."""
        try:
            # Load the image
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Could not read image from {frame_path}")

            # Process the frame using existing methods
            processed_frame = self.process_frame(frame)
            return processed_frame

        except Exception as e:
            self.logger.error(f"Error processing single frame: {str(e)}")
            self.logger.exception("Exception details:")
            return None


def add_audio_to_video(input_video_path, processed_video_path, output_video_path):
    command = (
        f'ffmpeg -i "{input_video_path}" -i "{processed_video_path}" '
        f'-c:v copy -c:a aac -map 0:a:0 -map 1:v:0 "{output_video_path}"'
    )
    result = os.system(command)
    if result != 0:
        logging.error("FFmpeg command failed")


def main():
    try:
        translator = VideoTextTranslator(batch_size=8)
        processed_video_path = "processed_no_audio.mp4"
        translator.process_video("input_video.mp4", processed_video_path)
        add_audio_to_video(
            "input_video.mp4", processed_video_path, "output_with_audio.mp4"
        )
        logging.info("Video processing and audio combining completed successfully.")

    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
        import sys

        sys.exit(1)
