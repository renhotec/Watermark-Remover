import os
from PIL import Image
import sys
from concurrent.futures import ThreadPoolExecutor
import time


class WatermarkOperator:
    def __init__(self, watermark_path, folder_path):
        self.watermark_path = watermark_path
        self.folder_path = folder_path
        self.watermark = Image.open(watermark_path).convert("RGBA")
        self.output_folder = folder_path + "-watermarked"
        os.makedirs(self.output_folder, exist_ok=True)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGBA")

        # Resize watermark to match the size of the image
        watermark_resized = self.watermark.resize(image.size, Image.LANCZOS)

        # Composite the watermark with the image
        watermarked_image = Image.alpha_composite(image, watermark_resized)

        # Save the watermarked image to the new folder
        output_path = os.path.join(self.output_folder, os.path.basename(image_path))
        # save as jpg
        watermarked_image.convert("RGB").save(output_path, "JPEG")

    def add_watermark(self):
        start_time = time.time()

        image_paths = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        total_files = len(image_paths)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, image_path)
                for image_path in image_paths
            ]
            for i, future in enumerate(futures):
                future.result()
                print(f"Add watermark to {i + 1}/{total_files} images", end="\r")

        end_time = time.time()
        print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

class ImageOperator:
    def __init__(self, size=(900, 900)):
        self.size = size

    def resize_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail(self.size, Image.LANCZOS)
                if img.size[0] < self.size[0] or img.size[1] < self.size[1]:
                    new_img = Image.new("RGB", self.size, (255, 255, 255))
                    new_img.paste(img, ((self.size[0] - img.size[0]) // 2, (self.size[1] - img.size[1]) // 2))
                    new_img.save(image_path)
                else:
                    img.save(image_path)
        except Exception as e:
            print(f"Failed to resize image {image_path}: {e}")

    def resize_images_in_directory(self, directory):
        def process_image(filename):
            if os.path.getsize(os.path.join(directory, filename)) == 0:
                os.remove(os.path.join(directory, filename))
                return None
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_path = os.path.join(directory, filename)
                self.resize_image(file_path)
                return filename

        total_files = len([f for f in os.listdir(directory) if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))])
        processed_files = 0

        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, filename) for filename in os.listdir(directory)]
            for future in futures:
                if future.result():
                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    speed = processed_files / elapsed_time if elapsed_time > 0 else 0
                    print(f"\rResized {processed_files}/{total_files} files at {speed:.2f} files/second", end="")

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        print(f"\nTotal time taken: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_operator.py <directory> [width] [watermark file]")
        sys.exit(1)

    directory = sys.argv[1]
    directory = directory.replace(".\\", "").rstrip("\\")

    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        sys.exit(1)

    if len(sys.argv) > 2:
        width = int(sys.argv[2])
    else:
        width = 900

    size = (width, width)
    image_operator = ImageOperator(size)
    image_operator.resize_images_in_directory(directory)

    if len(sys.argv) > 3:
        watermark_file = sys.argv[3]
        watermark_file = watermark_file.replace(".\\", "")

        if not os.path.isfile(watermark_file):
            print(f"The file {watermark_file} does not exist.")
            sys.exit(1)

        operator = WatermarkOperator(watermark_file, directory)
        operator.add_watermark()
