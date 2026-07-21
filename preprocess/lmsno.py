import os
import numpy as np
import cv2
import pandas as pd
import pickle
import json
from datetime import datetime
from scipy import stats
from tqdm import tqdm


class LMSFeatureExtractor:
    def __init__(self, output_dir="lms_features",
                 save_images_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_LMS"):
        # RGB to LMS conversion matrix (classic Von Kries transformation)
        self.rgb_to_lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])

        # Create output directories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create directory for saving LMS images
        self.save_images_dir = save_images_dir
        os.makedirs(save_images_dir, exist_ok=True)

    def rgb_to_lms(self, rgb_image):
        """Convert RGB image to LMS color space"""
        if rgb_image.dtype != np.float32:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.copy()

        original_shape = rgb_normalized.shape
        pixels = rgb_normalized.reshape(-1, 3)
        lms_pixels = np.dot(pixels, self.rgb_to_lms_matrix.T)

        # Optional log transformation (commented out)
        # epsilon = 1e-6
        # lms_pixels = np.log(lms_pixels + epsilon)
        lms_image = lms_pixels.reshape(original_shape)

        return lms_image

    def print_lms_pixel_values(self, lms_image, num_pixels=10):
        """Print pixel values of the LMS image"""
        print("\n=== LMS Image Pixel Values ===")
        print(f"Image shape: {lms_image.shape}")
        print(f"Data type: {lms_image.dtype}")
        print(f"Value range: L[{lms_image[:, :, 0].min():.4f}, {lms_image[:, :, 0].max():.4f}], "
              f"M[{lms_image[:, :, 1].min():.4f}, {lms_image[:, :, 1].max():.4f}], "
              f"S[{lms_image[:, :, 2].min():.4f}, {lms_image[:, :, 2].max():.4f}]")

        print(f"\nFirst {num_pixels} pixel LMS values:")
        print("Coordinates\tL value\t\tM value\t\tS value")
        print("-" * 50)

        height, width = lms_image.shape[:2]
        count = 0

        for y in range(min(5, height)):
            for x in range(min(5, width)):
                if count >= num_pixels:
                    break
                l_val = lms_image[y, x, 0]
                m_val = lms_image[y, x, 1]
                s_val = lms_image[y, x, 2]
                print(f"({y},{x})\t\t{l_val:.6f}\t{m_val:.6f}\t{s_val:.6f}")
                count += 1

    def analyze_lms_channels(self, lms_image):
        """Analyze statistical information of LMS channels"""
        print("\n=== LMS Channel Statistical Analysis ===")

        L_channel = lms_image[:, :, 0]
        M_channel = lms_image[:, :, 1]
        S_channel = lms_image[:, :, 2]

        channels = {'L (long)': L_channel, 'M (medium)': M_channel, 'S (short)': S_channel}

        for name, channel in channels.items():
            print(f"\n{name} channel:")
            print(f"  Min: {channel.min():.6f}")
            print(f"  Max: {channel.max():.6f}")
            print(f"  Mean: {channel.mean():.6f}")
            print(f"  Std: {channel.std():.6f}")
            print(f"  Median: {np.median(channel):.6f}")

    def extract_lms_statistical_features(self, lms_image):
        """Extract statistical features in LMS space"""
        features = {}

        L_channel = lms_image[:, :, 0]
        M_channel = lms_image[:, :, 1]
        S_channel = lms_image[:, :, 2]

        channels = {'L': L_channel, 'M': M_channel, 'S': S_channel}

        for channel_name, channel_data in channels.items():
            features[f'{channel_name}_mean'] = np.mean(channel_data)
            features[f'{channel_name}_std'] = np.std(channel_data)
            features[f'{channel_name}_skewness'] = stats.skew(channel_data.flatten())

        return features

    def save_lms_image(self, lms_image, original_image_path):
        """
        Save LMS image to the specified directory while preserving original directory structure
        """
        try:
            # Extract relative path from original path
            base_source_dir = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize"

            if original_image_path.startswith(base_source_dir):
                rel_path = os.path.relpath(original_image_path, base_source_dir)
            else:
                # If not in the base directory, use only the filename
                rel_path = os.path.basename(original_image_path)

            # Build new save path
            save_path = os.path.join(self.save_images_dir, rel_path)

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Convert LMS image to 8-bit unsigned integer for saving
            lms_normalized = np.zeros_like(lms_image)
            for i in range(3):
                channel = lms_image[:, :, i]
                # Normalize to 0-255
                lms_normalized[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)

            lms_uint8 = lms_normalized.astype(np.uint8)

            # Save image (convert RGB to BGR for OpenCV)
            cv2.imwrite(save_path, cv2.cvtColor(lms_uint8, cv2.COLOR_RGB2BGR))

            return save_path

        except Exception as e:
            print(f"Error saving LMS image: {e}")
            return None

    def extract_all_features(self, rgb_image, original_image_path=None):
        """Extract all LMS features, optionally save LMS image"""
        lms_image = self.rgb_to_lms(rgb_image)

        # Print LMS pixel values
        self.print_lms_pixel_values(lms_image, num_pixels=25)

        # Analyze LMS channels
        self.analyze_lms_channels(lms_image)

        # If original image path is provided, save the LMS image
        if original_image_path is not None:
            saved_path = self.save_lms_image(lms_image, original_image_path)
            if saved_path:
                print(f"\nLMS image saved: {saved_path}")

        statistical_features = self.extract_lms_statistical_features(lms_image)
        return statistical_features, lms_image

    def load_first_image(self, image_dirs, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """
        Load the first image found in the given directories
        """
        # If input is a single directory, convert to list
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]

        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                print(f"Warning: directory does not exist {image_dir}")
                continue

            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_path = os.path.join(root, file)
                        return image_path

        return None


def analyze_single_image():
    """Analyze LMS conversion for a single image"""
    # 1. Create feature extractor
    extractor = LMSFeatureExtractor(
        output_dir="lms_features",
        save_images_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_LMS"
    )

    # 2. Specify image set directories
    base_directory = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize"
    test_directory = os.path.join(base_directory, "test_images")
    train_directory = os.path.join(base_directory, "training_images")

    # Check if directories exist
    if not os.path.exists(test_directory):
        print(f"Test directory does not exist: {test_directory}")
        return
    if not os.path.exists(train_directory):
        print(f"Train directory does not exist: {train_directory}")
        return

    # 3. Load the first image
    print("Searching for the first image...")
    first_image_path = extractor.load_first_image([test_directory, train_directory])

    if not first_image_path:
        print("No image files found in the specified directories!")
        return

    print(f"Found image: {first_image_path}")

    # 4. Read and display original image info
    print("\n=== Original Image Information ===")
    rgb_image = cv2.imread(first_image_path)
    if rgb_image is None:
        print(f"Unable to read image: {first_image_path}")
        return

    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"Image size: {rgb_image_rgb.shape}")
    print(f"Data type: {rgb_image_rgb.dtype}")
    print(f"Value range: [{rgb_image_rgb.min()}, {rgb_image_rgb.max()}]")

    # Print first few pixel values of original image
    print(f"\nFirst 9 pixel values of original RGB image:")
    print("Coordinates\tR\t\tG\t\tB")
    print("-" * 50)
    for y in range(3):
        for x in range(3):
            r_val = rgb_image_rgb[y, x, 0]
            g_val = rgb_image_rgb[y, x, 1]
            b_val = rgb_image_rgb[y, x, 2]
            print(f"({y},{x})\t\t{r_val}\t\t{g_val}\t\t{b_val}")

    # 5. Extract features and analyze LMS values
    print("\nStarting LMS conversion analysis...")
    features, lms_image = extractor.extract_all_features(rgb_image_rgb, first_image_path)

    # 6. Display feature summary
    print("\n=== Statistical Feature Summary ===")
    for key, value in features.items():
        print(f"{key}: {value:.6f}")

    return first_image_path, rgb_image_rgb, lms_image, features


def compare_rgb_lms_values(rgb_image, lms_image):
    """Compare RGB and LMS values side by side"""
    print("\n=== RGB vs LMS Value Comparison ===")
    print("Coordinates\tRGB values\t\tLMS values")
    print("-" * 70)

    for y in range(3):
        for x in range(3):
            r, g, b = rgb_image[y, x]
            l, m, s = lms_image[y, x]
            print(f"({y},{x})\t\t({r:3d},{g:3d},{b:3d})\t\t({l:.4f},{m:.4f},{s:.4f})")

def batch_process_images():
    """Batch process all images"""
    # Create feature extractor
    extractor = LMSFeatureExtractor(
        output_dir="lms_features",
        save_images_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_LMS"
    )
    
    # Specify image set directories
    base_directory = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize"
    test_directory = os.path.join(base_directory, "test_images")
    train_directory = os.path.join(base_directory, "training_images")
    
    # Supported image formats
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Statistics counters
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Process both test and training sets
    for dataset_dir in [test_directory, train_directory]:
        print(f"\n=== Processing directory: {dataset_dir} ===")
        
        # Walk through all subdirectories and files
        for root, dirs, files in os.walk(dataset_dir):
            for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
                # Check file extension
                if any(file.lower().endswith(ext) for ext in extensions):
                    total_images += 1
                    image_path = os.path.join(root, file)
                    
                    try:
                        # Read image
                        rgb_image = cv2.imread(image_path)
                        if rgb_image is None:
                            print(f"Unable to read image: {image_path}")
                            failed_images += 1
                            continue
                        
                        # Convert to RGB (OpenCV reads as BGR)
                        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                        
                        # Convert to LMS and save
                        lms_image = extractor.rgb_to_lms(rgb_image_rgb)
                        
                        # Save LMS image
                        saved_path = extractor.save_lms_image(lms_image, image_path)
                        
                        if saved_path:
                            processed_images += 1
                            if processed_images % 100 == 0:  # Print progress every 100 images
                                print(f"Processed {processed_images} images, latest saved: {saved_path}")
                        else:
                            failed_images += 1
                            print(f"Failed to save: {image_path}")
                            
                    except Exception as e:
                        failed_images += 1
                        print(f"Error processing {image_path}: {str(e)}")
    
    # Print summary
    print(f"\n=== Processing Complete ===")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {failed_images}")
    
    # Save processing log
    log_data = {
        'total_images': total_images,
        'processed_images': processed_images,
        'failed_images': failed_images,
        'output_dir': extractor.save_images_dir,
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    log_file = os.path.join(extractor.output_dir, 'batch_process.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Processing log saved to: {log_file}")


if __name__ == "__main__":
    print("=== LMS Image Batch Conversion ===")
    
    # Select mode
    mode = input("Select mode (1=single image analysis, 2=batch conversion): ")
    
    if mode == "1":
        # Single image analysis
        result = analyze_single_image()
        if result:
            first_image_path, rgb_image, lms_image, features = result
            compare_rgb_lms_values(rgb_image, lms_image)
            print(f"\nAnalysis complete! First image path: {first_image_path}")
    elif mode == "2":
        # Batch convert all images
        batch_process_images()
    else:
        print("Invalid choice!")