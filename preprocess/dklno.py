# Classical matrix, works best
import os
import numpy as np
import cv2
import pandas as pd
import pickle
import json
from datetime import datetime
from scipy import stats


class DKLFeatureExtractor:
    def __init__(self, output_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL"):
        # RGB to LMS conversion matrix (classical Von Kries transform matrix)
        self.rgb_to_lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])

        # LMS to DKL conversion matrix (based on the literature matrix you provided)
        self.lms_to_dkl_matrix = np.array([
            [22.0454, 22.0454, 0.0],
            [13.8336, -32.4902, 0.0],
            [-12.7279, -12.7279, 9.6048]
        ])

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def rgb_to_lms(self, rgb_image):
        """Convert RGB image to LMS color space - without normalization"""
        # Use original RGB values directly, no normalization
        if rgb_image.dtype != np.float32:
            rgb_float = rgb_image.astype(np.float32)
        else:
            rgb_float = rgb_image.copy()

        original_shape = rgb_float.shape
        pixels = rgb_float.reshape(-1, 3)
        lms_pixels = np.dot(pixels, self.rgb_to_lms_matrix.T)

        # Add logarithmic transformation (according to the formula in the literature)
        epsilon = 1e-6  # avoid log of zero
        lms_pixels = np.log(lms_pixels + epsilon)    
        lms_image = lms_pixels.reshape(original_shape)

        return lms_image

    def lms_to_dkl(self, lms_image):
        """Convert LMS image to DKL color space"""
        original_shape = lms_image.shape
        pixels = lms_image.reshape(-1, 3)
        dkl_pixels = np.dot(pixels, self.lms_to_dkl_matrix.T)
        dkl_image = dkl_pixels.reshape(original_shape)

        return dkl_image

    def rgb_to_dkl(self, rgb_image):
        """Convert RGB image directly to DKL color space (via LMS intermediate conversion)"""
        lms_image = self.rgb_to_lms(rgb_image)
        dkl_image = self.lms_to_dkl(lms_image)
        return dkl_image

    def save_dkl_image(self, dkl_image, output_path):
        """Save DKL image (saved as 16-bit PNG, without normalization)"""
        # Directly save floating-point DKL image as 16-bit PNG
        dkl_16bit = np.zeros_like(dkl_image, dtype=np.uint16)

        # Scale floating-point values to 0-65535 range (for display only, does not affect raw data)
        for i in range(3):
            channel = dkl_image[:, :, i]
            min_val = channel.min()
            max_val = channel.max()
            # Linear scaling to 0-65535 (only for visualization)
            if max_val > min_val:
                normalized_channel = (channel - min_val) / (max_val - min_val) * 65535
            else:
                normalized_channel = np.zeros_like(channel)
            dkl_16bit[:, :, i] = normalized_channel.astype(np.uint16)

        # Save as 16-bit PNG
        cv2.imwrite(output_path, cv2.cvtColor(dkl_16bit, cv2.COLOR_RGB2BGR))

        # Also save the raw floating-point data as a .npy file
        npy_path = os.path.splitext(output_path)[0] + '.npy'
        np.save(npy_path, dkl_image)

        print(f"Saved DKL image: {output_path}")
        print(f"Saved raw data: {npy_path}")

        return output_path

    def extract_lms_statistical_features(self, lms_image):
        """Extract statistical features from LMS space"""
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

    def extract_dkl_statistical_features(self, dkl_image):
        """Extract statistical features from DKL space"""
        features = {}

        # DKL channel interpretation:
        # D channel = 22.0454×L + 22.0454×M + 0.0×S (luminance channel L+M)
        # K channel = 13.8336×L - 32.4902×M + 0.0×S (red-green opponent channel L-M)
        # L channel = -12.7279×L - 12.7279×M + 9.6048×S (yellow-blue opponent channel S-(L+M))

        D_channel = dkl_image[:, :, 0]  # luminance channel
        K_channel = dkl_image[:, :, 1]  # red-green opponent channel
        L_channel = dkl_image[:, :, 2]  # yellow-blue opponent channel

        channels = {
            'D_luminance': D_channel,
            'K_red_green': K_channel,
            'L_yellow_blue': L_channel
        }

        for channel_name, channel_data in channels.items():
            features[f'dkl_{channel_name}_mean'] = np.mean(channel_data)
            features[f'dkl_{channel_name}_std'] = np.std(channel_data)
            features[f'dkl_{channel_name}_skewness'] = stats.skew(channel_data.flatten())

        return features

    def extract_all_features(self, rgb_image):
        """Extract all LMS and DKL features"""
        # Convert to LMS space
        lms_image = self.rgb_to_lms(rgb_image)
        lms_features = self.extract_lms_statistical_features(lms_image)

        # Convert to DKL space
        dkl_image = self.lms_to_dkl(lms_image)
        dkl_features = self.extract_dkl_statistical_features(dkl_image)

        # Merge all features
        all_features = {**lms_features, **dkl_features}

        return all_features, lms_image, dkl_image

    def load_image_set(self, image_dirs, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """
        Load image set (supports multiple directories)
        """
        # If input is a single directory, convert to list
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]

        image_paths = []

        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                print(f"Warning: directory does not exist {image_dir}")
                continue

            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(root, file))

        print(f"Found {len(image_paths)} images in directories {image_dirs}")
        return sorted(image_paths)

    def extract_features_batch(self, image_paths, save_name=None, dataset_type="combined", save_dkl_images=True):
        """
        Batch extract features and save
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"lms_dkl_features_{timestamp}"

        all_features = []
        successful_images = []
        failed_images = []
        saved_dkl_images = []

        # Base DKL image save directory
        base_dkl_dir = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL"
        os.makedirs(base_dkl_dir, exist_ok=True)

        print(f"Starting processing of {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            try:
                # Read image
                rgb_image = cv2.imread(image_path)
                if rgb_image is None:
                    print(f"Warning: cannot read image {image_path}")
                    failed_images.append(image_path)
                    continue

                # Convert color space
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                # Extract features (now includes LMS and DKL)
                features, lms_image, dkl_image = self.extract_all_features(rgb_image)

                # Save DKL image - keep the same directory structure as the original image
                dkl_image_path = None
                if save_dkl_images:
                    # Get relative path relative to Image_set_Resize
                    base_source_dir = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize"
                    rel_path = os.path.relpath(image_path, base_source_dir)

                    # Build corresponding DKL image path
                    dkl_image_full_path = os.path.join(base_dkl_dir, rel_path)

                    # Change file extension to .png
                    dkl_dir = os.path.dirname(dkl_image_full_path)
                    original_filename = os.path.basename(dkl_image_full_path)
                    name_without_ext = os.path.splitext(original_filename)[0]
                    dkl_filename = f"{name_without_ext}.png"
                    dkl_image_final_path = os.path.join(dkl_dir, dkl_filename)

                    # Create target directory if it does not exist
                    os.makedirs(dkl_dir, exist_ok=True)

                    # Save DKL image
                    saved_path = self.save_dkl_image(dkl_image, dkl_image_final_path)
                    saved_dkl_images.append(dkl_image_final_path)
                    dkl_image_path = dkl_image_final_path

                # Add image information
                features['image_path'] = image_path
                features['image_name'] = os.path.basename(image_path)
                features['image_size'] = f"{rgb_image.shape[1]}x{rgb_image.shape[0]}"
                features['dataset_type'] = dataset_type
                if save_dkl_images:
                    features['dkl_image_path'] = dkl_image_path

                # Add subfolder information
                rel_path = os.path.relpath(image_path, "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize")
                features['relative_path'] = rel_path

                all_features.append(features)
                successful_images.append(image_path)

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    # Show example paths
                    if i == 0:
                        print(f"Example - Original: {image_path}")
                        print(f"Example - DKL: {dkl_image_path}")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                failed_images.append(image_path)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)

        # Save results
        self._save_features(features_df, save_name, successful_images, failed_images, saved_dkl_images, save_dkl_images)

        return features_df

    def _save_features(self, features_df, save_name, successful_images, failed_images, saved_dkl_images=None,
                       save_dkl_images=False):
        """Save feature data to files"""
        base_path = os.path.join(self.output_dir, save_name)

        # 1. Save as CSV file (recommended, good readability)
        csv_path = f"{base_path}.csv"
        features_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV file: {csv_path}")

        # 2. Save as JSON file
        json_path = f"{base_path}.json"
        features_dict = {
            'extraction_time': datetime.now().isoformat(),
            'total_images': len(successful_images) + len(failed_images),
            'successful_images': len(successful_images),
            'failed_images': len(failed_images),
            'features': features_df.to_dict('records'),
            'successful_list': successful_images,
            'failed_list': failed_images,
            'color_spaces': 'LMS and DKL',
            'dkl_channels_description': {
                'D_luminance': 'Luminance channel (L+M)',
                'K_red_green': 'Red-green opponent channel (L-M)',
                'L_yellow_blue': 'Yellow-blue opponent channel (S-(L+M))'
            }
        }

        # Add DKL image information
        if save_dkl_images and saved_dkl_images:
            features_dict['dkl_images_saved'] = len(saved_dkl_images)
            features_dict['dkl_images_directory'] = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL"
            features_dict['dkl_images_sample'] = saved_dkl_images[:5]  # Save first 5 as examples
            features_dict['dkl_image_format'] = '16-bit PNG + raw NPY data'

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(features_dict, f, ensure_ascii=False, indent=2)
        print(f"JSON file: {json_path}")

        # 3. Save as Pickle file (preserves full data structure)
        pkl_path = f"{base_path}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(features_df, f)
        print(f"Pickle file: {pkl_path}")

        print(f"\n=== Feature extraction complete ===")
        print(f"Successfully processed: {len(successful_images)} images")
        print(f"Failed: {len(failed_images)} images")
        print(f"Color spaces: LMS + DKL")
        print(f"DKL channels: Luminance(D), Red-Green opponent(K), Yellow-Blue opponent(L)")

        if save_dkl_images and saved_dkl_images:
            print(f"DKL images saved: {len(saved_dkl_images)}")
            print(f"DKL image directory structure maintained same as original")
            print(f"DKL image examples:")
            for i, path in enumerate(saved_dkl_images[:3]):
                print(f"  {i + 1}. {path}")

        if failed_images:
            print(f"\nList of failed images is saved in the JSON file.")


def find_image_directories():
    """Find actual image directories"""
    possible_base_dirs = [
        "/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-meg/Image_set_Resize",
        "/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-meg/Image_set_Resize"
    ]

    for base_dir in possible_base_dirs:
        if os.path.exists(base_dir):
            print(f"Found base directory: {base_dir}")

            # Check possible subdirectory structures
            possible_subdirs = [
                os.path.join(base_dir, "test_images"),
                os.path.join(base_dir, "train_images"),
                os.path.join(base_dir, "test"),
                os.path.join(base_dir, "train"),
                base_dir  # directly contains image files
            ]

            existing_dirs = []
            for subdir in possible_subdirs:
                if os.path.exists(subdir):
                    existing_dirs.append(subdir)
                    print(f"  Found subdirectory: {subdir}")

            if existing_dirs:
                return base_dir, existing_dirs

    print("Could not find image directories, please check the path.")
    return None, []


# Usage example
def main():
    """Complete usage example"""

    # 1. Create feature extractor - use specified path
    extractor = DKLFeatureExtractor(output_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg")

    # 2. Automatically find image directories
    print("Searching for image directories...")
    base_dir, image_dirs = find_image_directories()

    if not image_dirs:
        print("No image directories found!")
        return

    # 3. Load image set
    image_paths = extractor.load_image_set(image_dirs)

    if not image_paths:
        print("No image files found in the specified directories!")
        return

    # 4. Batch extract features and save (now also saves DKL images)
    save_name = "Image_set_Resize_LMS_DKL_Features"

    features_df = extractor.extract_features_batch(
        image_paths=image_paths,
        save_name=save_name,
        dataset_type="combined_test_train",
        save_dkl_images=True  # Enable DKL image saving
    )

    # 5. Display result summary
    print(f"\n=== Feature data summary ===")
    print(f"Total features: {len(features_df.columns) - 6}")  # subtract information columns (now one more: dkl_image_path)
    print(f"Feature dimensions: {len(features_df)} samples x {len(features_df.columns)} features")

    # Show feature type statistics
    lms_features = [col for col in features_df.columns if
                    not col.startswith('dkl_') and col not in ['image_path', 'image_name', 'image_size', 'dataset_type',
                                                               'relative_path', 'dkl_image_path']]
    dkl_features = [col for col in features_df.columns if col.startswith('dkl_')]

    print(f"LMS features: {len(lms_features)}")
    print(f"DKL features: {len(dkl_features)}")
    print(f"First 5 LMS features: {lms_features[:5]}")
    print(f"First 5 DKL features: {dkl_features[:5]}")

    # Show save path information
    print(f"\n=== File save information ===")
    print(f"Feature files saved to: /data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg")
    print(f"DKL images saved to: /data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL")
    print(f"Directory structure maintained same as original")

    return features_df


# Functions to process train and test sets separately
def process_separately():
    """Process train and test sets separately"""
    extractor = DKLFeatureExtractor(output_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg")

    base_dir, image_dirs = find_image_directories()

    if not image_dirs:
        print("No image directories found!")
        return None, None

    # Process test set
    print("=== Processing test set ===")
    test_paths = []
    for dir_path in image_dirs:
        if 'test' in dir_path.lower():
            test_paths.extend(extractor.load_image_set(dir_path))

    if test_paths:
        test_features = extractor.extract_features_batch(
            test_paths,
            "Test_Set_LMS_DKL_Features",
            "test",
            save_dkl_images=True
        )
    else:
        print("No test set images found.")
        test_features = None

    # Process training set
    print("\n=== Processing training set ===")
    train_paths = []
    for dir_path in image_dirs:
        if 'train' in dir_path.lower():
            train_paths.extend(extractor.load_image_set(dir_path))

    if train_paths:
        train_features = extractor.extract_features_batch(
            train_paths,
            "Train_Set_LMS_DKL_Features",
            "train",
            save_dkl_images=True
        )
    else:
        print("No training set images found.")
        train_features = None

    return test_features, train_features


# Function to only save DKL images without extracting features
def save_dkl_images_only():
    """Only save DKL images without extracting features"""
    extractor = DKLFeatureExtractor(output_dir="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg")

    base_dir, image_dirs = find_image_directories()

    if not image_dirs:
        print("No image directories found!")
        return

    image_paths = extractor.load_image_set(image_dirs)

    if not image_paths:
        print("No image files found in the specified directories!")
        return

    print(f"Starting to save {len(image_paths)} DKL images...")

    saved_count = 0
    for i, image_path in enumerate(image_paths):
        try:
            # Read image
            rgb_image = cv2.imread(image_path)
            if rgb_image is None:
                print(f"Warning: cannot read image {image_path}")
                continue

            # Convert color space
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Convert to DKL and save
            dkl_image = extractor.rgb_to_dkl(rgb_image)

            # Build save path
            base_source_dir = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/things-meg/Image_set_Resize/"
            base_dkl_dir = "/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL"
            rel_path = os.path.relpath(image_path, base_source_dir)
            dkl_image_full_path = os.path.join(base_dkl_dir, rel_path)
            dkl_dir = os.path.dirname(dkl_image_full_path)
            original_filename = os.path.basename(dkl_image_full_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            dkl_filename = f"{name_without_ext}.png"
            dkl_image_final_path = os.path.join(dkl_dir, dkl_filename)
            os.makedirs(dkl_dir, exist_ok=True)

            saved_path = extractor.save_dkl_image(dkl_image, dkl_image_final_path)

            if saved_path:
                saved_count += 1
                if (i + 1) % 10 == 0:
                    print(f"Saved {i + 1}/{len(image_paths)} DKL images")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"DKL image saving completed! Successfully saved {saved_count}/{len(image_paths)} images.")
    print(f"Save directory: /data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL")


if __name__ == "__main__":
    print("Process all images together")
    features = main()
