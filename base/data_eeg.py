import torch, os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import scipy.signal as signal

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .utils import instantiate_from_config, get_device


def load_eeg_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')

    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')
        train_dataset = EEGDataset(config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False,
                                 drop_last=False, num_workers=25, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True,
                                  drop_last=False, num_workers=32, pin_memory=True)
        return train_loader, test_loader, test_loader

    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')

        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config, mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False,
                                 drop_last=False, num_workers=25)  # , pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False,
                                drop_last=False, num_workers=32)  # , pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True,
                                  drop_last=False, num_workers=32)  # , pin_memory=True)
        return train_loader, val_loader, test_loader

class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.img_directories = [
            '/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-eeg/Image_set_LMSNO',
            '/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-eeg/Image_set_DKLNO',
            '/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-eeg/Image_set_Resize_CIELUV',
            '/data2/ww/Uncertainty-aware-Blur-Prior-main/data/things-eeg/Image_set_Resize'  
        ]
        
        self.img_dir_names = ['LMS', 'DKL', 'LUV', 'RGB']  

        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                         'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                         'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                         'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels

        self.avg = config['data'][f"{mode}_avg"]  # whether to average EEG trials

        self.blur_type = config['data']['blur_type']

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80

        # Generate data file paths for all subjects for the given mode (train/test)
        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        # Load all .pt data files for all subjects using load_data, store as list
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        # Modify feature cache directory to multi-modality version
        data_dir = os.path.join(self.data_dir, '../Union_Image_feature_updated',
                                f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}")
        os.makedirs(data_dir, exist_ok=True)

        features_filename = os.path.join(data_dir, f"{self.name}_{mode}.pt")  # cache file path for image features

        pretrain_map = {
            'RN50': {'pretrained': 'openai', 'resize': (224, 224)},  # 1024
            'RN101': {'pretrained': 'openai', 'resize': (224, 224)},  # 512
            'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},  # 512
            'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224)},  # 512
            'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224)},  # 768
            'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224)},  # 1024
            'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},  # 1024
            'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224)},  # 1280
        }

        # Instantiate uncertainty-aware blur transforms
        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift, tag in zip([-self.c, 0, self.c], ['low', 'medium', 'high']):
                blur_param = config['data']['blur_type']
                blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size'] + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])

        # Define image post-processing steps (ToTensor + normalization)
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                                    std=(0.26862954, 0.26130258, 0.27577711))]
        self.process_transform = transforms.Compose(process_term)

        # Initialize all sample match labels to 1 (corresponding to "medium" blur level)
        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

        # Add current modality attribute, default None, will be set dynamically during training
        self.current_modality = None

        # If feature cache file exists, load image features directly, no recomputation needed
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=False)
            self.img_features = saved_features['img_features']
        # Feature generation and caching (when cache does not exist)
        else:
            device = get_device('auto')
            # Load the specified CLIP model and pretrained weights
            self.vlmodel, self.preprocess, _ = open_clip.create_model_and_transforms(self.model_type,
                                                                                     device=f"cuda:{device}",
                                                                                     pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-bigG-14/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin")
            #pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-B/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin"
            #pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-B/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"
            #pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-L-14/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin"
            #pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-g-14/models--laion--CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin"
            #pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/ViT-bigG-14/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin"

            # pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/RN101/open_clip_model.safetensors"
            #pretrained="/data2/ww/.cache/huggingface/hub/models--timm--resnet50_clip.openai/snapshots/ec3d92c/open_clip_pytorch_model.bin"
            
            # Freeze parameters to ensure model is used only for inference, no weight updates
            for param in self.vlmodel.parameters():
                param.requires_grad = False

            self.vlmodel.eval()                         
            # Uncertainty-aware image feature generation
            if self.config['data']['uncertainty_aware']:
                self.img_features = {}
                for tag in ['low', 'medium', 'high']:
                    # Extract features for each image directory
                    modality_features = {}
                    for img_dir, modality_name in zip(self.img_directories, self.img_dir_names):
                        modality_features[modality_name] = self.ImageEncoder(self.loaded_data[0]['img'],
                                                                             self.blur_transform[tag], img_dir)
                    self.img_features[tag] = modality_features

                # Compute average features
                avg_features = {}
                for modality in self.img_dir_names:
                    avg_features[modality] = {
                        k: (sum(self.img_features[tag][modality][k] for tag in ['low', 'medium', 'high']) / 3) for k in
                        self.img_features['medium'][modality]}
                self.img_features['avg'] = avg_features
            else:
                # Extract features for each image directory
                self.img_features = {}
                for img_dir, modality_name in zip(self.img_directories, self.img_dir_names):
                    self.img_features[modality_name] = self.ImageEncoder(self.loaded_data[0]['img'], None, img_dir)

            # Save cache to file for future use
            torch.save({
                'img_features': self.img_features,
            }, features_filename)
            # Clean up resources, delete model, clear GPU cache, free memory
            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = loaded_data['text'][:, 0]

            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data

        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None, img_directory=None):
        if blur_transform == None:
            blur_transform = self.blur_transform
        if img_directory is None:
            img_directory = self.img_directories[0]

        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 128
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]

            device = next(self.vlmodel.parameters()).device

            # Build image path list
            image_paths = []
            for img in batch_images:
                if 'DKL' in img_directory:
                    # DKL modality: replace .jpg with .png
                    img_filename = os.path.splitext(img)[0] + '.png'
                    image_paths.append(os.path.join(img_directory, img_filename))
                elif 'LMS' in img_directory:
                    # LMS modality: replace .jpg with .png
                    img_filename = os.path.splitext(img)[0] + '.png'
                    image_paths.append(os.path.join(img_directory, img_filename))
                elif 'RGB' in img_directory:
                    # RGB modality: keep original filename, assume .jpg format
                    image_paths.append(os.path.join(img_directory, img))
                else:    
                    # Other modalities: keep original filename
                    image_paths.append(os.path.join(img_directory, img))

            ele = [self.process_transform(blur_transform(Image.open(img_path).convert("RGB")))
                   for img_path in image_paths]

            image_inputs = torch.stack(ele).to(device)

            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict

    def set_modality(self, modality_name):
        """Set the current image modality to use"""
        if modality_name in self.img_dir_names:
            self.current_modality = modality_name
        else:
            raise ValueError(f"Invalid modality: {modality_name}. Available modalities: {self.img_dir_names}")

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]

        img = 'None'
        match_label = self.match_label[index]

        # Decide which features to return based on whether current modality is set
        if self.current_modality is not None:
            # Single modality mode: return only features for the current modality, using unified key 'img_features'
            if self.config['data']['uncertainty_aware']:
                if self.mode == 'train':
                    if match_label == 0:
                        tag = 'low'
                    elif match_label == 2:
                        tag = 'high'
                    else:
                        tag = 'medium'
                else:
                    tag = 'medium'
                img_features = self.img_features[tag][self.current_modality][img_path]
            else:
                img_features = self.img_features[self.current_modality][img_path]

            sample = {
                'idx': index,
                'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
                'label': label,
                'img_path': img_path,
                'img': img,
                'text': f"This is a {self.loaded_data[subject]['text'][trial_index]}.",
                'session': self.loaded_data[subject]['session'][trial_index],
                'subject': subject,
                'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
                'img_features': img_features  # unified key name
            }
        else:
            # Multi-modality mode: return features for all modalities (for ensemble testing)
            if self.config['data']['uncertainty_aware']:
                if self.mode == 'train':
                    if match_label == 0:
                        tag = 'low'
                    elif match_label == 2:
                        tag = 'high'
                    else:
                        tag = 'medium'
                else:
                    tag = 'medium'
                # Return image features for all modalities
                img_features_dict = {}
                for modality in self.img_dir_names:
                    img_features_dict[f'img_features_{modality}'] = self.img_features[tag][modality][img_path]
            else:
                # Return image features for all modalities
                img_features_dict = {}
                for modality in self.img_dir_names:
                    img_features_dict[f'img_features_{modality}'] = self.img_features[modality][img_path]

            sample = {
                'idx': index,
                'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
                'label': label,
                'img_path': img_path,
                'img': img,
                'text': f"This is a {self.loaded_data[subject]['text'][trial_index]}.",
                'session': self.loaded_data[subject]['session'][trial_index],
                'subject': subject,
                'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
            }
            # Add image features for all modalities to the sample
            sample.update(img_features_dict)

        return sample

    def __len__(self):
        return self.trial_all_subjects