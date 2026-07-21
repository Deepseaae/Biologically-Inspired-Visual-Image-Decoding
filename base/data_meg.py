import torch
import os
import logging
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm

import open_clip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from base.utils import instantiate_from_config, get_device


def load_meg_data(config):
    """
    Returns: train_loader, val_loader, test_loader
    - intra-subject: val_loader reuses test_loader
    - inter-subject: build train/val/test by leave-one-subject
    """
    exp_setting = config.get('exp_setting', 'intra-subject')

    if exp_setting == 'intra-subject':
        test_dataset = MEGDataset(config, mode='test')
        print('init test_dataset success')
        train_dataset = MEGDataset(config, mode='train')
        print('init train_dataset success')

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['test_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=config['data'].get('num_workers_test', 8),
            pin_memory=config['data'].get('pin_memory', True)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['train_batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=config['data'].get('num_workers_train', 8),
            pin_memory=config['data'].get('pin_memory', True)
        )
        return train_loader, test_loader, test_loader

    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']

        test_dataset = MEGDataset(config, mode='test')
        print('init test_dataset success')

        all_subjects = [f'sub-{i:02}' for i in range(1, 5)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))

        leave_one_subjects_config = config.copy()
        leave_one_subjects_config['data'] = config['data'].copy()
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects

        val_dataset = MEGDataset(leave_one_subjects_config, mode='test')
        print('init val_dataset success')
        train_dataset = MEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['test_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=config['data'].get('num_workers_test', 8),
            pin_memory=config['data'].get('pin_memory', True)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['val_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=config['data'].get('num_workers_val', 8),
            pin_memory=config['data'].get('pin_memory', True)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['train_batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=config['data'].get('num_workers_train', 8),
            pin_memory=config['data'].get('pin_memory', True)
        )
        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unsupported exp_setting: {exp_setting}")


class MEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']

        # Four modality directories (adjust according to your actual MEG paths)
        self.img_directories = [
            '/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_LMS',
            '/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_DKL',
            '/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_CIELUV',
            '/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/things-meg/Image_set_Resize'
        ]
        self.img_dir_names = ['LMS', 'DKL', 'LUV', 'RGB']

        self.subjects = config['data']['subjects']
        print(f'subjects: {self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']

        # MEG usually does not filter by channel names; keep compatibility
        self.channels = None
        if self.selected_ch == "None":
            self.selected_ch = None

        self.avg = config['data'][f"{mode}_avg"]
        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 1 if self.mode == 'train' else 12

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        # Whether text field exists
        self.has_text = ('text' in self.loaded_data[0])

        # Multi-modal classification cache path
        data_dir = os.path.join(
            self.data_dir,
            '../MultiModal_Image_feature_classification',
            f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}"
        )
        os.makedirs(data_dir, exist_ok=True)
        features_filename = os.path.join(data_dir, f"{self.name}_{mode}.pt")

        # uncertainty-aware blur
        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift, tag in zip([-self.c, 0, self.c], ['low', 'medium', 'high']):
                blur_param = dict(config['data']['blur_type'])
                blur_param['params'] = dict(config['data']['blur_type']['params'])
                blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size'] + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])

        self.process_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        self.match_label = np.ones(self.trial_all_subjects, dtype=int)
        self.current_modality = None

        # Default values for text related fields
        self.text_features = None
        self.text_matrix = None
        self.class_to_idx = None

        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=False)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features.get('text_features', None)
            self.text_matrix = saved_features.get('text_matrix', None)
            self.class_to_idx = saved_features.get('class_to_idx', None)
            print(f"load cached features from: {features_filename}")
        else:
            print(f"cache not found, extracting features -> {features_filename}")
            dev = get_device('auto')
            self.vlmodel, self.preprocess, _ = open_clip.create_model_and_transforms(
                self.model_type,
                device=f"cuda:{dev}",
                pretrained="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/resnet50_clip.openai/snapshots/ec3d92c/open_clip_pytorch_model.bin"
            )

            for p in self.vlmodel.parameters():
                p.requires_grad = False
            self.vlmodel.eval()

            # Image features
            if self.config['data']['uncertainty_aware']:
                self.img_features = {}
                for tag in ['low', 'medium', 'high']:
                    modality_features = {}
                    for img_dir, modality_name in zip(self.img_directories, self.img_dir_names):
                        modality_features[modality_name] = self.ImageEncoder(
                            self.loaded_data[0]['img'],
                            blur_transform=self.blur_transform[tag],
                            img_directory=img_dir
                        )
                    self.img_features[tag] = modality_features

                avg_features = {}
                for modality in self.img_dir_names:
                    avg_features[modality] = {
                        k: (self.img_features['low'][modality][k]
                            + self.img_features['medium'][modality][k]
                            + self.img_features['high'][modality][k]) / 3.0
                        for k in self.img_features['medium'][modality]
                    }
                self.img_features['avg'] = avg_features
            else:
                self.img_features = {}
                for img_dir, modality_name in zip(self.img_directories, self.img_dir_names):
                    self.img_features[modality_name] = self.ImageEncoder(
                        self.loaded_data[0]['img'],
                        blur_transform=None,
                        img_directory=img_dir
                    )

            # Text features (if data contains text)
            if self.has_text:
                self.text_features, self.text_matrix, self.class_to_idx = self.TextEncoder(self.loaded_data[0]['text'])

            torch.save({
                'img_features': self.img_features,
                'text_features': self.text_features,
                'text_matrix': self.text_matrix,
                'class_to_idx': self.class_to_idx
            }, features_filename)

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self, data_path):
        logging.info(f"---- load {data_path} ----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch and self.channels is not None:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]

        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['img'] = np.array(loaded_data['img'])
            if 'label' in loaded_data:
                avg_data['label'] = loaded_data['label'][:, 0] if loaded_data['label'].ndim > 1 else loaded_data['label']
            if 'text' in loaded_data:
                avg_data['text'] = loaded_data['text'][:, 0] if loaded_data['text'].ndim > 1 else loaded_data['text']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['img'] = loaded_data['img'].reshape(-1)
            if 'label' in loaded_data:
                _data['label'] = loaded_data['label'].reshape(-1)
            if 'text' in loaded_data:
                _data['text'] = loaded_data['text'].reshape(-1)
            loaded_data = _data

        return loaded_data

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None, img_directory=None):
        if blur_transform is None:
            blur_transform = self.blur_transform
        if img_directory is None:
            img_directory = self.img_directories[0]

        self.vlmodel.eval()
        set_images = sorted(list(set(images)))
        batch_size = 128
        image_features_list = []

        for i in tqdm(range(0, len(set_images), batch_size), desc=f"Encoding {img_directory}"):
            batch_images = set_images[i:i + batch_size]
            dev = next(self.vlmodel.parameters()).device

            image_paths = []
            for img in batch_images:
                # Path rules for different modality directories
                if 'DKL' in img_directory or 'CIELUV' in img_directory:
                    img_filename = os.path.splitext(img)[0] + '.png'
                    image_paths.append(os.path.join(img_directory, img_filename))
                elif 'RGB' in img_directory or 'LMS' in img_directory:
                    image_paths.append(os.path.join(img_directory, img))
                else:
                    image_paths.append(os.path.join(img_directory, img))

            ele = [self.process_transform(blur_transform(Image.open(p).convert("RGB"))) for p in image_paths]
            image_inputs = torch.stack(ele).to(dev)

            batch_feats = self.vlmodel.encode_image(image_inputs)
            batch_feats = batch_feats / batch_feats.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_feats)

        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict

    @torch.no_grad()
    def TextEncoder(self, texts):
        self.vlmodel.eval()
        class_names = sorted(list(set(texts)))
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        prompts = [f"This is a {c}." for c in class_names]
        text_inputs = torch.cat([open_clip.tokenize(p) for p in prompts])

        dev = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(dev)

        text_feats = self.vlmodel.encode_text(text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        text_features = {class_names[i]: text_feats[i].float().cpu() for i in range(len(class_names))}
        text_matrix = text_feats.float().cpu()  # [num_classes, dim]
        return text_features, text_matrix, class_to_idx

    def set_modality(self, modality_name):
        if modality_name is None:
            self.current_modality = None
            return
        if modality_name in self.img_dir_names:
            self.current_modality = modality_name
        else:
            raise ValueError(f"Invalid modality: {modality_name}. Available: {self.img_dir_names}")

    def update_match_label(self, new_match_label):
        if len(new_match_label) != len(self.match_label):
            raise ValueError(f"new_match_label length {len(new_match_label)} != dataset length {len(self.match_label)}")
        self.match_label = np.array(new_match_label, dtype=int)

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        img_path = self.loaded_data[subject]['img'][trial_index]
        match_label = self.match_label[index]
        img = 'None'

        if 'label' in self.loaded_data[subject]:
            label = int(self.loaded_data[subject]['label'][trial_index])
        else:
            label = -1

        # Current sample text
        has_text_sample = self.has_text and (self.text_features is not None) and ('text' in self.loaded_data[subject])
        if has_text_sample:
            text_name = self.loaded_data[subject]['text'][trial_index]
            text_prompt = f"This is a {text_name}."
            text_feat = self.text_features[text_name]
            class_idx = int(self.class_to_idx[text_name]) if self.class_to_idx is not None else -1
        else:
            text_prompt = ""
            text_feat = None
            class_idx = -1

        if self.current_modality is not None:
            if self.config['data']['uncertainty_aware']:
                if self.mode == 'train':
                    tag = 'low' if match_label == 0 else ('high' if match_label == 2 else 'medium')
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
                'subject': subject,
                'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
                'img_features': img_features,
                'text': text_prompt,
                'class_idx': class_idx
            }
            if text_feat is not None:
                sample['text_features'] = text_feat
            return sample

        # Multi-modality branch
        if self.config['data']['uncertainty_aware']:
            if self.mode == 'train':
                tag = 'low' if match_label == 0 else ('high' if match_label == 2 else 'medium')
            else:
                tag = 'medium'
            img_features_dict = {
                f'img_features_{modality}': self.img_features[tag][modality][img_path]
                for modality in self.img_dir_names
            }
        else:
            img_features_dict = {
                f'img_features_{modality}': self.img_features[modality][img_path]
                for modality in self.img_dir_names
            }

        sample = {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'img_path': img_path,
            'img': img,
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
            'text': text_prompt,
            'class_idx': class_idx
        }
        if text_feat is not None:
            sample['text_features'] = text_feat

        sample.update(img_features_dict)
        return sample

    def __len__(self):
        return self.trial_all_subjects