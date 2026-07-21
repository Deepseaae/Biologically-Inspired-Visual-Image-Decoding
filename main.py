import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime

## import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config, ClipLoss, instantiate_from_config, get_device

device = get_device('auto')

# Define all color space modalities
ALL_MODALITIES = ['LMS', 'DKL', 'LUV', 'RGB']  # all modalities
THREE_MODALITIES = ['LMS', 'DKL', 'LUV']  # union of three modalities
# Define two-modality combinations
TWO_MODALITY_COMBINATIONS = {
    'LMS_DKL': ['LMS', 'DKL'],
    'LMS_LUV': ['LMS', 'LUV'],
    'DKL_LUV': ['DKL', 'LUV']
}


def load_model(config, modality_name):
    """Load model for a specific modality"""
    model = {}
    for k, v in config['models'].items():
        print(f"init {k} for modality {modality_name}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, modality_name)
    return pl_model


class PLModel(pl.LightningModule):
    def __init__(self, model, config, modality_name):
        super().__init__()

        self.config = config
        self.modality_name = modality_name  # current modality being processed

        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        # Store predictions per modality independently
        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']

        # Uncertainty-aware related variables per modality
        self.sim = None
        self.match_label = None
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0
        self.match_similarities = []

        # Feature storage (train and test sets)
        self.train_features = {
            'eeg_features': [],
            'img_features': [],
            'indices': []
        }
        
        self.test_features = {
            'eeg_features': [],
            'img_features': [],
            'indices': []
        }

    def setup(self, stage):
        """Initialize modality-specific variables before training starts"""
        if stage == 'fit' and self.sim is None:
            # Get training set size to initialize sim and match_label
            train_loader = self.trainer.train_dataloader
            if train_loader is not None:
                dataset_size = len(train_loader.dataset)
                self.sim = np.ones(dataset_size)
                self.match_label = np.ones(dataset_size, dtype=int)

    def forward(self, batch, sample_posterior=False):
        # Use image features for the current modality
        img_features_key = f'img_features_{self.modality_name}'
        img_z = batch[img_features_key]

        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']

        eeg_z = self.brain(eeg)
        img_z = img_z / img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        # Uncertainty-aware logic (unchanged)
        if self.config['data']['uncertainty_aware'] and self.sim is not None:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label

            loss = total_loss
        else:
            loss = total_loss

        return eeg_z, img_z, loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch, sample_posterior=True)

        # Add modality info to metric names
        self.log(f'train_loss_{self.modality_name}', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

            self.log(f'train_top1_acc_{self.modality_name}', top_1_accuracy,
                     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'train_top5_acc_{self.modality_name}', top_k_accuracy,
                     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            if self.match_label is not None:
                counter = Counter(self.match_label)
                count_dict = dict(counter)
                key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
                count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
                # Log uncertainty distribution per modality
                mapped_dict_with_modality = {f'{k}_{self.modality_name}': v for k, v in count_dict_mapped.items()}
                self.log_dict(mapped_dict_with_modality, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)

        self.log(f'val_loss_{self.modality_name}', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.log(f'val_top1_acc_{self.modality_name}', top_1_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'val_top5_acc_{self.modality_name}', top_k_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)

        self.log(f'test_loss_{self.modality_name}', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        # compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        # Log modality-specific test metrics
        self.log(f'test_top1_acc_{self.modality_name}', top_1_accuracy, sync_dist=True)
        self.log(f'test_top5_acc_{self.modality_name}', top_k_accuracy, sync_dist=True)
        self.log(f'mAP_{self.modality_name}', self.mAP, sync_dist=True)
        self.log(f'similarity_{self.modality_name}', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        return {
            f'test_loss_{self.modality_name}': self.trainer.callback_metrics[f'test_loss_{self.modality_name}'].item(),
            f'test_top1_acc_{self.modality_name}': top_1_accuracy.item(),
            f'test_top5_acc_{self.modality_name}': top_k_accuracy.item(),
            f'mAP_{self.modality_name}': self.mAP,
            f'similarity_{self.modality_name}': self.match_similarities
        }
    
    def extract_and_save_features(self, train_loader, test_loader, save_dir):
        """Extract and save EEG and image features for train and test sets"""
        print(f"\nStarting feature extraction for modality {self.modality_name}...")
        
        # Set model to evaluation mode
        self.eval()
        
        # Extract training set features
        print(f"Extracting training set features...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                idx = batch['idx']
                eeg = batch['eeg']
                img_features_key = f'img_features_{self.modality_name}'
                img_z = batch[img_features_key]
                
                # Process EEG features
                eeg_z = self.brain(eeg)
                eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
                
                # Process image features
                img_z = img_z / img_z.norm(dim=-1, keepdim=True)
                
                # Save features
                self.train_features['eeg_features'].append(eeg_z.cpu())
                self.train_features['img_features'].append(img_z.cpu())
                self.train_features['indices'].append(idx.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"  Processed training batch {batch_idx}")
        
        # Concatenate training features
        if len(self.train_features['eeg_features']) > 0:
            self.train_features['eeg_features'] = torch.cat(self.train_features['eeg_features'], dim=0)
            self.train_features['img_features'] = torch.cat(self.train_features['img_features'], dim=0)
            self.train_features['indices'] = torch.cat(self.train_features['indices'], dim=0)
            
            print(f"Training set feature extraction completed: EEG ({self.train_features['eeg_features'].shape}), "
                  f"Image ({self.train_features['img_features'].shape})")
        
        # Extract test set features
        print(f"Extracting test set features...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                idx = batch['idx']
                eeg = batch['eeg']
                img_features_key = f'img_features_{self.modality_name}'
                img_z = batch[img_features_key]
                
                # Process EEG features
                eeg_z = self.brain(eeg)
                eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
                
                # Process image features
                img_z = img_z / img_z.norm(dim=-1, keepdim=True)
                
                # Save features
                self.test_features['eeg_features'].append(eeg_z.cpu())
                self.test_features['img_features'].append(img_z.cpu())
                self.test_features['indices'].append(idx.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"  Processed test batch {batch_idx}")
        
        # Concatenate test features
        if len(self.test_features['eeg_features']) > 0:
            self.test_features['eeg_features'] = torch.cat(self.test_features['eeg_features'], dim=0)
            self.test_features['img_features'] = torch.cat(self.test_features['img_features'], dim=0)
            self.test_features['indices'] = torch.cat(self.test_features['indices'], dim=0)
            
            print(f"Test set feature extraction completed: EEG ({self.test_features['eeg_features'].shape}), "
                  f"Image ({self.test_features['img_features'].shape})")
        
        # Save features to files
        modality_dir = os.path.join(save_dir, f'modality_{self.modality_name}')
        os.makedirs(modality_dir, exist_ok=True)
        
        # Save training set features - 4 files
        torch.save(self.train_features['eeg_features'], 
                  os.path.join(modality_dir, f'train_eeg_features_{self.modality_name}.pt'))
        torch.save(self.train_features['img_features'], 
                  os.path.join(modality_dir, f'train_img_features_{self.modality_name}.pt'))
        
        # Save test set features - 4 files
        torch.save(self.test_features['eeg_features'], 
                  os.path.join(modality_dir, f'test_eeg_features_{self.modality_name}.pt'))
        torch.save(self.test_features['img_features'], 
                  os.path.join(modality_dir, f'test_img_features_{self.modality_name}.pt'))
        
        # Save metadata
        metadata = {
            'modality': self.modality_name,
            'train_samples': len(self.train_features['eeg_features']),
            'test_samples': len(self.test_features['eeg_features']),
            'train_eeg_shape': list(self.train_features['eeg_features'].shape),
            'train_img_shape': list(self.train_features['img_features'].shape),
            'test_eeg_shape': list(self.test_features['eeg_features'].shape),
            'test_img_shape': list(self.test_features['img_features'].shape),
            'z_dim': self.z_dim,
            'extraction_time': str(datetime.now()),
            'files': [
                f'train_eeg_features_{self.modality_name}.pt',
                f'train_img_features_{self.modality_name}.pt',
                f'test_eeg_features_{self.modality_name}.pt',
                f'test_img_features_{self.modality_name}.pt'
            ]
        }
        
        with open(os.path.join(modality_dir, f'features_metadata_{self.modality_name}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n{self.modality_name} modality features (4 files) saved to: {modality_dir}")
        print(f"  1. train_eeg_features_{self.modality_name}.pt")
        print(f"  2. train_img_features_{self.modality_name}.pt")
        print(f"  3. test_eeg_features_{self.modality_name}.pt")
        print(f"  4. test_img_features_{self.modality_name}.pt")
        
        return modality_dir
    
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](
            self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        return [optimizer]


class EnsembleTester:
    """Ensemble tester for computing union metrics across color space modalities"""

    def __init__(self, modalities):
        self.modalities = modalities
        self.all_predictions = {modality: [] for modality in modalities}
        self.all_true_labels = []

    def collect_predictions(self, batch, models):
        """Collect predictions from all models for the current batch"""
        batch_size = batch['idx'].shape[0]

        # Store true labels for the current batch
        true_labels = torch.arange(0, batch_size).cpu().numpy()
        self.all_true_labels.extend(true_labels)

        # Run inference for each modality model and collect predictions
        for modality in self.modalities:
            model = models[modality]
            eeg = batch['eeg']
            img_features = batch[f'img_features_{modality}']

            with torch.no_grad():
                eeg_z = model.brain(eeg)
                eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
                img_z = img_features / img_features.norm(dim=-1, keepdim=True)

                similarity = (eeg_z @ img_z.T)
                top_kvalues, top_k_indices = similarity.topk(5, dim=-1)

                self.all_predictions[modality].append(top_k_indices.cpu().numpy())

    def compute_union_metrics(self, union_modalities=None):
        """Compute union metrics, can specify a list of modalities"""
        # If no modalities specified, use all modalities
        if union_modalities is None:
            union_modalities = self.modalities
            
        # Concatenate all batch predictions
        all_true_labels = np.array(self.all_true_labels)
        all_predictions = {}

        for modality in union_modalities:
            all_predictions[modality] = np.concatenate(self.all_predictions[modality], axis=0)

        # Compute individual accuracy for each modality
        individual_results = {}
        for modality in union_modalities:
            predictions = all_predictions[modality]
            top1_correct = predictions[:, 0] == all_true_labels
            top5_correct = (predictions == all_true_labels[:, np.newaxis]).any(axis=1)

            individual_results[modality] = {
                'top1_acc': top1_correct.mean(),
                'top5_acc': top5_correct.mean()
            }

        # Compute union accuracy
        union_top1_correct = np.zeros(len(all_true_labels), dtype=bool)
        union_top5_correct = np.zeros(len(all_true_labels), dtype=bool)

        for modality in union_modalities:
            predictions = all_predictions[modality]
            union_top1_correct |= (predictions[:, 0] == all_true_labels)
            union_top5_correct |= (predictions == all_true_labels[:, np.newaxis]).any(axis=1)

        union_results = {
            'union_top1_acc': union_top1_correct.mean(),
            'union_top5_acc': union_top5_correct.mean(),
            'union_modalities': union_modalities
        }

        return individual_results, union_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    # Create main save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(
        config)
    print(
        f"train num: {len(train_loader.dataset)}, val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")

    # Store all models and results
    all_models = {}
    all_test_results = {}

    # Create main feature save directory
    features_main_dir = os.path.join(config['save_dir'], 'all_modality_features')
    os.makedirs(features_main_dir, exist_ok=True)
    
    print(f"All modality features will be saved to: {features_main_dir}")
    print(f"Each modality will save 4 files (train EEG, train image, test EEG, test image), "
          f"total of {len(ALL_MODALITIES) * 4} files")

    # Train independent model for each color space modality
    for modality in ALL_MODALITIES:
        print(f"\n{'='*60}")
        print(f"=== Training model for color space modality: {modality} ===")
        print(f"{'='*60}")

        # Create separate save directory and logger for each modality
        modality_save_dir = os.path.join(config['save_dir'], f"modality_{modality}")
        os.makedirs(modality_save_dir, exist_ok=True)

        logger = TensorBoardLogger(
            modality_save_dir,
            name=config['name'],
            version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}"
        )
        os.makedirs(logger.log_dir, exist_ok=True)
        shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

        # Load current modality model
        pl_model = load_model(config, modality)

        # Set callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=logger.log_dir,
            filename=f'model_{modality}' + '_{epoch}_{val_top1_acc_' + modality + ':.4f}',
            save_last=True,
            monitor=f'val_top1_acc_{modality}' if config[
                                                      'exp_setting'] == 'inter-subject' else f'train_loss_{modality}',
            mode='max' if config['exp_setting'] == 'inter-subject' else 'min'
        )

        if config['exp_setting'] == 'inter-subject':
            early_stop_callback = EarlyStopping(
                monitor=f'val_top1_acc_{modality}',
                min_delta=0.001,
                patience=5,
                verbose=False,
                mode='max'
            )
        else:
            early_stop_callback = EarlyStopping(
                monitor=f'train_loss_{modality}',
                min_delta=0.001,
                patience=5,
                verbose=False,
                mode='min'
            )

        # Create trainer
        trainer = Trainer(
            log_every_n_steps=10,
            strategy=DDPStrategy(find_unused_parameters=True),
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config['train']['epoch'],
            devices=[device],
            accelerator='cuda',
            logger=logger
        )

        print(f"Training model for {modality}, log dir: {trainer.logger.log_dir}")

        # Train model
        ckpt_path = 'last'
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

        # Test model
        if config['exp_setting'] == 'inter-subject':
            test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
        else:
            test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

        # Save test results
        all_test_results[modality] = test_results[0] if test_results else {}

        # Save model reference for ensemble testing later
        all_models[modality] = pl_model

        # Save current modality test results
        with open(os.path.join(logger.log_dir, f'test_results_{modality}.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Extract and save current modality features (EEG and image features for train and test sets)
        print(f"\nStarting feature extraction and saving for {modality} modality...")
        pl_model.extract_and_save_features(train_loader, test_loader, features_main_dir)

    # Perform ensemble testing to compute union metrics
    print(f"\n{'='*60}")
    print("=== Performing Ensemble Testing for Color Space Modalities ===")
    print(f"{'='*60}")
    
    # Create ensemble tester with all modalities
    ensemble_tester = EnsembleTester(ALL_MODALITIES)
    
    # Collect predictions from all models
    for batch_idx, batch in enumerate(test_loader):
        ensemble_tester.collect_predictions(batch, all_models)
    
    # Store results for all combinations
    all_union_results = {}
    
    # 1. Compute individual modality results
    print(f"\n{'='*60}")
    print("=== Individual Modality Results ===")
    print(f"{'='*60}")
    for modality in ALL_MODALITIES:
        individual_results_single, _ = ensemble_tester.compute_union_metrics([modality])
        all_union_results[f'single_{modality}'] = {
            'individual_results': individual_results_single,
            'union_results': {
                'union_top1_acc': individual_results_single[modality]['top1_acc'],
                'union_top5_acc': individual_results_single[modality]['top5_acc'],
                'union_modalities': [modality]
            }
        }
    
    # 2. Compute two-modality combination results
    print(f"\n{'='*60}")
    print("=== Two-Modality Combination Results ===")
    print(f"{'='*60}")
    for combo_name, modalities in TWO_MODALITY_COMBINATIONS.items():
        print(f"\nComputing combination {combo_name} ({modalities[0]}+{modalities[1]})")
        individual_results_two, union_results_two = ensemble_tester.compute_union_metrics(modalities)
        all_union_results[f'two_modality_{combo_name}'] = {
            'individual_results': individual_results_two,
            'union_results': union_results_two
        }
    
    # 3. Compute union for three modalities (LMS, DKL, LUV)
    print(f"\n{'='*60}")
    print("=== Union of Three Modalities (LMS, DKL, LUV) ===")
    print(f"{'='*60}")
    individual_results_three, union_results_three = ensemble_tester.compute_union_metrics(THREE_MODALITIES)
    all_union_results['three_modality_union'] = {
        'individual_results': individual_results_three,
        'union_results': union_results_three
    }
    
    # 4. Compute union for four modalities (LMS, DKL, LUV, RGB)
    print(f"\n{'='*60}")
    print("=== Union of Four Modalities (LMS, DKL, LUV, RGB) ===")
    print(f"{'='*60}")
    individual_results_four, union_results_four = ensemble_tester.compute_union_metrics(ALL_MODALITIES)
    all_union_results['four_modality_union'] = {
        'individual_results': individual_results_four,
        'union_results': union_results_four
    }
    
    # Save ensemble test results
    ensemble_results = {
        'all_union_results': all_union_results,
        'detailed_test_results': all_test_results,
        'feature_files_summary': {
            'total_files': len(ALL_MODALITIES) * 4,
            'files_per_modality': 4,
            'two_modality_combinations': TWO_MODALITY_COMBINATIONS,
            'three_modalities': THREE_MODALITIES,
            'four_modalities': ALL_MODALITIES,
            'feature_dir': features_main_dir,
            'files': []
        }
    }
    
    # Add file list information
    for modality in ALL_MODALITIES:
        modality_files = [
            f'train_eeg_features_{modality}.pt',
            f'train_img_features_{modality}.pt',
            f'test_eeg_features_{modality}.pt',
            f'test_img_features_{modality}.pt'
        ]
        ensemble_results['feature_files_summary']['files'].extend(modality_files)
    
    ensemble_save_path = os.path.join(config['save_dir'], 'ensemble_test_results.json')
    with open(ensemble_save_path, 'w') as f:
        json.dump(ensemble_results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("=== Color Space Ensemble Test Results ===")
    print(f"{'='*60}")
    
    # 1. Individual modality results
    print(f"\n1. Individual Modality Results:")
    for modality in ALL_MODALITIES:
        results = all_union_results[f'single_{modality}']['union_results']
        print(f"   {modality}: Top-1 Acc: {results['union_top1_acc']:.4f}, Top-5 Acc: {results['union_top5_acc']:.4f}")
    
    # 2. Two-modality combination results
    print(f"\n2. Two-Modality Combination Results:")
    for combo_name, modalities in TWO_MODALITY_COMBINATIONS.items():
        results = all_union_results[f'two_modality_{combo_name}']['union_results']
        print(f"   {combo_name} ({modalities[0]}+{modalities[1]}): Top-1 Acc: {results['union_top1_acc']:.4f}, Top-5 Acc: {results['union_top5_acc']:.4f}")
        
        # Print individual results for each modality in the combination
        individual_results = all_union_results[f'two_modality_{combo_name}']['individual_results']
        for modality in modalities:
            print(f"     - {modality}: Top-1 Acc: {individual_results[modality]['top1_acc']:.4f}, "
                  f"Top-5 Acc: {individual_results[modality]['top5_acc']:.4f}")
    
    # 3. Three-modality union
    print(f"\n3. Three-Modality Union (LMS, DKL, LUV):")
    results_three = all_union_results['three_modality_union']['union_results']
    print(f"   Union Top-1 Acc: {results_three['union_top1_acc']:.4f}")
    print(f"   Union Top-5 Acc: {results_three['union_top5_acc']:.4f}")
    individual_results_three = all_union_results['three_modality_union']['individual_results']
    for modality, results in individual_results_three.items():
        print(f"     {modality}: Top-1 Acc: {results['top1_acc']:.4f}, Top-5 Acc: {results['top5_acc']:.4f}")
    
    # 4. Four-modality union
    print(f"\n4. Four-Modality Union (LMS, DKL, LUV, RGB):")
    results_four = all_union_results['four_modality_union']['union_results']
    print(f"   Union Top-1 Acc: {results_four['union_top1_acc']:.4f}")
    print(f"   Union Top-5 Acc: {results_four['union_top5_acc']:.4f}")
    individual_results_four = all_union_results['four_modality_union']['individual_results']
    for modality, results in individual_results_four.items():
        print(f"     {modality}: Top-1 Acc: {results['top1_acc']:.4f}, Top-5 Acc: {results['top5_acc']:.4f}")
    
    print(f"\nFeature Files Summary:")
    print(f"  Total files: {len(ALL_MODALITIES) * 4}")
    print(f"  Each modality saves 4 files: train_eeg, train_img, test_eeg, test_img")
    for modality in ALL_MODALITIES:
        print(f"  {modality}:")
        print(f"    - train_eeg_features_{modality}.pt")
        print(f"    - train_img_features_{modality}.pt")
        print(f"    - test_eeg_features_{modality}.pt")
        print(f"    - test_img_features_{modality}.pt")
    
    print(f"\nAll feature files saved to: {features_main_dir}")
    print(f"Detailed results saved to: {ensemble_save_path}")
    
    print(f"\n{'='*60}")
    print("=== Training and Feature Extraction Completed! ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()