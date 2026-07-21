# config_manager.py
"""
Manage all model path configurations
"""
import os
from pathlib import Path

class ModelPaths:
    """Model path manager"""
    
    def __init__(self, base_path="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/"):
        self.base_path = Path(base_path)
        
        # SDXL-Turbo model path
        self.sdxl_turbo_path = self.base_path / "ModelWe/models--stabilityai--sdxl-turbo/"
        
        # IP-Adapter model path
        self.ip_adapter_path = self.base_path / "ModelWeights/IP-Adapter/"
        
        # Check if model paths exist
        self._check_paths()
    
    def _check_paths(self):
        """Check if model paths exist"""
        print("Checking model paths...")
        
        if not self.sdxl_turbo_path.exists():
            print(f"Warning: SDXL-Turbo path does not exist: {self.sdxl_turbo_path}")
        else:
            print(f"SDXL-Turbo path: {self.sdxl_turbo_path}")
            
        if not self.ip_adapter_path.exists():
            print(f"Warning: IP-Adapter path does not exist: {self.ip_adapter_path}")
        else:
            print(f"IP-Adapter path: {self.ip_adapter_path}")
    
    def get_sdxl_path(self):
        """Get SDXL path"""
        # Check path structure to determine the final model path
        if (self.sdxl_turbo_path / "snapshots").exists():
            # Find the first folder inside the snapshots directory
            snapshots = list((self.sdxl_turbo_path / "snapshots").iterdir())
            if snapshots and snapshots[0].is_dir():
                return str(snapshots[0])
        return str(self.sdxl_turbo_path)
    
    def get_ip_adapter_sdxl_weights(self):
        """Get IP-Adapter SDXL weights path"""
        sdxl_models_path = self.ip_adapter_path / "sdxl_models"
        
        # Check for weight files
        possible_weights = [
            "ip-adapter_sdxl_vit-h.safetensors",
            "ip-adapter_sdxl_vit-h.bin",
            "ip-adapter_sdxl.safetensors",
            "ip-adapter_sdxl.bin"
        ]
        
        for weight in possible_weights:
            weight_path = sdxl_models_path / weight
            if weight_path.exists():
                return str(weight_path)
        
        # If not found, return the directory path
        return str(sdxl_models_path)
    
    def get_ip_adapter_image_encoder_path(self):
        """Get IP-Adapter image encoder path"""
        image_encoder_path = self.ip_adapter_path / "models" / "image_encoder"
        
        if image_encoder_path.exists():
            return str(image_encoder_path)
        
        # Try other possible paths
        image_encoder_path = self.ip_adapter_path / "image_encoder"
        if image_encoder_path.exists():
            return str(image_encoder_path)
        
        return None

# Create global configuration
MODEL_PATHS = ModelPaths("/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/")