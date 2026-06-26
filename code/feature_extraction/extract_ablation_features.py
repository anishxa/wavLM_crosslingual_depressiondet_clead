import os
import re
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import WavLMModel
from tqdm import tqdm
import argparse

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="WavLM Feature Extraction Ablation")
parser.add_argument("--model", type=str, choices=["base-plus", "large"], default="base-plus",
                    help="WavLM model variant (base-plus or large)")
parser.add_argument("--device", type=str, default="auto",
                    help="Device to run inference on (cpu, cuda, mps, auto)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for extraction")
args = parser.parse_args()

# === Select device ===
if args.device == "auto":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print(f"Using device: {device} for feature extraction")

# === Model configuration ===
if args.model == "large":
    model_name = "microsoft/wavlm-large"
    layers = [12, 14, 16, 18]
    model_dir = "wavlm_large"
    hidden_dim = 1024
else:
    model_name = "microsoft/wavlm-base-plus"
    layers = [6, 7, 8, 9]
    model_dir = "wavlm_base_plus"
    hidden_dim = 768

# Load WavLM Model
print(f"Loading {model_name} model...")
model = WavLMModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.file_paths = df["file_path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            # Downmix stereo to mono
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if not 16000
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze(0)
            
            # Standardize length to exactly 160000 samples (10.0 seconds at 16kHz)
            if waveform.shape[0] < 160000:
                pad_len = 160000 - waveform.shape[0]
                waveform = torch.cat([waveform, torch.zeros(pad_len)], dim=0)
            elif waveform.shape[0] > 160000:
                waveform = waveform[:160000]
                
            return waveform, label, path
        except Exception as e:
            # Return silence on error so we don't crash
            return torch.zeros(160000), label, path

def extract_for_dataset(metadata_csv, dataset_name, batch_size=32):
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found: {metadata_csv}")
        return
        
    df = pd.read_csv(metadata_csv)
    print(f"\n==========================================")
    print(f"Processing Dataset: {dataset_name} ({len(df)} rows)")
    
    # Initialize output structures for layers
    for layer in layers:
        os.makedirs(f"features/{model_dir}/features_{dataset_name}_layer{layer}", exist_ok=True)
        
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if len(df_split) == 0:
            continue
            
        print(f"Extracting {split} split ({len(df_split)} items)...")
        dataset = SpeechDataset(df_split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Prepare list to accumulate features for each layer and pooling method
        X_accum_mean = {layer: [] for layer in layers}
        X_accum_max = {layer: [] for layer in layers}
        X_accum_concat = {layer: [] for layer in layers}
        y_accum = []
        
        with torch.no_grad():
            for waveforms, labels, paths in tqdm(dataloader, desc=f"{dataset_name} - {split}"):
                waveforms = waveforms.to(device)
                
                # Perform standard batch normalization on GPU
                mean = waveforms.mean(dim=-1, keepdim=True)
                var = waveforms.var(dim=-1, keepdim=True)
                waveforms = (waveforms - mean) / torch.sqrt(var + 1e-7)
                
                # Forward pass
                out = model(input_values=waveforms)
                
                # Extract pooled features for each layer
                for layer in layers:
                    # hidden_states shape: (batch_size, seq_len, hidden_dim)
                    layer_feat = out.hidden_states[layer]
                    
                    mean_pooled = layer_feat.mean(dim=1).cpu().numpy()  # shape (batch_size, hidden_dim)
                    max_pooled = layer_feat.max(dim=1).values.cpu().numpy()  # shape (batch_size, hidden_dim)
                    concat_pooled = np.concatenate([mean_pooled, max_pooled], axis=1)  # shape (batch_size, hidden_dim * 2)
                    
                    X_accum_mean[layer].append(mean_pooled)
                    X_accum_max[layer].append(max_pooled)
                    X_accum_concat[layer].append(concat_pooled)
                    
                y_accum.append(labels.numpy())
                
        # Stack and save features for each layer
        y_stacked = np.concatenate(y_accum, axis=0)
        for layer in layers:
            X_stacked_mean = np.concatenate(X_accum_mean[layer], axis=0)
            X_stacked_max = np.concatenate(X_accum_max[layer], axis=0)
            X_stacked_concat = np.concatenate(X_accum_concat[layer], axis=0)
            
            # Paths
            y_path = f"features/{model_dir}/features_{dataset_name}_layer{layer}/y_{split}.npy"
            
            # Save all poolings
            np.save(f"features/{model_dir}/features_{dataset_name}_layer{layer}/X_{split}.npy", X_stacked_mean) # compatibility
            np.save(f"features/{model_dir}/features_{dataset_name}_layer{layer}/X_{split}_mean.npy", X_stacked_mean)
            np.save(f"features/{model_dir}/features_{dataset_name}_layer{layer}/X_{split}_max.npy", X_stacked_max)
            np.save(f"features/{model_dir}/features_{dataset_name}_layer{layer}/X_{split}_concat.npy", X_stacked_concat)
            np.save(y_path, y_stacked)
            
            print(f"  Layer {layer} saved: Mean shape {X_stacked_mean.shape}, Max shape {X_stacked_max.shape}, Concat shape {X_stacked_concat.shape}")

if __name__ == "__main__":
    # Run extraction on all three datasets
    extract_for_dataset("data/utterance_table_modma_segmented_split.csv", "modma", batch_size=args.batch_size)
    extract_for_dataset("data/utterance_table_mix_segmented_split.csv", "mix", batch_size=args.batch_size)
    extract_for_dataset("data/utterance_table_edaic_segmented_split.csv", "edaic", batch_size=args.batch_size)

    print("\nAll feature extractions completed successfully!")
