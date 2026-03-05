"""
Baseline Comparison for Tamil Dialect Classification
Compares standard MFCC approaches with proposed nasalization-focused features
Uses the same fixed speaker split for fair comparison
"""

import os
import numpy as np
import librosa
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
RANDOM_SEED = 42
DATA_PATH = './data'
SPLIT_FILE = 'fixed_split.json'
SAMPLE_RATE = 16000
DURATION = 5

np.random.seed(RANDOM_SEED)

# ==========================================
# Load data with existing split
# ==========================================
def load_data_with_split():
    dialect_names = {'Central_Dialect':0, 'Northern_Dialect':1, 
                     'Southern_Dialect':2, 'Western_Dialect':3}
    
    all_samples = []
    train_path = os.path.join(DATA_PATH, 'Train')
    
    for dialect_name, dialect_idx in dialect_names.items():
        dialect_path = os.path.join(train_path, dialect_name)
        if not os.path.exists(dialect_path):
            continue
            
        for item in os.listdir(dialect_path):
            item_path = os.path.join(dialect_path, item)
            if os.path.isdir(item_path) and item.endswith('_audio'):
                speaker = item.replace('_audio', '')
                
                wavs = sorted([os.path.join(item_path, f) for f in os.listdir(item_path) 
                              if f.endswith('.wav')])
                
                for wav in wavs:
                    all_samples.append({
                        'audio_path': wav,
                        'dialect_idx': dialect_idx,
                        'speaker_id': speaker
                    })
    
    with open(SPLIT_FILE, 'r') as f:
        split_info = json.load(f)
    split_info = {int(k): v for k, v in split_info.items()}
    
    train_spk = set()
    val_spk = set()
    for dialect, speakers in split_info.items():
        train_spk.update(speakers['train'])
        val_spk.update(speakers['val'])
    
    train_samples = [s for s in all_samples if s['speaker_id'] in train_spk]
    val_samples = [s for s in all_samples if s['speaker_id'] in val_spk]
    
    return train_samples, val_samples

# ==========================================
# Feature Extractors
# ==========================================
class MFCC13Extractor:
    def extract(self, path):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13,
                                         n_fft=1024, hop_length=256)
            return np.mean(mfccs, axis=1)
        except:
            return None

class MFCC39Extractor:
    def extract(self, path):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13,
                                         n_fft=1024, hop_length=256)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            return np.concatenate([np.mean(mfccs, axis=1),
                                  np.mean(delta, axis=1),
                                  np.mean(delta2, axis=1)])
        except:
            return None

class ProposedExtractor:
    def extract(self, path):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
            features = []
            
            # Nasalization
            D = librosa.stft(audio, n_fft=1024, hop_length=256)
            mag = np.abs(D)
            freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)
            nasal_mask = (freqs >= 200) & (freqs <= 400)
            oral_mask = (freqs >= 400) & (freqs <= 3500)
            
            nasal_energy = np.sum(mag[nasal_mask, :], axis=0)
            oral_energy = np.sum(mag[oral_mask, :], axis=0)
            nasal_per_frame = nasal_energy / (oral_energy + 1e-10)
            
            features.extend([np.mean(nasal_per_frame), np.std(nasal_per_frame),
                           np.max(nasal_per_frame) - np.min(nasal_per_frame)])
            
            # MFCCs (first 5)
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=5)
            features.extend(np.mean(mfccs, axis=1))
            
            return np.array(features)
        except:
            return None

# ==========================================
# Evaluation
# ==========================================
def evaluate(extractor, train_samples, val_samples):
    def extract_features(samples):
        X, y = [], []
        for s in tqdm(samples):
            feats = extractor.extract(s['audio_path'])
            if feats is not None:
                X.append(feats)
                y.append(s['dialect_idx'])
        return np.array(X), np.array(y)
    
    X_train, y_train = extract_features(train_samples)
    X_val, y_val = extract_features(val_samples)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    
    return f1_score(y_val, y_pred, average='macro')

# ==========================================
# Main
# ==========================================
print("\n" + "="*60)
print("BASELINE COMPARISON FOR TAMIL DIALECT CLASSIFICATION")
print("="*60)

train_samples, val_samples = load_data_with_split()
print(f"\n📊 Loaded fixed split:")
print(f"   Train: {len(train_samples)} samples")
print(f"   Val: {len(val_samples)} samples")

extractors = [
    (MFCC13Extractor(), "MFCC-13 (13 features)"),
    (MFCC39Extractor(), "MFCC-39 (13 + deltas)"),
    (ProposedExtractor(), "Proposed (15 features)")
]

results = []
for ext, name in extractors:
    print(f"\n🔍 Evaluating {name}...")
    f1 = evaluate(ext, train_samples, val_samples)
    results.append((name, f1))
    print(f"   Macro F1: {f1:.4f}")

# ==========================================
# Display Comparison Table
# ==========================================
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print()
print("-" * 60)
print(f"{'Method':<35} {'Features':<12} {'Macro F1':<12}")
print("-" * 60)

baseline_f1 = results[0][1]  # MFCC-13 as baseline
for name, f1 in results:
    if "MFCC-13" in name:
        print(f"{name:<35} 13           {f1:.4f}   (baseline)")
    else:
        features = "39" if "MFCC-39" in name else "15"
        diff = ((f1 - baseline_f1) / baseline_f1) * 100
        print(f"{name:<35} {features:<12} {f1:.4f}   ({diff:+.1f}%)")

print("-" * 60)
print("\n✅ Baseline comparison complete!")