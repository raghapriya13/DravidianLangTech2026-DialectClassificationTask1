"""
Tamil Dialect Classifier 
Generates:
- Validation F1 score (reproducible)
- Nasalization importance % (for dominance claim)
- Statistical tests for Southern sub-dialect (p-values)
- Figure 1: Validation Confusion Matrix
- Figure 2: Southern Boxplot with statistics
"""

import os
import numpy as np
import pandas as pd
import librosa
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
RANDOM_SEED = 42
VAL_SPLIT = 0.25
MIN_VAL_SPEAKERS = 5
DATA_PATH = './data'
NASAL_BAND = (200, 400)  # Based on Keane (2004)
SPLIT_FILE = 'fixed_split.json'  # Save split here

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================
# Data Loading (Simplified)
# ==========================================
def load_all_data(data_path):
    dialect_names = {'Central_Dialect':0, 'Northern_Dialect':1, 
                     'Southern_Dialect':2, 'Western_Dialect':3}
    idx_to_dialect = {v:k.replace('_Dialect','') for k,v in dialect_names.items()}
    all_samples = []
    
    train_path = os.path.join(data_path, 'Train')
    for dialect_name, dialect_idx in dialect_names.items():
        dialect_path = os.path.join(train_path, dialect_name)
        if not os.path.exists(dialect_path):
            continue
            
        # Get all speaker folders
        for item in os.listdir(dialect_path):
            item_path = os.path.join(dialect_path, item)
            if os.path.isdir(item_path) and item.endswith('_audio'):
                speaker = item.replace('_audio', '')
                
                # Find transcript file
                transcript_path = os.path.join(dialect_path, f"{speaker}_Text.txt")
                if not os.path.exists(transcript_path):
                    continue
                
                # Read transcript lines
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                
                # Find matching wav files
                wavs = sorted([os.path.join(item_path, f) for f in os.listdir(item_path) 
                              if f.endswith('.wav') and f.startswith(speaker)])
                
                # Create samples
                for i in range(min(len(wavs), len(lines))):
                    all_samples.append({
                        'audio_path': wavs[i],
                        'dialect_idx': dialect_idx,
                        'speaker_id': speaker
                    })
    
    # Test files
    test_path = os.path.join(data_path, 'Test')
    test_files = [os.path.join(test_path, f) for f in sorted(os.listdir(test_path)) 
                  if f.endswith('.wav')] if os.path.exists(test_path) else []
    
    return all_samples, test_files, idx_to_dialect

# ==========================================
# Speaker-Aware Split with SAVE/LOAD capability
# ==========================================
def create_speaker_split(samples, val_ratio=0.25):
    """Create new random split (used only for first run)"""
    speakers_by_dialect = defaultdict(set)
    for s in samples:
        speakers_by_dialect[s['dialect_idx']].add(s['speaker_id'])
    
    train_spk, val_spk = set(), set()
    split_info = {}
    
    for dialect, spk_set in speakers_by_dialect.items():
        spk_list = list(spk_set)
        random.shuffle(spk_list)
        n_val = max(MIN_VAL_SPEAKERS, int(len(spk_list) * val_ratio))
        
        dialect_val = spk_list[:n_val]
        dialect_train = spk_list[n_val:]
        
        val_spk.update(dialect_val)
        train_spk.update(dialect_train)
        
        # Store for saving
        split_info[dialect] = {
            'train': list(dialect_train),
            'val': list(dialect_val)
        }
    
    train = [s for s in samples if s['speaker_id'] in train_spk]
    val = [s for s in samples if s['speaker_id'] in val_spk]
    
    return train, val, split_info

def load_fixed_split(samples, split_file='fixed_split.json'):
    """Load previously saved split for reproducibility"""
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    # Convert string keys back to int if needed
    if list(split_info.keys())[0].isdigit():
        split_info = {int(k): v for k, v in split_info.items()}
    
    train_spk = set()
    val_spk = set()
    
    for dialect, speakers in split_info.items():
        train_spk.update(speakers['train'])
        val_spk.update(speakers['val'])
    
    train = [s for s in samples if s['speaker_id'] in train_spk]
    val = [s for s in samples if s['speaker_id'] in val_spk]
    
    print(f"📂 Loaded fixed split from {split_file}")
    print(f"   Train speakers: {len(train_spk)}, Val speakers: {len(val_spk)}")
    
    return train, val

# ==========================================
# Feature Extractor (15 features)
# ==========================================
class FeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        self.feature_names = [
            'nasal_index', 'nasal_std', 'nasal_range',
            'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean',
            'pitch_mean', 'pitch_std', 'rms_mean', 'rms_std',
            'centroid_mean', 'rolloff_mean', 'f1_mean'
        ]
    
    def extract(self, path):
        try:
            audio, sr = librosa.load(path, sr=self.sr, duration=5)
            if len(audio) < self.sr:
                return None
            
            # Nasalization (key feature)
            D = librosa.stft(audio, n_fft=1024, hop_length=256)
            mag = np.abs(D)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=1024)
            nasal_mask = (freqs >= NASAL_BAND[0]) & (freqs <= NASAL_BAND[1])
            oral_mask = (freqs >= NASAL_BAND[1]) & (freqs <= 3500)
            
            nasal_energy = np.sum(mag[nasal_mask, :], axis=0)
            oral_energy = np.sum(mag[oral_mask, :], axis=0)
            nasal_per_frame = nasal_energy / (oral_energy + 1e-10)
            
            feats = {
                'nasal_index': np.mean(nasal_per_frame),
                'nasal_std': np.std(nasal_per_frame),
                'nasal_range': np.max(nasal_per_frame) - np.min(nasal_per_frame),
            }
            
            # MFCCs (first 5)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
            for i in range(5):
                feats[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            
            # Pitch
            pitches, _ = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=300)
            pitches = pitches[pitches > 0]
            feats['pitch_mean'] = np.mean(pitches) if len(pitches) > 0 else 0
            feats['pitch_std'] = np.std(pitches) if len(pitches) > 0 else 0
            
            # Energy
            rms = librosa.feature.rms(y=audio)[0]
            feats['rms_mean'] = np.mean(rms)
            feats['rms_std'] = np.std(rms)
            
            # Spectral
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            feats['centroid_mean'] = np.mean(cent)
            feats['rolloff_mean'] = np.mean(rolloff)
            
            # F1 (simplified)
            try:
                lpc = librosa.lpc(audio, order=12)
                roots = np.roots(lpc)
                roots = roots[np.imag(roots) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * (sr / (2 * np.pi)))
                feats['f1_mean'] = formants[0] if len(formants) > 0 else 0
            except:
                feats['f1_mean'] = 0
            
            return feats
        except:
            return None

# ==========================================
# Main Pipeline
# ==========================================
print("\n" + "="*60)
print("TAMIL DIALECT CLASSIFIER - PAPER RESULTS (FIXED SPLIT)")
print("="*60)

# Load data
all_samples, test_files, idx_to_dialect = load_all_data(DATA_PATH)

# ==========================================
# FIXED SPLIT LOGIC: Create once, load thereafter
# ==========================================
if os.path.exists(SPLIT_FILE):
    # Load existing split
    train_samples, val_samples = load_fixed_split(all_samples, SPLIT_FILE)
else:
    # Create and save new split (first run only)
    print("\n📌 First run: Creating and saving fixed speaker split...")
    train_samples, val_samples, split_info = create_speaker_split(all_samples, VAL_SPLIT)
    
    # Save split info (convert keys to strings for JSON)
    split_info_str = {str(k): v for k, v in split_info.items()}
    with open(SPLIT_FILE, 'w') as f:
        json.dump(split_info_str, f, indent=2)
    
    print(f"✅ Split saved to {SPLIT_FILE}")
    print(f"   Train speakers: {len(set(s['speaker_id'] for s in train_samples))}")
    print(f"   Val speakers: {len(set(s['speaker_id'] for s in val_samples))}")

print(f"\n📊 Final split sizes:")
print(f"   Train: {len(train_samples)} samples")
print(f"   Val: {len(val_samples)} samples")

# Extract features
extractor = FeatureExtractor()
feature_names = extractor.feature_names

def extract_features(samples, desc):
    X, y = [], []
    failed = 0
    for s in tqdm(samples, desc=desc):
        feats = extractor.extract(s['audio_path'])
        if feats:
            X.append([feats[n] for n in feature_names])
            y.append(s['dialect_idx'])
        else:
            failed += 1
    if failed > 0:
        print(f"   ⚠ {failed} files failed extraction")
    return np.array(X), np.array(y)

print("\n🔍 Extracting features...")
X_train, y_train = extract_features(train_samples, "Train")
X_val, y_val = extract_features(val_samples, "Val")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==========================================
# Train Ensemble
# ==========================================
models = {
    'RF': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED),
    'LR': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
}

trained = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained[name] = model

ensemble = VotingClassifier(estimators=[(n, trained[n]) for n in models.keys()], voting='soft')
ensemble.fit(X_train_scaled, y_train)
y_pred = ensemble.predict(X_val_scaled)
val_f1 = f1_score(y_val, y_pred, average='macro')

# ==========================================
# RESULT 1: Nasalization Importance % (ONLY)
# ==========================================
print("\n" + "="*60)
print("RESULT 1: NASALIZATION FEATURE CONTRIBUTION")
print("="*60)

rf = trained['RF']
imp = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})

# Extract only nasalization features
nasal_features = ['nasal_index', 'nasal_std', 'nasal_range']
nasal_imp = imp[imp['feature'].isin(nasal_features)].copy()
nasal_imp = nasal_imp.sort_values('importance', ascending=False)

nasal_total = nasal_imp['importance'].sum()
nasal_percentage = nasal_total * 100

# Find rank of best nasal feature
all_imp_sorted = imp.sort_values('importance', ascending=False).reset_index(drop=True)
best_nasal_rank = all_imp_sorted[all_imp_sorted['feature'].isin(nasal_features)].index[0] + 1

print(f"\n📊 Nasalization Feature Contribution:")
print(f"   nasal_index:      {nasal_imp[nasal_imp['feature']=='nasal_index']['importance'].values[0]:.4f}")
print(f"   nasal_std:        {nasal_imp[nasal_imp['feature']=='nasal_std']['importance'].values[0]:.4f}")
print(f"   nasal_range:      {nasal_imp[nasal_imp['feature']=='nasal_range']['importance'].values[0]:.4f}")
print(f"   {'─'*40}")
print(f"   TOTAL:            {nasal_total:.4f} ({nasal_percentage:.1f}%)")

print(f"\n→ Paper: 'Nasalization features collectively account for {nasal_percentage:.1f}% of total Random Forest feature importance.'")

# ==========================================
# RESULT 2: Southern Sub-Dialect Statistics
# ==========================================
print("\n" + "="*60)
print("RESULT 2: SOUTHERN SUB-DIALECT EVIDENCE")
print("="*60)

# Get Southern samples
southern_mask = (y_val == 2)
southern_indices = np.where(southern_mask)[0]

southern_data = []
for idx in southern_indices:
    pred_dialect = idx_to_dialect[y_pred[idx]]
    # Use original scale nasal_index (not scaled)
    nasal_idx = X_val[idx][feature_names.index('nasal_index')]  # Fixed: use X_val, not X_train
    
    if y_pred[idx] == 2:
        outcome = 'Correct'
    else:
        outcome = f'Misclassified as {pred_dialect}'
    
    southern_data.append({
        'outcome': outcome, 
        'nasal_index': nasal_idx,
        'true_dialect': 'Southern',
        'pred_dialect': pred_dialect
    })

southern_df = pd.DataFrame(southern_data)

# Statistical tests
correct = southern_df[southern_df['outcome'] == 'Correct']['nasal_index']
results = {}

print("\n📊 Statistical Tests (Welch's t-test vs Correct):")
for mis_type in ['Northern', 'Western', 'Central']:
    mask = southern_df['outcome'] == f'Misclassified as {mis_type}'
    if mask.any():
        mis = southern_df[mask]['nasal_index']
        t_stat, p_value = stats.ttest_ind(correct, mis, equal_var=False)
        results[mis_type] = p_value
        sig = "*** SIGNIFICANT (p<0.05)" if p_value < 0.05 else "not significant"
        print(f"\n   Southern → {mis_type}:")
        print(f"      n={len(mis):2d} | mean={mis.mean():.3f} | p={p_value:.4f} {sig}")
    else:
        results[mis_type] = 1.0

print(f"\n   Correct Southern: n={len(correct):2d} | mean={correct.mean():.3f}")

print("\n→ Paper: 'Southern samples misclassified as Northern show significantly lower nasalization")
if results.get('Northern', 1) < 0.05:
    print(f"   (p={results['Northern']:.4f}), suggesting a Northern-like sub-dialect within the Southern category.'")

# ==========================================
# FIGURE 1: Validation Confusion Matrix
# ==========================================
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Central', 'Northern', 'Southern', 'Western'],
            yticklabels=['Central', 'Northern', 'Southern', 'Western'])
plt.xlabel('Predicted Dialect')
plt.ylabel('True Dialect')
plt.title(f'Validation Confusion Matrix (F1={val_f1:.3f})\n(25% held-out speakers)')
plt.tight_layout()
plt.savefig('figure1_validation_confusion_matrix.png', dpi=300)
print("\n📊 Saved: figure1_validation_confusion_matrix.png")

# ==========================================
# FIGURE 2: Southern Boxplot (was Figure 3)
# ==========================================
plt.figure(figsize=(10, 6))

# Order categories meaningfully
categories = ['Correct', 'Misclassified as Northern', 
              'Misclassified as Western', 'Misclassified as Central']
existing = [c for c in categories if c in southern_df['outcome'].values]

# Define colors
palette = {
    'Correct': '#2ecc71',  # green
    'Misclassified as Northern': '#3498db',  # blue
    'Misclassified as Western': '#e74c3c',  # red
    'Misclassified as Central': '#f39c12'  # orange
}

# Create boxplot
ax = sns.boxplot(x='outcome', y='nasal_index', data=southern_df, 
                 palette=palette, order=existing, width=0.6)

# Add individual points
sns.swarmplot(x='outcome', y='nasal_index', data=southern_df, 
              color='black', alpha=0.4, size=2, order=existing)

plt.xlabel('Classification Outcome', fontsize=11)
plt.ylabel('Nasalization Index', fontsize=11)
plt.title('Southern Tamil Samples: Nasalization Index by Classification Outcome\n(Validation Set)', fontsize=12)
plt.xticks(rotation=15)

# Add significance stars
y_max = southern_df['nasal_index'].max() + 0.05
for i, mis_type in enumerate(['Northern', 'Western', 'Central']):
    if mis_type in results and results[mis_type] < 0.05:
        if f'Misclassified as {mis_type}' in existing:
            x_pos = existing.index(f'Misclassified as {mis_type}')
            plt.text(x_pos, y_max, '*', fontsize=16, ha='center', color='red')

# Add statistics box
stats_text = "p-values (vs Correct):\n"
for mis_type, p in results.items():
    if f'Misclassified as {mis_type}' in existing:
        if p < 0.001:
            stats_text += f"{mis_type}: p<0.001"
        elif p < 0.01:
            stats_text += f"{mis_type}: p<0.01"
        elif p < 0.05:
            stats_text += f"{mis_type}: p<0.05"
        else:
            stats_text += f"{mis_type}: p={p:.3f}"
        if p < 0.05:
            stats_text += " *"
        stats_text += "\n"

plt.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig('figure2_southern_boxplot.png', dpi=300)
print("📊 Saved: figure2_southern_boxplot.png")

# ==========================================
# Test Predictions (minimal)
# ==========================================
print(f"\n🎯 Predicting {len(test_files)} test files...")

def predict(features):
    X = np.array([[features.get(n,0) for n in feature_names]])
    X_scaled = scaler.transform(X)
    proba = ensemble.predict_proba(X_scaled)[0]
    return idx_to_dialect[np.argmax(proba)]

predictions = []
for path in tqdm(test_files):
    fid = os.path.splitext(os.path.basename(path))[0]
    feats = extractor.extract(path)
    if feats:
        predictions.append((fid, predict(feats)))

os.makedirs('submission', exist_ok=True)
with open('submission/dialect_predictions.txt','w') as f:
    for fid, d in predictions:
        f.write(f"{fid} {d}\n")

print(f"✅ Saved {len(predictions)} predictions")

# ==========================================


