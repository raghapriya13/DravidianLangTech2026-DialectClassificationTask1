# DravidianLangTech2026-DialectClassificationTask1
This repository contains the submission for the DravidianLangTech@ACL 2026 Task 1: Dialect-based speech recognition and classification in Tamil. 
We present a nasalization-focused approach using only 15 acoustic features to classify four Tamil dialects (Northern, Southern, Central, Western).

Project Overview
This work demonstrates that a linguistically-motivated reduced feature set can achieve competitive performance while maintaining interpretability. Our approach is grounded in phonological evidence that nasal realization varies across Tamil dialects, with Southern Tamil exhibiting stronger vowel nasalization compared to Northern varieties.

Key Contributions:
1) A compact 15-feature set with nasalization at its core
2) Evidence for sub-dialectal variation within Southern Tamil
3) Reproducible speaker-aware data partitioning
4) Comparison against standard MFCC baselines

Commited Files:
1) tamil_dialect_nasalization.py
The main experiment script that loads the dataset, creates a fixed speaker-aware split, and extracts 15 acoustic features (3 nasalization, 5 MFCCs, 7 prosodic). It trains a soft-voting ensemble (Random Forest, SVM-RBF, Logistic Regression), computes feature importance to quantify nasalization's contribution, performs statistical analysis on Southern samples to detect sub-dialect patterns, generates visualizations (confusion matrix, boxplot), and produces test set predictions for the shared task.

2) mfcc_baseline_comparison.py
Evaluates standard MFCC approaches against our proposed method using the same fixed speaker split for fair comparison. It implements two baselines—MFCC-13 (13 static coefficients) and MFCC-39 (13 MFCC + 13 deltas + 13 delta-deltas)—and compares their performance against our 15-feature set using Random Forest. The script demonstrates that our approach achieves comparable accuracy with 60% fewer features, providing quantitative evidence for the efficiency of linguistically-motivated feature engineering.

fixed_split.json - This file stores the fixed speaker assignment for train/validation splits, ensuring perfect reproducibility of all experiments and will be Auto-generated on running tamil_dialect_nasalization.py 
