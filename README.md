# DravidianLangTech2026-DialectClassificationTask1
This repository contains the submission for the DravidianLangTech@ACL 2026 Task 1: Dialect-based speech recognition and classification in Tamil. 
We present a nasalization-focused approach using only 15 acoustic features to classify four Tamil dialects (Northern, Southern, Central, Western).

Approach
Feature Set: 15 acoustic features (3 nasalization, 5 MFCCs, 7 prosodic)
Speaker-Aware Split: 75/25 train/validation with no speaker overlap (fixed split for reproducibility)
Ensemble Model: Soft voting of Random Forest, SVM (RBF), and Logistic Regression
Key Finding: Southern samples misclassified as Northern show significantly lower nasalization (p=0.0247), suggesting sub-dialectal variation
