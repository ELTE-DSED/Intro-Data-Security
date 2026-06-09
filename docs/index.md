---
hide:
  - navigation
---

# Introduction to Data Security Practicum

This course provides a comprehensive introduction to the security and privacy of machine learning systems. You will learn to *attack*, *defend*, and *audit* AI models through practical labs organized into 8 thematic modules.

## Instructors & Staff

- **Instructor**: Prof. Lendák Imre 
- **Teaching Assistant**: Ahmed F. Lagha

## Lab Curriculum

| Module | Lab | Topic | Notebook |
|--------|-----|-------|----------|
| **1. Foundations** | 1 | DNN Training & Robust Model Baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_01_foundations/Lab1_DNN_Training_and_Robust_Models.ipynb) |
| **2. Input Manipulation** | 2 | Evasion Attacks (FGSM, PGD) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_02_input_manipulation/Lab2_Evasion_Attacks.ipynb) |
| **3. Data Poisoning** | 3a | Label Flipping Attacks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_03_data_poisoning/Lab_3a_Data_Poisoning_Label_Flipping.ipynb) |
| | 3b | Backdoor & Trigger Injection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_03_data_poisoning/Lab_3b_Data_Poisoning_Backdoor_Attacks.ipynb) |
| **4. Model Poisoning** | 4a | Model Trojans & Supply Chain Attacks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_04_model_poisoning/Lab_4a_Model_Trojans_and_Supply_Chain_Attacks.ipynb) |
| | 4b | Trojan Detection & Certified Defenses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_04_model_poisoning/Lab_4b_Trojan_Detection_and_Certified_Defenses.ipynb) |
| **5. Availability** | 5a | Sponge Attacks & Resource Exhaustion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_05_sponge_attacks/Lab_5a_Sponge_Attacks_and_Resource_Exhaustion.ipynb) |
| | 5b | Sponge Attack Defenses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_05_sponge_attacks/Lab_5b_Sponge_Attack_Defenses_and_Resource_Constraints.ipynb) |
| **6. Confidentiality** | 6a | Membership Inference Attacks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_06_confidentiality_attacks/Lab_6a_Membership_Inference_Attacks.ipynb) |
| | 6b | Model Inversion & Feature Reconstruction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_06_confidentiality_attacks/Lab_6b_Model_Inversion_Attacks_and_Defenses.ipynb) |
| **7. Synthetic Data** | 7 | Tabular Synthetic Data (VAE, GAN) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_07_synthetic_data_generation/Lab_7_Tabular_Synthetic_Data_Generation.ipynb) |
| **8. Defenses** | 8a | Differential Privacy & DP-SGD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_08_defenses/Lab_8a_Differential_Privacy_and_DP_SGD.ipynb) |
| | 8b | Federated Learning & Adversarial Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELTE-DSED/Intro-Data-Security/blob/main/module_08_defenses/Lab_8b_Federated_Learning_and_Adversarial_Training.ipynb) |


## Learning Outcomes

| # | Skill | Description |
|---|-------|-------------|
| 1 | **Understand** | Fundamental concepts of machine-learning security and privacy |
| 2 | **Implement** | State-of-the-art attacks (Evasion, Poisoning, Inversion) in PyTorch |
| 3 | **Evaluate** | Model robustness using quantitative metrics and certified bounds |
| 4 | **Design** | Multi-layered defense strategies (DP, FL, Robust Training) for production |
| 5 | **Generate** | Privacy-preserving synthetic data for sensitive domains (healthcare, finance) |


## References & Acknowledgments

- [unica-mlsec/mlsec](https://github.com/unica-mlsec/mlsec) — Prof. Battista Biggio (University of Cagliari)
- *Practical Data Privacy* — Katharine Jarmul (O'Reilly, 2023)
- *Adversarial Machine Learning* — Goodfellow, Biggio et al. (Cambridge University Press)
