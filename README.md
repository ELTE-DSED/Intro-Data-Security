# Introduction to Data Security Practicum

<p align="center">
  <img src="https://img.shields.io/badge/Term-Spring%202026-blue" alt="Term"/>
  <img src="https://img.shields.io/badge/Level-Master's-orange" alt="Level"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="Framework"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

> **A hands-on practicum for adversarial machine learning, privacy attacks, and secure AI systems.**

**Department of Data Science and Engineering (DDSE)**  
**Faculty of Informatics, Eötvös Loránd University (ELTE)**

---

## Instructors & Staff

| Role | Name | Affiliation |
|------|------|-------------|
| **Instructor** | Prof. Lendák Imre | Full Professor, ELTE |
| **Teaching Assistant** | Ahmed F. Lagha | PhD Candidate, ELTE |

## Course Overview

This course provides a comprehensive, hands-on introduction to the security and privacy of machine learning systems. Students will learn to **attack**, **defend**, and **audit** AI models through 16 practical labs organized into 9 thematic modules.

### Key Topics
- **Adversarial Attacks:** Evasion, Poisoning, Trojans, Prompt Injection
- **Privacy Attacks:** Membership Inference, Model Inversion
- **Defenses:** Differential Privacy (DP-SGD), Federated Learning, Adversarial Training
- **Synthetic Data:** VAE/GAN/Diffusion for privacy-preserving data generation

## Lab Curriculum

The course is organized into **9 modules**, transitioning from foundations to advanced security integration.

### Module 1: Foundations & Threat Modeling
| Lab | Topic | Link |
|-----|-------|------|
| 1a | DNN Training & Robust Model Baselines | [Notebook](module_01_foundations/Lab_1a_DNN_Training_and_Robust_Models.ipynb) |
| 1b | Threat Modeling & Attack Taxonomy | [Notebook](module_01_foundations/Lab_1b_Threat_Modeling_and_Attack_Taxonomy.ipynb) |

### Module 2: Input Data Manipulation Attacks
| Lab | Topic | Link |
|-----|-------|------|
| 2a | Evasion Attacks (FGSM, PGD) | [Notebook](module_02_input_manipulation/Lab_2a_Evasion_Attacks.ipynb) |
| 2b | Adversarial Prompt Engineering (LLMs) | [Notebook](module_02_input_manipulation/Lab_2b_Adversarial_Prompt_Engineering.ipynb) |

### Module 3: Integrity Attacks — Data Poisoning
| Lab | Topic | Link |
|-----|-------|------|
| 3a | Data Poisoning Fundamentals (Label Flipping) | [Notebook](module_03_data_poisoning/Lab_3a_Data_Poisoning_Label_Flipping.ipynb) |
| 3b | Backdoor & Trigger Injection | [Notebook](module_03_data_poisoning/Lab_3b_Data_Poisoning_Backdoor_Attacks.ipynb) |

### Module 4: Integrity Attacks — Model Poisoning
| Lab | Topic | Link |
|-----|-------|------|
| 4a | Model Trojans & Supply Chain Attacks | [Notebook](module_04_model_poisoning/Lab_4a_Model_Trojans_and_Supply_Chain_Attacks.ipynb) |
| 4b | Trojan Detection & Certified Defenses | [Notebook](module_04_model_poisoning/Lab_4b_Trojan_Detection_and_Certified_Defenses.ipynb) |

### Module 5: Availability Attacks
| Lab | Topic | Link |
|-----|-------|------|
| 5a | Sponge Attacks & Resource Exhaustion | [Notebook](module_05_sponge_attacks/Lab_5a_Sponge_Attacks_and_Resource_Exhaustion.ipynb) |
| 5b | Sponge Attack Defenses & Resource Constraints | [Notebook](module_05_sponge_attacks/Lab_5b_Sponge_Attack_Defenses_and_Resource_Constraints.ipynb) |

### Module 6: Confidentiality Attacks
| Lab | Topic | Link |
|-----|-------|------|
| 6a | Membership Inference Attacks | [Notebook](module_06_confidentiality_attacks/Lab_6a_Membership_Inference_Attacks.ipynb) |
| 6b | Model Inversion & Feature Reconstruction | [Notebook](module_06_confidentiality_attacks/Lab_6b_Model_Inversion_Attacks_and_Defenses.ipynb) |

### Module 7: Synthetic Data Generation
| Lab | Topic | Link |
|-----|-------|------|
| 7a | Tabular Synthetic Data (VAE, GAN) | [Notebook](module_07_synthetic_data_generation/Lab_7a_Tabular_Synthetic_Data_Generation.ipynb) |
| 7b | Time-Series Synthetic Data (LSTM-VAE, Diffusion) | [Notebook](module_07_synthetic_data_generation/Lab_7b_Time_Series_Synthetic_Data_Generation.ipynb) |

### Module 8: Defenses & Secure Learning
| Lab | Topic | Link |
|-----|-------|------|
| 8a | Differential Privacy & DP-SGD | [Notebook](module_08_defenses/Lab_8a_Differential_Privacy_and_DP_SGD.ipynb) |
| 8b | Federated Learning & Adversarial Training | [Notebook](module_08_defenses/Lab_8b_Federated_Learning_and_Adversarial_Training.ipynb) |

### Module 9: Integration & Capstone
| Lab | Topic | Link |
|-----|-------|------|
| 9 | End-to-End Secure ML Pipeline | [Notebook](module_09_capstone_integration/Lab_9_Capstone_End_to_End_Secure_ML_Pipeline.ipynb) |

## Learning Outcomes

By the end of this course, students will be able to:

| # | Skill | Description |
|---|-------|-------------|
| 1 | **Understand** | Fundamental concepts of machine-learning security and privacy |
| 2 | **Implement** | State-of-the-art attacks (Evasion, Poisoning, Inversion) in PyTorch |
| 3 | **Evaluate** | Model robustness using quantitative metrics and certified bounds |
| 4 | **Design** | Multi-layered defense strategies (DP, FL, Robust Training) for production |
| 5 | **Generate** | Privacy-preserving synthetic data for sensitive domains (healthcare, finance) |


## References & Acknowledgments

This course is built upon foundational materials from:

- [unica-mlsec/mlsec](https://github.com/unica-mlsec/mlsec) — Prof. Battista Biggio (University of Cagliari)
- *Practical Data Privacy* — Katharine Jarmul (O'Reilly, 2023)
- *Adversarial Machine Learning* — Goodfellow, Biggio et al. (Cambridge University Press)

## Technical Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, torchvision |
| **Data Science** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Privacy** | Opacus (optional for DP-SGD) |

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>© 2026 ELTE Department of Data Science and Engineering</b>
</p>
