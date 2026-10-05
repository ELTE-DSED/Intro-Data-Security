---
layout: page
title: Resources
permalink: /resources/
---

## Compute & Environment Setup

All laboratory sessions are delivered as interactive Jupyter notebooks designed to run directly in Google Colab with zero local installation required.

### Google Colab (Recommended)

1. Sign in with your Google account.
2. Open any lab notebook using the **Open in Colab** button on the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page.
3. Select *Runtime &rarr; Run all* to execute the notebook from start to finish.

Notebooks are pulled directly from the [{{ site.repo_name }}]({{ site.repo_url }}) repository to ensure you are always working with the current version.

#### Hardware Acceleration (GPU)

Certain laboratories, such as data poisoning, backdoor injection, and federated learning simulations, run substantially faster with hardware acceleration. To enable a GPU in Google Colab:

- Navigate to *Runtime &rarr; Change runtime type*.
- Under *Hardware accelerator*, select **T4 GPU** and click *Save*.

Standard free Colab quotas provide sufficient daily GPU compute time for all course assignments.

### Local Python Environment

If you prefer working locally, configure a dedicated virtual environment with Python 3.10+ and install the core dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Core scientific computing and deep learning
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pandas matplotlib seaborn jupyterlab

# Security, robustness, privacy, and LLM toolkits
pip install adversarial-robustness-toolbox foolbox opacus
pip install transformers accelerate
```

Start the Jupyter environment with:

```bash
jupyter lab
```

---

## Security & Robustness Toolkits

The following open-source frameworks provide state-of-the-art implementations of attacks, defenses, verification techniques, and privacy-preserving primitives:

- **[Adversarial Robustness Toolbox (ART)](https://github.com/trusted-ai/adversarial-robustness-toolbox)** (IBM / Linux Foundation)  
  A comprehensive Python library for machine learning security providing defense and evaluation against evasion, poisoning, extraction, and inference across all major ML frameworks (PyTorch, TensorFlow, Scikit-learn, ONNX).

- **[Foolbox](https://github.com/bethgelab/foolbox)** (Bethge Lab, University of Tübingen)  
  A high-performance Python toolbox that lets you benchmark the adversarial robustness of neural networks with native PyTorch support and standardized $L_p$ perturbation metrics.

- **[Opacus](https://github.com/pytorch/opacus)** (Meta / PyTorch)  
  A high-speed, scalable library for training PyTorch models with Differential Privacy (DP-SGD) with minimal code changes and minimal performance overhead.

- **[SecML: Secure Machine Learning](https://github.com/pralab/secml)** (PRALab, University of Cagliari)  
  A Python library for secure and explainable machine learning developed by Prof. Battista Biggio's research team, featuring gradient-based evasion and poisoning attacks.

- **[PySyft](https://github.com/OpenMined/PySyft)** (OpenMined)  
  A foundational library for secure, privacy-preserving computation, federated learning, and remote data science using secure multi-party computation and differential privacy.

- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**  
  The industry-standard library for loading, fine-tuning, and red-teaming open-source foundation models and generative AI systems.

---

## Foundational Books & Textbooks

- **Adversarial Machine Learning**  
  *Ian Goodfellow, Battista Biggio, and Pavel Laskov* (Cambridge University Press, 2018).  
  A foundational reference covering the threat landscape, evasion mechanics, poisoning, and game-theoretic formulations of ML security.

- **Practical Data Privacy: Enhancing Machine Learning with Robust Data Protection**  
  *Katharine Jarmul* (O'Reilly Media, 2023).  
  An applied, modern engineering guide covering differential privacy, synthetic data generation, cryptographic techniques, and privacy auditing for production ML systems.

- **The Algorithmic Foundations of Differential Privacy**  
  *Cynthia Dwork and Aaron Roth* (Foundations and Trends in Theoretical Computer Science, 2014).  
  The seminal mathematical monograph establishing the formal privacy definitions, Laplace and Gaussian mechanisms, composition theorems, and query release algorithms.

- **Deep Learning**  
  *Ian Goodfellow, Yoshua Bengio, and Aaron Courville* (MIT Press, 2016).  
  The definitive textbook covering representation learning, optimization dynamics, and deep neural network architectures.

---

## Seminal Research Papers

### Evasion & Adversarial Perturbations

- **Explaining and Harnessing Adversarial Examples**  
  *Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy* (ICLR 2015)  
  Introduces the Fast Gradient Sign Method (FGSM) and establishes the hypothesis that neural network vulnerability stems from linear behavior in high-dimensional spaces.  
  [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)

- **Towards Evaluating the Robustness of Neural Networks**  
  *Nicholas Carlini and David Wagner* (IEEE Symposium on Security and Privacy 2017)  
  Formulates targeted optimization-based attacks ($L_0$, $L_2$, $L_\infty$) that defeated defensive distillation and established the gold standard for empirical robustness evaluation.  
  [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)

- **Towards Deep Learning Models Resistant to Adversarial Attacks**  
  *Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* (ICLR 2018)  
  Casts adversarial defense as robust optimization and shows that Projected Gradient Descent (PGD) adversarial training provides principled empirical defense against first-order adversaries.  
  [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

- **HopSkipJumpAttack: A Query-Efficient Decision-Based Attack**  
  *Jianbo Chen, Michael I. Jordan, Martin J. Wainwright* (IEEE Symposium on Security and Privacy 2020)  
  Presents a decision-based black-box boundary attack relying solely on predicted class labels without probability or gradient feedback.  
  [arXiv:1904.02144](https://arxiv.org/abs/1904.02144)

### Data Poisoning & Backdoor Trojans

- **BadNets: Evaluating Backdooring Attacks on Deep Neural Networks**  
  *Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg* (IEEE Access 2019 / arXiv 2017)  
  Demonstrates targeted backdoor trojans in image classifiers that retain clean accuracy while misclassifying triggered inputs.  
  [arXiv:1708.06733](https://arxiv.org/abs/1708.06733)

- **Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks**  
  *Ali Shafahi, W. Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, Tom Goldstein* (NeurIPS 2018)  
  Shows that an attacker can inject correctly labeled, imperceptibly modified training images to hijack classifications at inference time.  
  [arXiv:1804.00792](https://arxiv.org/abs/1804.00792)

### Privacy Attacks & Defenses

- **Membership Inference Attacks Against Machine Learning Models**  
  *Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov* (IEEE Symposium on Security and Privacy 2017)  
  Uses shadow model training to determine with high confidence whether an individual record was part of a target model's private training dataset.  
  [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)

- **Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures**  
  *Matt Fredrikson, Somesh Jha, Thomas Ristenpart* (ACM CCS 2015)  
  Reconstructs recognizable face images of individuals from facial recognition classifiers using output class probabilities and gradient descent on the input space.  
  [DOI:10.1145/2810103.2813677](https://doi.org/10.1145/2810103.2813677)

- **Deep Learning with Differential Privacy**  
  *Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang* (ACM CCS 2016)  
  Develops Differentially Private Stochastic Gradient Descent (DP-SGD) with gradient clipping, Gaussian noise addition, and the Moments Accountant for tight privacy budget tracking.  
  [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

### Model Extraction & Confidentiality

- **Stealing Machine Learning Models via Prediction APIs**  
  *Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, Thomas Ristenpart* (USENIX Security 2016)  
  Demonstrates practical query-based extraction of proprietary model parameters and hyperplanes across commercial ML services.  
  [arXiv:1609.02943](https://arxiv.org/abs/1609.02943)

### Generative AI & Foundation Model Security

- **Universal and Transferable Adversarial Attacks on Aligned Language Models**  
  *Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson* (arXiv 2023)  
  Introduces Greedy Coordinate Gradient (GCG) search, demonstrating that token suffixes can reliably bypass alignment across proprietary and open-source LLMs.  
  [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)

- **Not what you've signed up for: Compromising Real-World LLM Applications with Indirect Prompt Injection**  
  *Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz* (AISec 2023)  
  Formalizes indirect prompt injection as a fundamental threat vector where untrusted data sources hijack autonomous LLM application control flow.  
  [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)

---


## Academic Integrity & Responsible Research

The techniques examined in this course (adversarial attacks, backdoor injection, membership inference, and model extraction) are studied exclusively for vulnerability assessment, threat modeling, and defense engineering.

Students are expected to submit original work for all laboratory exercises and projects. Understanding and explaining the algorithmic steps of your implementation is mandatory. When adapting published attack methods or referencing external implementations, provide explicit attribution and citations.

---

## Course Repository

All course notebooks, helper scripts, and datasets are hosted centrally:

- **Repository**: [{{ site.repo_url }}]({{ site.repo_url }})
- **Issue Tracker**: [Submit questions or errata]({{ site.repo_url }}/issues)
