---
layout: home
---

## Prerequisites

Students should be comfortable with:
- **Programming:** Python and basic PyTorch (tensors, neural network layers, custom training loops, and data loaders).
- **Machine Learning Foundations:** Supervised learning, loss functions, optimization via gradient descent, and standard classification metrics.
- **Compute:** No dedicated local hardware or GPU is required &mdash; all labs run directly in Google Colab.

## Course Topics

The course covers 8 core modules across the machine learning security lifecycle:

1. **Module 1: Foundations** &mdash; Deep neural network training and robust model baselines
2. **Module 2: Input Manipulation** &mdash; Evasion attacks at inference time (FGSM, PGD)
3. **Module 3: Data Poisoning** &mdash; Training set manipulation (label flipping, backdoor triggers, and clean-label poisoning)
4. **Module 4: Model Poisoning** &mdash; Supply chain risks, pre-trained weight trojans, and certified trojan detection
5. **Module 5: Availability Attacks** &mdash; Sponge examples, inference latency inflation, and energy-depletion defenses
6. **Module 6: Confidentiality &amp; Privacy** &mdash; Membership Inference Attacks (MIA) and model inversion
7. **Module 7: Synthetic Data** &mdash; Privacy-preserving tabular generation using VAEs and GANs
8. **Module 8: Defenses &amp; Robustness** &mdash; Differential Privacy (DP-SGD), Federated Learning, and Adversarial Training

Interactive notebooks, launch links, and detailed lab descriptions are available on the [**Labs**]({{ '/labs/' | prepend: site.baseurl }}) page.

## Learning Outcomes

By the end of this course, students will be able to:

- **Understand** fundamental security and privacy threat models across the machine learning lifecycle.
- **Implement** state-of-the-art adversarial attacks against deep neural networks in PyTorch.
- **Evaluate** model vulnerability using quantitative empirical metrics and certified robustness bounds.
- **Design & Deploy** privacy-preserving and robust defenses, including Differential Privacy (DP-SGD), adversarial training, and federated learning protocols.
- **Generate** privacy-preserving synthetic tabular datasets for sensitive domain applications.

## Getting Started

1. Check the [Schedule]({{ '/schedule/' | prepend: site.baseurl }}) for weekly lab sessions and assignment deadlines.
2. Open the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page to access interactive notebooks and launch them in Google Colab using your Google account.
3. Consult the [Materials]({{ '/materials/' | prepend: site.baseurl }}) page for setup recommendations, background reading, and academic integrity policies.
