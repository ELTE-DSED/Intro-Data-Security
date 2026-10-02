---
layout: page
title: Materials
permalink: /materials/
---

## Setup

You do not need to install anything. Every lab is a Jupyter notebook that runs
in Google Colab:

1. Sign in to Google with your ELTE account (or any Google account).
2. Open a lab with the **Open in Colab** button on the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page.
3. Choose *Runtime &rarr; Run all* to execute the notebook from top to bottom.

Colab opens notebooks straight from the
[ELTE-DSED/Intro-Data-Security]({{ site.repo_url }}) repository, so the notebooks
are always up to date with the latest version.

### GPU access

A few labs (notably the poisoning, backdoor and federated learning labs) are much
faster on a GPU. In Colab, choose *Runtime &rarr; Change runtime type &rarr; T4
GPU*. Free accounts get a limited amount of GPU time per day, which is enough
for these labs.

### Running locally instead

If you prefer a local setup, install Jupyter and the dependencies the notebooks
declare. The notebooks are plain PyTorch, so roughly:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pandas
pip install adversarial-robustness-toolbox
jupyter lab
```

## Course Repository

All lab notebooks, images and helper modules live in one place:

- Repository: [{{ site.repo_url }}]({{ site.repo_url }})
- Issues and questions: [open an issue]({{ site.repo_url }}/issues)

## Background Reading

- [unica-mlsec/mlsec](https://github.com/unica-mlsec/mlsec), Prof. Battista Biggio (University of Cagliari)
- *Practical Data Privacy*, Katharine Jarmul (O'Reilly, 2023)
- *Adversarial Machine Learning*, Goodfellow, Biggio, Laskov (Cambridge University Press, 2018)

## Academic Integrity

The labs are graded coursework. You are expected to submit your own work and to
understand every line of the notebooks you submit. Reusing or adapting
published attack implementations is fine as long as you cite them and can
explain what they do.