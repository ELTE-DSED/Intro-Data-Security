<p align="center">
  <img src="https://img.shields.io/badge/Level-Master's-orange" alt="Level"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="Framework"/>
  <img src="https://img.shields.io/badge/Security-Foolbox%20%7C%20ART%20%7C%20Opacus-blueviolet" alt="Toolkits"/>
  <img src="https://img.shields.io/badge/Colab-Ready-blue" alt="Colab Ready"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

# Introduction to Data Security Practicum

> A comprehensive, hands-on master's course on the security, privacy, and safety of machine learning and generative AI systems. Students learn to **attack**, **defend**, and **audit** AI models through practical Jupyter labs.


## Course Topics (2027 Edition)

The curriculum covers 8 core modules across the machine learning and GenAI security lifecycle:

1. **Module 1: Foundations**: Deep neural network training, loss landscapes, and robust model baselines in PyTorch.
2. **Module 2: Input Manipulation**: Evasion attacks at inference time (FGSM, PGD) benchmarked using **Foolbox** and **ART**.
3. **Module 3: Poisoning & Supply Chain**: Training set manipulation, clean-label backdoors, model Trojans, and certified detection (Neural Cleanse).
4. **Module 4: Availability & Resource Exhaustion**: Sponge examples, latency and energy inflation, and multi-layer DoS defenses.
5. **Module 5: Confidentiality & Privacy Auditing**: Membership Inference Attacks (MIA) using **ART**, ROC-AUC evaluation, and model inversion.
6. **Module 6: Synthetic Data Generation**: Privacy-preserving tabular data generation with Variational Autoencoders (VAEs) and CTGAN.
7. **Module 7: Provable Defenses**: Formal $(\epsilon, \delta)$-Differential Privacy (DP-SGD) with Meta's **Opacus**, Federated Learning, and Adversarial Training.
8. **Module 8: GenAI & LLM Security**: Prompt injection, automated jailbreaking (GCG), safety alignment auditing, and RAG data poisoning.

For the full lab schedule, detailed descriptions, and interactive Colab notebooks, visit the [Course Labs Page](https://elte-dsed.github.io/Intro-Data-Security/labs/).

## Course Website & Repository Architecture

The course website is built with Jekyll and automatically deployed to GitHub Pages via GitHub Actions.

| Path | Purpose |
|------|---------|
| `modules/` | All 8 thematic course modules containing Jupyter notebooks (`.ipynb`), helper scripts, and image assets. |
| `_data/labs.yml` | **Single source of truth** for all labs (titles, dates, Colab paths, summaries). |
| `_data/modules.yml` | The 8 modules, their titles, and overviews. |
| `_data/events.yml` | Course calendar events (practical evaluations, security clinics, and holidays). |
| `_data/people.yml` | Course instructors and teaching assistants. |
| `_data/nav.yml` | Header navigation menu. |
| `_announcements/` | Markdown announcements shown in the Updates box on the home page. |
| `_config.yml` | Site configuration, semester, and exclude rules. |
| `_layouts/`, `_includes/`, `_sass/` | HTML templates, modular partials, and SCSS stylesheets. |
| `index.md`, `schedule.md`, `labs.md`, `resources.md` | The core site pages. |

To add or update a lab, edit **only** `_data/labs.yml`:

```yaml
- id: lab-8
  module: 8
  title: "LLM Security: Prompt Injection & Jailbreaking"
  date: 2027-05-04T10:00:00+02:00
  notebook: modules/module_08_genai_security/Lab_8_LLM_Security_Prompt_Injection_and_Jailbreaks.ipynb
  summary: "Red-team foundation models against direct and indirect prompt injection and GCG jailbreaks."
```

The lab appears automatically on the Labs page, the Schedule, and the Updates feed.

Running the site locally requires Ruby 3.3+ and Bundler:

```bash
bundle install
bundle exec jekyll serve
```

The site is served locally at <http://127.0.0.1:4000/Intro-Data-Security/>.


## Instructors & Staff

- **Instructor**: Imre Lendák, Associate Professor ([staff page](https://www.inf.elte.hu/en/staff/imre-lendak))
- **Teaching Assistant**: Ahmed Fouad Lagha, PhD Candidate ([homepage](https://ahmed-fouad-lagha.github.io))

---

© 2027 ELTE Faculty of Informatics, Department of Data Science and Engineering
