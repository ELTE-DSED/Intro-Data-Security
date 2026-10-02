<p align="center">
  <img src="https://img.shields.io/badge/Level-Master's-orange" alt="Level"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="Framework"/>
  <img src="https://img.shields.io/badge/Colab-Ready-blue" alt="Colab Ready"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

# Introduction to Data Security Practicum

> A comprehensive, hands-on master's course on the security and privacy of machine learning systems. Students learn to **attack**, **defend**, and **audit** AI models through practical Jupyter labs.


## Course Topics

The course covers 8 core modules across the machine learning security lifecycle:

1. **Module 1: Foundations**: Deep neural network training and robust model evaluation.
2. **Module 2: Input Manipulation**: Evasion attacks at inference time (FGSM, PGD).
3. **Module 3: Data Poisoning**: Training set manipulation (label flipping, backdoors, and trigger injection).
4. **Module 4: Model Poisoning**: Supply chain vulnerabilities, model Trojans, and certified detection.
5. **Module 5: Availability Attacks**: Sponge examples, latency inflation, and resource exhaustion defenses.
6. **Module 6: Confidentiality & Privacy**: Membership Inference Attacks (MIA) and model inversion.
7. **Module 7: Synthetic Data**: Privacy-preserving tabular data generation with VAEs and GANs.
8. **Module 8: Defenses & Robustness**: Differential Privacy (DP-SGD), Federated Learning, and Adversarial Training.

For the full lab schedule, detailed descriptions, and interactive Colab notebooks, visit the [Course Labs Page](https://elte-dsed.github.io/Intro-Data-Security/labs/).

## Course Website & Repository Architecture

The course website is built with Jekyll and automatically deployed to GitHub Pages via GitHub Actions.


| Path | Purpose |
|------|---------|
| `modules/` | All 8 thematic course modules containing Jupyter notebooks (`.ipynb`), helper scripts, and image assets. |
| `_data/labs.yml` | **Single source of truth** for all labs (titles, dates, Colab paths, summaries). |
| `_data/modules.yml` | The 8 modules, their titles, and overviews. |
| `_data/events.yml` | Deadlines (`type: due`) and custom calendar events (`type: raw_event`). Titles inherit from `labs.yml`. |
| `_data/people.yml` | Course instructors and teaching assistants. |
| `_data/nav.yml` | Header navigation menu. |
| `_announcements/` | Markdown announcements shown in the Updates box on the home page. |
| `_config.yml` | Site configuration, semester, and exclude rules. |
| `_layouts/`, `_includes/`, `_sass/` | HTML templates, modular partials, and SCSS stylesheets. |
| `index.md`, `schedule.md`, `labs.md`, `materials.md` | The core site pages. |


To add or update a lab, edit **only** `_data/labs.yml`:

```yaml
- id: lab-9
  module: 8
  title: Robust Distillation Under Attack
  date: 2027-05-11T10:00:00+02:00
  notebook: modules/module_08_defenses/Lab_9_Robust_Distillation.ipynb
  summary: One-line description shown on the Labs page.
```

The lab appears automatically on the Labs page, the Schedule, and the Updates feed. If graded, add a deadline entry in `_data/events.yml` (`lab: lab-9`), which automatically inherits the lab's title.

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
