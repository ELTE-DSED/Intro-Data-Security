<p align="center">
  <img src="https://img.shields.io/badge/Term-Spring%202027-blue" alt="Term"/>
  <img src="https://img.shields.io/badge/Level-Master's-orange" alt="Level"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="Framework"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

# Introduction to Data Security Practicum

> This course provides a comprehensive, hands-on introduction to the security and privacy of machine learning systems. Students will learn to **attack**, **defend**, and **audit** AI models through 13 practical labs organized into 8 thematic modules.

**Course website:** <https://elte-dsed.github.io/Intro-Data-Security/>

---

## Instructors & Staff

- **Instructor**: Imre Lendák, Associate Professor ([staff page](https://www.inf.elte.hu/en/staff/imre-lendak))
- **Teaching Assistant**: Ahmed Fouad Lagha, PhD Candidate ([homepage](https://ahmed-fouad-lagha.github.io))

## Lab Curriculum

| Module | Lab | Topic | Link |
|--------|-----|-------|------|
| **1. Foundations** | 1 | DNN Training & Robust Model Baselines | [Notebook](module_01_foundations/Lab1_DNN_Training_and_Robust_Models.ipynb) |
| **2. Input Manipulation** | 2 | Evasion Attacks (FGSM, PGD) | [Notebook](module_02_input_manipulation/Lab2_Evasion_Attacks.ipynb) |
| **3. Data Poisoning** | 3a | Label Flipping Attacks | [Notebook](module_03_data_poisoning/Lab_3a_Data_Poisoning_Label_Flipping.ipynb) |
| | 3b | Backdoor & Trigger Injection | [Notebook](module_03_data_poisoning/Lab_3b_Data_Poisoning_Backdoor_Attacks.ipynb) |
| **4. Model Poisoning** | 4a | Model Trojans & Supply Chain Attacks | [Notebook](module_04_model_poisoning/Lab_4a_Model_Trojans_and_Supply_Chain_Attacks.ipynb) |
| | 4b | Trojan Detection & Certified Defenses | [Notebook](module_04_model_poisoning/Lab_4b_Trojan_Detection_and_Certified_Defenses.ipynb) |
| **5. Availability** | 5a | Sponge Attacks & Resource Exhaustion | [Notebook](module_05_sponge_attacks/Lab_5a_Sponge_Attacks_and_Resource_Exhaustion.ipynb) |
| | 5b | Sponge Attack Defenses | [Notebook](module_05_sponge_attacks/Lab_5b_Sponge_Attack_Defenses_and_Resource_Constraints.ipynb) |
| **6. Confidentiality** | 6a | Membership Inference Attacks | [Notebook](module_06_confidentiality_attacks/Lab_6a_Membership_Inference_Attacks.ipynb) |
| | 6b | Model Inversion & Feature Reconstruction | [Notebook](module_06_confidentiality_attacks/Lab_6b_Model_Inversion_Attacks_and_Defenses.ipynb) |
| **7. Synthetic Data** | 7 | Tabular Synthetic Data (VAE, GAN) | [Notebook](module_07_synthetic_data_generation/Lab_7_Tabular_Synthetic_Data_Generation.ipynb) |
| **8. Defenses** | 8a | Differential Privacy & DP-SGD | [Notebook](module_08_defenses/Lab_8a_Differential_Privacy_and_DP_SGD.ipynb) |
| | 8b | Federated Learning & Adversarial Training | [Notebook](module_08_defenses/Lab_8b_Federated_Learning_and_Adversarial_Training.ipynb) |

---

## Course website

The course website is a Jekyll site built from this repository and deployed to
the `gh-pages` branch. It is based on
[kazemnejad/jekyll-course-website-template](https://github.com/kazemnejad/jekyll-course-website-template)
(itself derived from [svmiller/course-website](https://github.com/svmiller/course-website)),
adapted for a lab-based course.

### Running it locally

You need Ruby and Bundler. From the repository root:

```bash
bundle install
bundle exec jekyll serve
```

The site is then at <http://127.0.0.1:4000/Intro-Data-Security/>.

Note that the lab notebooks themselves are **not** rendered into the site. Each
lab links to the notebook on GitHub and opens it directly in Google Colab, so
students never need a local setup.

### Where things live

| Path | Purpose |
|------|---------|
| `_data/labs.yml` | **Every lab**: title, date, summary, notebook path. The main file to edit. |
| `_images/logo.svg` | The ELTE emblem used in the header, cropped to its 269×269 square. |
| `_data/modules.yml` | The 8 modules and their descriptions. |
| `_data/people.yml` | Instructors and TAs. Photos are optional — without one, a person shows their initials. |
| `_images/pp/` | Staff photos, referenced by `profile_pic` in `_data/people.yml`. |
| `_data/nav.yml` | Top navigation menu. |
| `_data/events.yml` | Deadlines (`type: due`) and custom events (`type: raw_event`). |
| `_announcements/*.md` | Manual announcements shown in the Updates box on the home page. |
| `_config.yml` | Site name, semester, department, and the `exclude` list. |
| `_layouts/`, `_includes/`, `_sass/` | Templates and styling. |
| `index.md`, `schedule.md`, `labs.md`, `materials.md` | The four site pages. |

### Adding a lab

Append an entry to `_data/labs.yml`:

```yaml
- id: lab-9
  module: 8
  title: Robust Distillation Under Attack
  date: 2027-05-11T10:00:00+02:00
  notebook: module_08_defenses/Lab_9_Robust_Distillation.ipynb
  summary: One-line description shown on the Labs page.
```

The lab then appears automatically on the home-page curriculum table, the Labs
page (under `module: 8`), the Schedule, and the Updates box. Add a matching
deadline in `_data/events.yml` if it is graded, with `lab: lab-9` pointing back
at the lab so CI can check the two titles agree.

### Dates and timezones

Dates are written with an explicit UTC offset, and the offset **changes with
daylight saving**: Budapest is `+01:00` in winter and `+02:00` from the last
Sunday of March. A single hardcoded offset is a real trap — it renders summer
deadlines an hour late, which pushes a `23:59` deadline onto the following day.

`timezone: Europe/Budapest` in `_config.yml` pins the display timezone so the
same source renders identically on your machine, in CI, and on GitHub Pages.

If you ever need to publish a provisional timetable, set
`dates_are_placeholders: true` in `_config.yml`; the Schedule then shows a
warning banner. It is `false` now that the Spring 2027 dates are real, and CI
asserts the banner appears **only** when the flag is set.

Note that lab sessions start at 10:00. That is an assumption, not a confirmed
teaching time — correct it in `_data/labs.yml` if ELTE schedules them otherwise.

### Deployment

- `.github/workflows/build-site.yml` builds the site and **fails on any Liquid
  warning**, so template mistakes do not reach `main`. It also asserts on the
  generated markup, because Jekyll exits 0 even when a loop renders nothing:
  every lab has a Colab badge and an anchor, the schedule has 28 rows, no
  front-matter leaks into a page, no deadline crossed midnight, the curriculum
  table stays horizontally scrollable, and every deadline's title still matches
  its lab in `_data/labs.yml`. It runs on every branch.
- `.github/workflows/deploy-site.yml` builds on `main` and force-pushes `_site`
  to the `gh-pages` branch, which is the branch GitHub Pages is set to serve.

GitHub Pages already serves the `gh-pages` branch, so no change to repository
settings is needed.

---

## References & Acknowledgments

- [unica-mlsec/mlsec](https://github.com/unica-mlsec/mlsec) — Prof. Battista Biggio (University of Cagliari)
- *Practical Data Privacy* — Katharine Jarmul (O'Reilly, 2023)
- *Adversarial Machine Learning* — Goodfellow, Biggio, Laskov (Cambridge University Press, 2018)
- [jekyll-course-website-template](https://github.com/kazemnejad/jekyll-course-website-template) — Kazemnejad (MIT License)

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
© 2027 ELTE Faculty of Informatics, Department of Data Science and Engineering
— 1117 Budapest, Pázmány Péter sétány 1/C, Hungary