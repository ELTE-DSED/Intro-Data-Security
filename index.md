---
layout: home
---

## Prerequisites

Students should be comfortable with:
- **Programming:** Python and basic PyTorch (tensors, neural network layers, custom training loops, and data loaders).
- **Machine Learning Foundations:** Supervised learning, loss functions, optimization via gradient descent, and standard classification metrics.
- **Compute:** No dedicated local hardware or GPU is required. All labs run directly in Google Colab.

## Course Topics

The course covers 8 core modules across the machine learning security lifecycle:

1. **Module 1: Foundations**: Deep neural network training and robust model baselines
2. **Module 2: Input Manipulation**: Evasion attacks at inference time (FGSM, PGD)
3. **Module 3: Data Poisoning**: Training set manipulation (label flipping, backdoor triggers, and clean-label poisoning)
4. **Module 4: Model Poisoning**: Supply chain risks, pre-trained weight trojans, and certified trojan detection
5. **Module 5: Availability Attacks**: Sponge examples, inference latency inflation, and energy-depletion defenses
6. **Module 6: Confidentiality &amp; Privacy**: Membership Inference Attacks (MIA) and model inversion
7. **Module 7: Synthetic Data**: Privacy-preserving tabular generation using VAEs and GANs
8. **Module 8: Defenses &amp; Robustness**: Differential Privacy (DP-SGD), Federated Learning, and Adversarial Training

Interactive notebooks, launch links, and detailed lab descriptions are available on the [**Labs**]({{ '/labs/' | prepend: site.baseurl }}) page.

## Learning Outcomes

By the end of this course, students will be able to:

- **Understand** fundamental security and privacy threat models across the machine learning lifecycle.
- **Implement** state-of-the-art adversarial attacks against deep neural networks in PyTorch.
- **Evaluate** model vulnerability using quantitative empirical metrics and certified robustness bounds.
- **Design & Deploy** privacy-preserving and robust defenses, including Differential Privacy (DP-SGD), adversarial training, and federated learning protocols.
- **Generate** privacy-preserving synthetic tabular datasets for sensitive domain applications.

## Grading & Evaluation

This course is an applied practicum (**Gyakorlat**) evaluated via continuous assessment throughout the 14-week semester, resulting in a practical grade (**Gyakorlati jegy**).

To reflect modern engineering realities, take-home notebooks serve as continuous practice, while the majority of the grade is evaluated through live, supervised in-class practical milestones to ensure students develop genuine, independent problem-solving skills.

### Weight Distribution

| Component | Weight | Assessment Format | Schedule & Scope |
|-----------|--------|-------------------|------------------|
| **Lab Attendance & In-Class Work** | **15%** | Active participation during weekly sessions | Mandatory weekly presence (max 3 absences per TVSZ) |
| **Weekly Laboratory Notebooks** | **25%** | Hands-on PyTorch notebooks executed in Colab | Submitted weekly by 23:59 via Google Colab |
| **Midterm Practical Evaluation (ZH 1)** | **30%** | In-class supervised practical test | Week 7: Modules 1 to 4 (Foundations, Evasion, Poisoning, Trojans) |
| **End-Term Practical Evaluation (ZH 2)** | **30%** | In-class supervised practical test | Week 14: Modules 5 to 8 (Availability, Privacy, Synthetic Data, Defenses) |

---

### Course Components & Policies

#### 1. Lab Attendance & In-Class Engagement (15%)
Laboratory sessions provide guided environments to implement attacks, inspect model behavior under perturbation, and verify defenses under instructor supervision.
- **Attendance Requirement:** In accordance with Section 50 of the ELTE Academic and Examination Regulations (TVSZ), attendance at practical classes is mandatory. Missing more than 3 lab sessions without an officially certified medical or institutional excuse results in refusal of the practical mark (**aláírás megtagadva / Grade 1**).
- **In-Class Activity:** Full points require on-time arrival and active engagement with the assigned notebook during class hours.

#### 2. Weekly Laboratory Notebooks (25%)
Weekly assignments reinforce lecture concepts through applied experimentation.
- **Format:** Interactive Jupyter notebooks completed via Google Colab. Submitted notebooks must include all code cells and executed outputs.
- **Deadlines:** Submissions are due by **23:59** on the dates indicated on the [Schedule]({{ '/schedule/' | prepend: site.baseurl }}) page.
- **Late Policy:** Late submissions receive a 10% point deduction per 24 hours, accepted up to a maximum of 48 hours late.
- **Evaluation Criteria:** Notebooks are evaluated on implementation correctness (40%), experimental methodology and metric visualizations (35%), and concise analytical interpretations answering guided questions (25%).

#### 3. Generative AI & Code Assistant Policy
Modern practitioners use AI coding assistants (e.g. ChatGPT, GitHub Copilot, Claude) for productivity. Their use is permitted under clear academic rules:
- **Student Responsibility:** You are 100% accountable for the correctness and operation of your submitted code. Submitting unverified code with runtime errors or non-functional logic will receive zero credit.
- **Oral Verification (Beszámoló):** Instructors and teaching assistants reserve the right to conduct oral spot-checks during lab hours. You may be asked to explain your logic line by line, modify parameters on the spot, or defend your design choices. Inability to explain submitted work results in zero points for that assignment.
- **Independent In-Class Tests:** Take-home notebooks account for 25% of the grade; the primary evaluative weight (60%) is measured via supervised in-class practical tests where external AI prompting is not permitted.

#### 4. Practical In-Class Evaluations (ZH 1 & ZH 2: 60%)
Two supervised practical milestone examinations (**Zárthelyi dolgozat / ZH**) evaluate independent hands-on implementation and analytical competence:
- **Midterm Practical Test (ZH 1 - 30%):** Held during regular lab hours in Week 7. Evaluates attack generation (FGSM, PGD), poisoning techniques (label flipping, backdoor triggers), and model trojan mechanics.
- **End-Term Practical Test (ZH 2 - 30%):** Held during the scheduled evaluation session in Week 14. Evaluates membership inference, model inversion, synthetic tabular generation, and defense implementations (DP-SGD, federated learning).
- **Environment:** Conducted in the laboratory environment under time constraints without external communication or generative AI assistance.

#### 5. Retake & Substitution Policy (Pótlási lehetőség)
In accordance with ELTE TVSZ Section 52, students are provided an opportunity to improve or make up missed work:
- **Practical Test Retake (Pót-ZH):** Students who fail or miss either ZH 1 or ZH 2 may sit for one comprehensive retake examination during the designated retake period at the end of the semester.
- **Notebook Resubmission:** Students may resubmit or make up up to 2 weekly lab notebooks during the retake window.

---

### University 5-Point Grading Scale

Final aggregate scores map to the official Hungarian 5-point university grading scale (**Gyakorlati jegy**):

| Percentage Range | Grade | Descriptor | Hungarian Term |
|------------------|-------|------------|----------------|
| 85% to 100% | **5** | Excellent | Jeles |
| 70% to 84% | **4** | Good | Jó |
| 55% to 69% | **3** | Satisfactory | Közepes |
| 40% to 54% | **2** | Pass | Elégséges |
| Below 40% | **1** | Fail | Elégtelen |

**Passing Requirements:**
1. Achieve an overall weighted score of at least 40% (Grade 2).
2. Achieve at least 40% on each of the two in-class practical evaluations (ZH 1 and ZH 2), either on the first attempt or during the designated retake (Pót-ZH).
3. Fulfill the mandatory attendance requirement (no more than 3 unexcused absences).

## Getting Started

1. Check the [Schedule]({{ '/schedule/' | prepend: site.baseurl }}) for weekly lab sessions and assignment deadlines.
2. Open the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page to access interactive notebooks and launch them in Google Colab using your Google account.
3. Consult the [Resources]({{ '/resources/' | prepend: site.baseurl }}) page for setup recommendations, background reading, and tools.
