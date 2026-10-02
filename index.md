---
layout: home
---

## Prerequisites

Students should be comfortable with:
- **Programming:** Python and basic PyTorch (tensors, neural network layers, custom training loops, and data loaders).
- **Machine Learning Foundations:** Supervised learning, loss functions, optimization via gradient descent, and standard classification metrics.
- **Compute:** No dedicated local hardware or GPU is required. All labs run directly in Google Colab.

## Course Curriculum

The curriculum spans 8 core modules across the 14-week practicum. Select any lab topic to jump directly to its technical objectives and launch the notebook on the [**Labs**]({{ '/labs/' | prepend: site.baseurl }}) page:

<div class="table-scroll">
<table class="curriculum">
  <caption class="visually-hidden">All {{ site.data.labs | size }} labs, by module</caption>
  <thead>
    <tr>
      <th scope="col">Module</th>
      <th scope="col">Lab</th>
      <th scope="col">Topic</th>
      <th scope="col">Session Date</th>
    </tr>
  </thead>
  <tbody>
  {%- for module in site.data.modules %}
    {%- assign module_labs = site.data.labs | where: "module", module.number -%}
    {%- for lab in module_labs %}
    <tr>
      {%- if forloop.first %}
      <td class="module-cell" rowspan="{{ module_labs | size }}">
        <strong>Module {{ module.number }}</strong><br>
        <span class="module-title-sub">{{ module.title }}</span>
      </td>
      {%- endif %}
      <td class="lab-id-cell">{{ lab.id | replace: "lab-", "" | upcase }}</td>
      <td><a href="{{ '/labs/' | prepend: site.baseurl }}#{{ lab.id }}">{{ lab.title }}</a></td>
      <td class="lab-date-cell">{{ lab.date | date: site.dateformat }}</td>
    </tr>
    {%- endfor %}
  {%- endfor %}
  </tbody>
</table>
</div>

## Learning Outcomes

By the end of this course, students will be able to:

- **Understand** fundamental security and privacy threat models across the machine learning lifecycle.
- **Implement** state-of-the-art adversarial attacks against deep neural networks in PyTorch.
- **Evaluate** model vulnerability using quantitative empirical metrics and certified robustness bounds.
- **Design & Deploy** privacy-preserving and robust defenses, including Differential Privacy (DP-SGD), adversarial training, and federated learning protocols.
- **Generate** privacy-preserving synthetic tabular datasets for sensitive domain applications.

## Grading & Evaluation

Coursework is evaluated continuously throughout the 14-week practicum (**Gyakorlati jegy**). To ensure independent mastery in an AI-assisted environment, the majority of the grade is determined by supervised in-class practical milestones:

| Component | Weight | Assessment | Schedule & Scope |
|-----------|--------|------------|------------------|
| **Attendance & Participation** | **15%** | Weekly in-lab presence | Mandatory attendance (max 3 absences per TVSZ) |
| **Weekly Lab Notebooks** | **25%** | Colab PyTorch assignments | Continuous practice, submitted weekly by 23:59 |
| **Midterm Practical Test (ZH 1)** | **30%** | In-class supervised test | Week 7 (Modules 1 to 4: Evasion & Poisoning) |
| **End-Term Practical Test (ZH 2)** | **30%** | In-class supervised test | Week 14 (Modules 5 to 8: Privacy & Defenses) |

### Grading Scale

| Grade | Percentage | Descriptor (HU) | Criteria |
|-------|------------|-----------------|----------|
| **5** | 85% to 100% | Excellent (*Jeles*) | Comprehensive mastery of attack mechanics and defenses |
| **4** | 70% to 84% | Good (*Jó*) | Solid practical implementation and analytical interpretation |
| **3** | 55% to 69% | Satisfactory (*Közepes*) | Functional implementations with minor conceptual gaps |
| **2** | 40% to 54% | Pass (*Elégséges*) | Reaches minimum passing threshold across evaluations |
| **1** | Below 40% | Fail (*Elégtelen*) | Below passing threshold or exceeded absence limit |

### Core Policies
- **Passing Threshold:** Requires at least 40% (Grade 2) overall and at least 40% on each of the two in-class tests (ZH 1 and ZH 2).
- **Retake Policy (Pótlás):** In accordance with ELTE TVSZ, students may sit for one comprehensive retake examination (Pót-ZH) at semester end to make up or improve a test score.
- **Attendance Limit:** In accordance with ELTE TVSZ, missing more than 3 lab sessions results in refusal of the practical mark.
- **AI & Oral Verification:** AI tools are permitted as coding aids, but students must be able to explain all submitted code. Instructors conduct oral spot-checks during lab hours. Inability to explain code results in zero credit.

## Getting Started

1. Check the [Schedule]({{ '/schedule/' | prepend: site.baseurl }}) for weekly lab sessions and assignment deadlines.
2. Open the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page to access interactive notebooks and launch them in Google Colab using your Google account.
3. Consult the [Resources]({{ '/resources/' | prepend: site.baseurl }}) page for setup recommendations, background reading, and tools.
