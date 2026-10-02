---
layout: home
---

## Lab Curriculum

All {{ site.data.labs | size }} labs are runnable in Google Colab with no local setup.
Open one from the [Labs]({{ '/labs/' | prepend: site.baseurl }}) page and work through it.

<div class="table-scroll">
<table class="curriculum">
  <caption class="visually-hidden">All {{ site.data.labs | size }} labs, by module</caption>
  <thead>
    <tr><th scope="col">Module</th><th scope="col">Lab</th><th scope="col">Topic</th><th scope="col">Notebook</th></tr>
  </thead>
  <tbody>
  {%- for module in site.data.modules %}
    {%- assign module_labs = site.data.labs | where: "module", module.number -%}
    {%- for lab in module_labs %}
    <tr>
      {%- if forloop.first %}
      <td rowspan="{{ module_labs | size }}"><strong>Module {{ module.number }}</strong><br>{{ module.title }}</td>
      {%- endif %}
      <td>{{ lab.id | replace: "lab-", "" | upcase }}</td>
      <td>{{ lab.title }}</td>
      <td>{% include lab_links.html lab=lab %}</td>
    </tr>
    {%- endfor %}
  {%- endfor %}
  </tbody>
</table>
</div>

## Learning Outcomes

| Skill | Description |
|-------|-------------|
| Understand | Fundamental concepts of machine-learning security and privacy |
| Implement | State-of-the-art attacks (Evasion, Poisoning, Inversion) in PyTorch |
| Evaluate | Model robustness using quantitative metrics and certified bounds |
| Design | Multi-layered defense strategies (DP, FL, Robust Training) for production |
| Generate | Privacy-preserving synthetic data for sensitive domains (healthcare, finance) |

## Getting Started

1. Sign in to your [ELTE account](https://account.elte.hu/) if you do not have one yet.
2. Open the [Schedule]({{ '/schedule/' | prepend: site.baseurl }}) to see when each lab session is held.
3. Work through the labs in order &mdash; each one builds on the previous module.
4. Read the [Materials]({{ '/materials/' | prepend: site.baseurl }}) page for setup notes and background reading.

## References & Acknowledgments

- [unica-mlsec/mlsec](https://github.com/unica-mlsec/mlsec) &mdash; Prof. Battista Biggio (University of Cagliari)
- *Practical Data Privacy* &mdash; Katharine Jarmul (O'Reilly, 2023)
- *Adversarial Machine Learning* &mdash; Goodfellow, Biggio, Laskov (Cambridge University Press, 2018)