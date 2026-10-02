---
layout: labs
title: Labs
permalink: /labs/
---
Every lab is a Jupyter notebook that runs directly in Google Colab with no local
installation required. Press **Open in Colab** to get started; Colab opens the
notebook directly from this repository and saves your progress to your own Drive.

The notebooks reference images and helper modules stored next to them in this
repository, so keep the whole repository intact if you download a copy.

Some labs need a GPU. Colab offers one automatically; if a cell fails with an
out-of-memory error, go to *Runtime &rarr; Change runtime type &rarr; T4 GPU*.

## Curriculum Overview

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
      <td><a href="#{{ lab.id }}">{{ lab.title }}</a></td>
      <td>{% include lab_links.html lab=lab %}</td>
    </tr>
    {%- endfor %}
  {%- endfor %}
  </tbody>
</table>
</div>

## Detailed Modules
