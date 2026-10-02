---
layout: labs
title: Labs
permalink: /labs/
---
All labs run directly in Google Colab with no local setup required. Use the index below to jump to any module, or browse the sections below to review lab objectives and launch notebooks. For GPU acceleration and local environment setup, consult the [Resources]({{ '/resources/' | prepend: site.baseurl }}) page.

## Quick Navigation

<div class="table-scroll">
<table class="curriculum">
  <caption class="visually-hidden">All {{ site.data.labs | size }} labs, by module</caption>
  <thead>
    <tr><th scope="col">Module</th><th scope="col">Lab</th><th scope="col">Topic</th><th scope="col">Session Date</th></tr>
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
      <td>{{ lab.date | date: site.dateformat }}</td>
    </tr>
    {%- endfor %}
  {%- endfor %}
  </tbody>
</table>
</div>

## Detailed Modules
