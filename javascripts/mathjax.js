window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", function() {
  var script = document.createElement('script');
  script.src = 'https://cdnjs.cloudflare.com/polyfill.min.js?features=es6';
  document.head.appendChild(script);

  var script2 = document.createElement('script');
  script2.id = 'MathJax-script';
  script2.async = true;
  script2.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  document.head.appendChild(script2);
});
