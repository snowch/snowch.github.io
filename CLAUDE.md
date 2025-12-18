# CLAUDE.md

Project notes for AI assistants.

## GitHub Actions Deployment

The site is deployed via GitHub Actions (`.github/workflows/deploy.yml`). When adding new notebook dependencies:

1. **Add to workflow**: Update the `Install Python dependencies` step in `.github/workflows/deploy.yml`
2. **Current dependencies**: jupyter-book, mystmd, numpy, matplotlib, matplotlib-venn, scipy, wavedrom, graphviz

**Example:** If a new notebook requires `pandas`, add it to the pip install line:
```yaml
- name: Install Python dependencies
  run: pip install jupyter-book mystmd numpy matplotlib matplotlib-venn scipy pandas wavedrom graphviz
```

## Matplotlib LaTeX Rendering

Matplotlib supports LaTeX rendering for mathematical equations and symbols. Enable it with:

```python
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
```

**When to use:**
- Mathematical equations in plots (e.g., $D_1 = Q_0 \cdot X$)
- Technical notation with subscripts, superscripts, and special symbols
- Professional-looking typography in diagrams

**Reference:** https://matplotlib.org/stable/users/explain/text/usetex.html

**Note:** Requires LaTeX installation on the system. The GitHub Actions workflow has LaTeX available (`texlive-latex-base` and `texlive-latex-extra` are installed) for building Jupyter Books with matplotlib LaTeX rendering.
