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
