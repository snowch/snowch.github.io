# CLAUDE.md

Project notes for AI assistants.

## Notebook to HTML Conversion

When converting Jupyter notebooks to HTML, honor hidden cell tags:

```bash
jupyter nbconvert --to html notebook.ipynb --TagRemovePreprocessor.remove_input_tags='["hide-input"]'
```

This hides code cells tagged with `hide-input` while still showing their output (plots, tables, etc.).

**Workflow:** When editing a notebook (`.ipynb`) for a blog post, always regenerate and commit the corresponding HTML file:

1. Edit the notebook
2. Run `jupyter nbconvert --to html <notebook>.ipynb --TagRemovePreprocessor.remove_input_tags='["hide-input"]'`
3. Commit both the `.ipynb` and `.html` files

If jupyter is not installed, install it first:

```bash
pip install jupyter nbconvert
```

This installs the tools needed to convert notebooks to HTML while honoring cell tags like `hide-input`.
