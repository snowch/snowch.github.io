# CLAUDE.md

Project notes for AI assistants.

## Notebook to HTML Conversion

When converting Jupyter notebooks to HTML, honor hidden cell tags:

```bash
jupyter nbconvert --to html notebook.ipynb --TagRemovePreprocessor.remove_input_tags='["hide-input"]'
```

This hides code cells tagged with `hide-input` while still showing their output (plots, tables, etc.).
