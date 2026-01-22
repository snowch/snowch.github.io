#!/bin/bash
set -e

echo "Creating observability-anomaly-detection repository..."

# Change to your projects directory (adjust as needed)
cd ~/projects 2>/dev/null || cd ~

# Create directory structure
mkdir -p observability-anomaly-detection
cd observability-anomaly-detection

echo "Creating directory structure..."
mkdir -p .github/workflows
mkdir -p appendix-code/{config,logs,scripts,services/{auth-service,load-generator,payment-worker,web-api}}
mkdir -p data
mkdir -p notebooks

echo "Repository structure created at: $(pwd)"
echo ""
echo "Next steps:"
echo "1. Copy files from your snowch.github.io clone:"
echo "   cp -r /path/to/snowch.github.io/ai-eng/embedding-anomaly-detection/* ."
echo ""
echo "2. Add repo-specific files (I'll create these for you next)..."

# Create .gitignore
cat > .gitignore << 'EOF'
# MyST build artifacts
_build/
.myst.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data (large files)
data/*.parquet
data/*.zip
data/*.npy
data/*.pkl
data/*.pt

# Keep data directory but ignore large files
!data/.gitkeep
!data/README.md
EOF

echo "✓ Created .gitignore"

# Create GitHub Actions workflow
cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy MyST site to GitHub Pages

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install MyST
        run: npm install -g mystmd

      - name: Build MyST site
        run: myst build --html

      - name: Upload artifact
        if: github.event_name != 'pull_request'
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./_build/html

  deploy:
    if: github.event_name != 'pull_request'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
EOF

echo "✓ Created .github/workflows/deploy.yml"

# Create README.md
cat > README.md << 'EOF'
# Embedding-Based Anomaly Detection for Observability

A comprehensive tutorial series on building production-ready anomaly detection systems using ResNet embeddings for OCSF (Open Cybersecurity Schema Framework) observability data.

## 📖 Read the Tutorial

👉 **[Start the tutorial series](https://snowch.github.io/observability-anomaly-detection/)**

## What You'll Learn

How to build, train, and deploy a **custom embedding model** (TabularResNet) specifically designed for OCSF observability data:

- Build and train TabularResNet using self-supervised learning on unlabeled logs
- Deploy the model as a FastAPI service for real-time inference
- Store embeddings in a vector database for fast k-NN similarity search
- Detect anomalies through vector operations (no separate detection model needed)
- Monitor embedding quality and trigger automated retraining

## Tutorial Series

1. **Part 1**: Understanding ResNet Architecture
2. **Part 2**: Adapting ResNet for Tabular Data
3. **Part 3**: Feature Engineering for OCSF Data
4. **Part 4**: Self-Supervised Training
5. **Part 5**: Evaluating Embedding Quality
6. **Part 6**: Anomaly Detection Methods
7. **Part 7**: Production Deployment
8. **Part 8**: Production Monitoring
9. **Part 9**: Multi-Source Event Correlation

**Plus**: Hands-on Jupyter notebooks and sample data

## Who This Is For

- ML engineers building anomaly detection systems
- Security engineers working with observability data
- Data scientists interested in self-supervised learning
- Anyone wanting to apply ResNet to tabular/observability data

## Prerequisites

- Basic Python and PyTorch
- Understanding of neural networks (or see our [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) series)

## Quick Start

### Run the Hands-on Notebooks

1. Install dependencies:
```bash
pip install pandas numpy torch scikit-learn matplotlib pyarrow
```

2. Download sample data and notebooks from the [Appendix](https://snowch.github.io/observability-anomaly-detection/appendix-notebooks)

3. Run the notebooks:
   - `03-feature-engineering.md` - Extract features from OCSF data
   - `04-self-supervised-training.md` - Train TabularResNet
   - `05-embedding-evaluation.md` - Evaluate embedding quality
   - `06-anomaly-detection.md` - Detect anomalies

### Generate Your Own Data

Use the [Docker Compose stack](https://snowch.github.io/observability-anomaly-detection/appendix-generating-training-data) to generate realistic OCSF observability data with labeled anomalies.

## Key Features

✅ **Production-ready code** - All examples are deployable
✅ **No labels required** - Self-supervised learning on unlabeled data
✅ **Hands-on notebooks** - Working code you can run immediately
✅ **Sample data included** - Pre-generated OCSF events
✅ **Complete MLOps** - Deployment, monitoring, retraining

## Applicability Beyond OCSF

While this series uses OCSF security logs as the running example, the TabularResNet approach applies to **any structured observability data**:

- **Telemetry/Metrics**: CPU%, memory, latency with metadata
- **Configuration data**: Key-value pairs, settings
- **Distributed traces**: Span attributes
- **Application logs**: JSON logs, syslog

**The key requirement**: Your data can be represented as rows with categorical and numerical features.

## License

This tutorial series is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Chris Snow** - [snowch.github.io](https://snowch.github.io)

## Related Resources

- [Embeddings at Scale Book](https://snowch.github.io/embeddings-at-scale-book/) - Deep dive into production embedding systems
- [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) - Learn NN fundamentals

## Contributing

Found an issue or have suggestions? [Open an issue](https://github.com/snowch/observability-anomaly-detection/issues) or submit a pull request!
EOF

echo "✓ Created README.md"

# Create myst.yml
cat > myst.yml << 'EOF'
# See docs at: https://mystmd.org/guide/frontmatter
version: 1
project:
  id: observability-anomaly-detection
  title: Embedding-Based Anomaly Detection for Observability
  description: A comprehensive tutorial series on building production-ready anomaly detection systems using ResNet embeddings for OCSF observability data
  keywords:
    - anomaly detection
    - embeddings
    - ResNet
    - OCSF
    - observability
    - machine learning
    - deep learning
    - self-supervised learning
  authors:
    - name: Chris Snow
      url: https://snowch.github.io
  github: https://github.com/snowch/observability-anomaly-detection
  license: CC-BY-4.0
  bibliography:
    - references.bib
  toc:
    - file: index.md
      title: Series Introduction
    - file: part1-understanding-resnet.md
      title: "Part 1: Understanding ResNet Architecture"
    - file: part2-tabular-resnet.md
      title: "Part 2: Adapting ResNet for Tabular Data"
    - file: part3-feature-engineering.md
      title: "Part 3: Feature Engineering for OCSF Data"
    - file: part4-self-supervised-training.md
      title: "Part 4: Self-Supervised Training"
    - file: part5-embedding-quality.md
      title: "Part 5: Evaluating Embedding Quality"
    - file: part6-anomaly-detection.md
      title: "Part 6: Anomaly Detection Methods"
    - file: part7-production-deployment.md
      title: "Part 7: Production Deployment"
    - file: part8-production-monitoring.md
      title: "Part 8: Production Monitoring"
    - file: part9-multi-source-correlation.md
      title: "Part 9: Multi-Source Event Correlation"
    - file: appendix-generating-training-data.md
      title: "Appendix: Generating Training Data"
    - file: appendix-notebooks.md
      title: "Appendix: Notebooks & Sample Data"
    - file: notebooks/03-feature-engineering.md
      title: "Notebook: Feature Engineering"
    - file: notebooks/04-self-supervised-training.md
      title: "Notebook: Self-Supervised Training"
    - file: notebooks/05-embedding-evaluation.md
      title: "Notebook: Embedding Evaluation"
    - file: notebooks/05-model-inference.md
      title: "Notebook: Model Inference"
    - file: notebooks/06-anomaly-detection.md
      title: "Notebook: Anomaly Detection"

site:
  template: book-theme
  title: Observability Anomaly Detection
  options:
    logo_text: Observability Anomaly Detection
  nav:
    - title: GitHub
      url: https://github.com/snowch/observability-anomaly-detection
    - title: Author
      url: https://snowch.github.io
EOF

echo "✓ Created myst.yml"

# Create data README
cat > data/README.md << 'EOF'
# Data Directory

This directory contains sample data for the tutorial notebooks.

Data files (*.npy, *.parquet, *.pt, *.pkl) are not stored in git due to size.

Download sample data from the [Appendix: Notebooks & Sample Data](../appendix-notebooks.md).
EOF

echo "✓ Created data/README.md"

echo ""
echo "================================================================"
echo "Repository skeleton created!"
echo "================================================================"
echo ""
echo "Location: $(pwd)"
echo ""
echo "Now copy the content files from your snowch.github.io clone:"
echo ""
echo "  SOURCE=/path/to/snowch.github.io/ai-eng/embedding-anomaly-detection"
echo "  cp \$SOURCE/*.md ."
echo "  cp \$SOURCE/references.bib ."
echo "  cp -r \$SOURCE/notebooks/*.md notebooks/"
echo "  cp -r \$SOURCE/appendix-code/* appendix-code/"
echo ""
echo "Then initialize git and push:"
echo ""
echo "  git init"
echo "  git branch -m main"
echo "  git add -A"
echo "  git commit -m 'Initial commit: Migrate embedding anomaly detection tutorial series'"
echo "  git remote add origin https://github.com/snowch/observability-anomaly-detection.git"
echo "  git push -u origin main"
echo ""
echo "================================================================"
EOF
