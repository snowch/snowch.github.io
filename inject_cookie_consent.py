#!/usr/bin/env python3
"""
Inject cookie consent banner and Google Consent Mode into built HTML files.
This script adds the necessary JavaScript and CSS for GDPR/UK cookie compliance.
"""

import os
from pathlib import Path

# Read the cookie consent files
script_content = Path('cookie-consent.js').read_text()
css_content = Path('cookie-consent.css').read_text()

# Create the injection HTML (to be inserted before </head>)
injection = f"""
<!-- Cookie Consent & Google Consent Mode -->
<style>
{css_content}
</style>
<script>
{script_content}
</script>
"""

def inject_into_html(html_path):
    """Inject cookie consent code into an HTML file."""
    try:
        content = html_path.read_text(encoding='utf-8')

        # Only inject if not already present and if there's a </head> tag
        if 'cookie-consent-banner' not in content and '</head>' in content:
            # Insert before </head> tag to ensure it loads before GA
            modified = content.replace('</head>', f'{injection}\n</head>')
            html_path.write_text(modified, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not process {html_path}: {e}")
        return False

def main():
    """Find all HTML files and inject cookie consent code."""
    build_dir = Path('_build/html')

    if not build_dir.exists():
        print(f"Error: Build directory {build_dir} does not exist")
        return

    # Find all HTML files
    html_files = list(build_dir.rglob('*.html'))

    if not html_files:
        print(f"Warning: No HTML files found in {build_dir}")
        return

    print(f"Found {len(html_files)} HTML files")
    injected_count = 0

    for html_file in html_files:
        if inject_into_html(html_file):
            injected_count += 1

    print(f"Successfully injected cookie consent into {injected_count} files")

if __name__ == '__main__':
    main()
