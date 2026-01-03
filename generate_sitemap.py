#!/usr/bin/env python3
"""
Generate sitemap index, main sitemap, and robots.txt for the site.
This script should be run after `myst build` completes.

For sites with project pages (e.g., snowch.github.io/project-name):
- Creates sitemap-main.xml for the main site
- Creates sitemap_index.xml that references main + project sitemaps
- robots.txt points to sitemap_index.xml
"""
import os
import yaml
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

def load_myst_config(config_path='myst.yml'):
    """Load the MyST configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_pages_from_toc(toc, pages=None):
    """Recursively extract all page files from the table of contents."""
    if pages is None:
        pages = []

    for item in toc:
        if isinstance(item, dict):
            # Add the file if it exists
            if 'file' in item:
                pages.append(item['file'])
            # Recursively process children
            if 'children' in item:
                extract_pages_from_toc(item['children'], pages)

    return pages

def file_to_url(file_path, base_url):
    """Convert a file path to a URL."""
    # Remove .md or .ipynb extension and add .html
    if file_path.endswith('.md'):
        url_path = file_path[:-3] + '.html'
    elif file_path.endswith('.ipynb'):
        url_path = file_path[:-6] + '.html'
    else:
        url_path = file_path

    # Handle index.html specially
    if url_path == 'index.html':
        return base_url.rstrip('/') + '/'

    return urljoin(base_url, url_path)

def get_file_lastmod(file_path, build_dir='_build/html'):
    """Get last modification time of the built HTML file."""
    # Convert source path to built HTML path
    html_path = file_path.replace('.md', '.html').replace('.ipynb', '.html')
    full_path = os.path.join(build_dir, html_path)

    if os.path.exists(full_path):
        mtime = os.path.getmtime(full_path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

    # Fallback to current date if file doesn't exist
    return datetime.now().strftime('%Y-%m-%d')

def generate_main_sitemap(config, build_dir='_build/html', base_url='https://snowch.github.io/'):
    """Generate sitemap for main site pages."""
    pages = []

    # Extract pages from project TOC
    if 'project' in config and 'toc' in config['project']:
        pages = extract_pages_from_toc(config['project']['toc'])

    # Start building the sitemap XML
    sitemap_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    sitemap_lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Add each page to the sitemap
    for page in pages:
        url = file_to_url(page, base_url)
        lastmod = get_file_lastmod(page, build_dir)

        sitemap_lines.append('  <url>')
        sitemap_lines.append(f'    <loc>{url}</loc>')
        sitemap_lines.append(f'    <lastmod>{lastmod}</lastmod>')
        sitemap_lines.append('    <changefreq>monthly</changefreq>')
        sitemap_lines.append('    <priority>0.8</priority>')
        sitemap_lines.append('  </url>')

    sitemap_lines.append('</urlset>')

    return '\n'.join(sitemap_lines)

def generate_sitemap_index(config, base_url='https://snowch.github.io/'):
    """Generate sitemap index that references main + project sitemaps."""
    lastmod = datetime.now().strftime('%Y-%m-%d')

    index_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    index_lines.append('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Add main sitemap
    index_lines.append('  <sitemap>')
    index_lines.append(f'    <loc>{urljoin(base_url, "sitemap-main.xml")}</loc>')
    index_lines.append(f'    <lastmod>{lastmod}</lastmod>')
    index_lines.append('  </sitemap>')

    # Add project site sitemaps if configured
    project_sites = config.get('project', {}).get('project_sites', [])
    for project in project_sites:
        project_sitemap_url = urljoin(base_url, f'{project}/sitemap.xml')
        index_lines.append('  <sitemap>')
        index_lines.append(f'    <loc>{project_sitemap_url}</loc>')
        index_lines.append('  </sitemap>')

    index_lines.append('</sitemapindex>')

    return '\n'.join(index_lines)

def generate_robots_txt(base_url='https://snowch.github.io/', has_project_sites=False):
    """Generate robots.txt file."""
    # Point to sitemap index if there are project sites, otherwise to main sitemap
    if has_project_sites:
        sitemap_url = urljoin(base_url, 'sitemap_index.xml')
    else:
        sitemap_url = urljoin(base_url, 'sitemap-main.xml')

    robots_lines = [
        '# robots.txt for snowch.github.io',
        '',
        'User-agent: *',
        'Allow: /',
        '',
        f'Sitemap: {sitemap_url}',
        ''
    ]

    return '\n'.join(robots_lines)

def main():
    """Main function to generate sitemap files and robots.txt."""
    # Load MyST configuration
    config = load_myst_config()

    # Determine base URL
    base_url = 'https://snowch.github.io/'

    # Build directory
    build_dir = '_build/html'

    # Ensure build directory exists
    if not os.path.exists(build_dir):
        print(f"Warning: Build directory {build_dir} does not exist.")
        print("Creating directory and proceeding...")
        os.makedirs(build_dir, exist_ok=True)

    # Generate main sitemap
    main_sitemap_xml = generate_main_sitemap(config, build_dir, base_url)
    main_sitemap_path = os.path.join(build_dir, 'sitemap-main.xml')

    with open(main_sitemap_path, 'w') as f:
        f.write(main_sitemap_xml)

    print(f"✓ Generated {main_sitemap_path}")

    # Check if there are project sites
    project_sites = config.get('project', {}).get('project_sites', [])
    has_project_sites = len(project_sites) > 0

    # Generate sitemap index if there are project sites
    if has_project_sites:
        sitemap_index_xml = generate_sitemap_index(config, base_url)
        index_path = os.path.join(build_dir, 'sitemap_index.xml')

        with open(index_path, 'w') as f:
            f.write(sitemap_index_xml)

        print(f"✓ Generated {index_path}")
        print(f"  References: sitemap-main.xml + {len(project_sites)} project site(s)")
        for project in project_sites:
            print(f"    - {project}/sitemap.xml")

    # Generate robots.txt
    robots_txt = generate_robots_txt(base_url, has_project_sites)
    robots_path = os.path.join(build_dir, 'robots.txt')

    with open(robots_path, 'w') as f:
        f.write(robots_txt)

    print(f"✓ Generated {robots_path}")

    # Print summary
    print(f"\nMain sitemap contains {main_sitemap_xml.count('<url>')} URLs")
    print(f"Site: {base_url}")

if __name__ == '__main__':
    main()
