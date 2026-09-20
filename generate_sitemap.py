#!/usr/bin/env python3
"""
Generate sitemap index, main sitemap, and robots.txt for the site.
This script should be run after `myst build` completes.

For sites with project pages (e.g., snowch.github.io/project-name):
- Creates sitemap-main.xml for the main site
- Creates sitemap_index.xml that references main + project sitemaps
- robots.txt points to sitemap_index.xml

URLs come from the built HTML tree, not from the source TOC. MyST publishes
each page under its own slug -- probability/inclusion_exclusion_tutorial.md is
served at /inclusion-exclusion-tutorial/ -- so source paths do not predict
published URLs.
"""
import json
import os
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

def load_myst_config(config_path='myst.yml'):
    """Load the MyST configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def discover_built_pages(build_dir='_build/html'):
    """Find every page the build actually published.

    MyST writes one directory per page, each holding an index.html, so a
    directory containing an index.html is a page and anything else (theme
    assets, static files) is not. Returns routes sorted, with '' for the
    home page.
    """
    build_path = Path(build_dir)

    if not build_path.is_dir():
        return []

    routes = []
    for index_file in build_path.rglob('index.html'):
        route = index_file.parent.relative_to(build_path).as_posix()
        routes.append('' if route == '.' else route)

    return sorted(routes)

def page_to_url(route, base_url):
    """Convert a built route to its published URL."""
    base = base_url.rstrip('/') + '/'
    return base if not route else f'{base}{route}/'

def get_page_source(route, build_dir):
    """Map a built route back to the file it was written from.

    The build writes a JSON payload beside each page recording its source
    location, which is the only link back from a slug to its markdown.
    """
    payload = Path(build_dir) / ((route or 'index') + '.json')

    try:
        location = json.loads(payload.read_text()).get('location')
    except (OSError, ValueError):
        return None

    if not location:
        return None

    source = Path(location.lstrip('/'))
    return source if source.exists() else None

def git_last_modified(source):
    """Date of the last commit touching a file, or None if git cannot say."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cs', '--', str(source)],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None

def is_shallow_clone():
    """Whether git history is truncated, which hides real modification dates."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-shallow-repository'],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False

    return result.stdout.strip() == 'true'

def get_lastmod(route, build_dir, trust_git_dates):
    """Get the date a page last changed, or None if there is no honest answer.

    <lastmod> is optional, and a wrong one is worse than none. Built files are
    all stamped at build time, and a shallow clone reports the boundary commit
    for anything it did not fetch, so only a full history answers this.
    """
    if not trust_git_dates:
        return None

    source = get_page_source(route, build_dir)
    if not source:
        return None

    return git_last_modified(source)

def generate_main_sitemap(routes, build_dir='_build/html', base_url='https://snowch.github.io/',
                          trust_git_dates=True):
    """Generate sitemap for main site pages."""
    # Start building the sitemap XML
    sitemap_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    sitemap_lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Add each page to the sitemap
    for route in routes:
        lastmod = get_lastmod(route, build_dir, trust_git_dates)

        sitemap_lines.append('  <url>')
        sitemap_lines.append(f'    <loc>{page_to_url(route, base_url)}</loc>')
        if lastmod:
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

    # Without a build there are no URLs to publish. Fail rather than write an
    # empty sitemap over a good one.
    routes = discover_built_pages(build_dir)
    if not routes:
        print(f"Error: no built pages found in {build_dir}.")
        print("Run `myst build --html` before this script.")
        sys.exit(1)

    # A shallow clone dates unchanged pages to the boundary commit, so omit
    # <lastmod> entirely rather than publish dates that look real and are not.
    trust_git_dates = not is_shallow_clone()
    if not trust_git_dates:
        print("Warning: shallow git clone -- omitting <lastmod>.")
        print("Set fetch-depth: 0 on actions/checkout for real modification dates.")

    # Generate main sitemap
    main_sitemap_xml = generate_main_sitemap(routes, build_dir, base_url,
                                             trust_git_dates)
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

    # Remove MyST's own sitemap.xml: it is built with the dev server's host,
    # so its URLs point at localhost rather than the published site.
    old_sitemap = os.path.join(build_dir, 'sitemap.xml')
    if os.path.exists(old_sitemap):
        os.remove(old_sitemap)
        print(f"✓ Removed old {old_sitemap}")

    # Print summary
    print(f"\nMain sitemap contains {main_sitemap_xml.count('<url>')} URLs")
    print(f"Site: {base_url}")

if __name__ == '__main__':
    main()
