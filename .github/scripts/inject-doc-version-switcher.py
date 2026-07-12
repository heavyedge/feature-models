#!/usr/bin/env python3
import argparse
from pathlib import Path

MARKER = "version-switcher.js"


def asset_path(html_file, site_dir, asset_name):
    depth = len(html_file.relative_to(site_dir).parent.parts)
    return Path(*([".."] * depth), "_static", asset_name).as_posix()


def inject_html(html_file, site_dir):
    html = html_file.read_text(encoding="utf-8")
    if MARKER in html:
        return False

    head_end = html.lower().rfind("</head>")
    if head_end == -1:
        return False

    css_href = asset_path(html_file, site_dir, "version-switcher.css")
    js_src = asset_path(html_file, site_dir, "version-switcher.js")
    snippet = (
        f'    <link rel="stylesheet" href="{css_href}">\n'
        f'    <script defer src="{js_src}"></script>\n'
    )
    html_file.write_text(html[:head_end] + snippet + html[head_end:], encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", required=True)
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    injected = 0
    for html_file in site_dir.rglob("*.html"):
        if any(part == ".git" for part in html_file.relative_to(site_dir).parts):
            continue
        if inject_html(html_file, site_dir):
            injected += 1

    print(f"Injected documentation version switcher into {injected} HTML files.")


if __name__ == "__main__":
    main()
