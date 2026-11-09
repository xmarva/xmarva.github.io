#!/usr/bin/env python3
"""Tell IndexNow (Bing, Yandex and the other engines that share it) which pages a deploy added or changed.

A page is sent when its address is new to the sitemap, or when the push changed its source: a post, the post's
assets, a project or a page in _pages. Edits to layouts and data files touch every page and are not sent. With --all
every address in the sitemap is sent, which is meant for the first run.

The deploy saves the live sitemap before it builds and runs this script after it publishes. The script waits until
GitHub Pages serves the new build of every page it is about to send, so the engines do not fetch the old one.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

ENDPOINT = "https://api.indexnow.org/indexnow"
SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
ROOT = Path(__file__).resolve().parents[1]


def sitemap_urls(path):
    text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if not text:
        return None
    return [loc.text.strip() for loc in ET.fromstring(text).iterfind("s:url/s:loc", SITEMAP_NS)]


def read_key():
    for path in ROOT.glob("*.txt"):
        if re.fullmatch(r"[0-9a-f]{32}", path.stem) and path.read_text().strip() == path.stem:
            return path.stem
    sys.exit("no IndexNow key file (<32 hex digits>.txt holding its own name) in the site root")


def changed_paths(changed, urls):
    """The sitemap addresses whose source files the push changed."""
    by_path = {urlparse(u).path: u for u in urls}
    found = set()
    for name in changed:
        post = re.fullmatch(r"_posts/\d{4}-\d{2}-\d{2}-(.+)\.\w+", name) or re.fullmatch(r"assets/posts/([^/]+)/.+", name)
        project = re.fullmatch(r"_projects/(.+)\.\w+", name)
        if post:
            found.update(u for p, u in by_path.items() if p.startswith("/blog/") and p.endswith(f"/{post[1]}/"))
        elif project:
            found.update(u for p, u in by_path.items() if p == f"/projects/{project[1]}/")
        elif re.fullmatch(r"_pages/.+\.\w+", name) and (ROOT / name).is_file():
            head = (ROOT / name).read_text(encoding="utf-8").split("\n---", 1)[0]
            if m := re.search(r"^permalink:\s*(\S+)", head, re.MULTILINE):
                found.update(u for p, u in by_path.items() if p == m[1])
    return found


def built_file(site, url):
    path = urlparse(url).path
    return site / (path.lstrip("/") + ("index.html" if path.endswith("/") else ""))


def live_bytes(url):
    try:
        with urllib.request.urlopen(f"{url}?indexnow={time.time_ns()}", timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError):
        return None


def wait_until_live(site, urls, timeout):
    """Wait until GitHub Pages serves the built bytes of every address; return the ones still stale."""
    built = {u: built_file(site, u) for u in urls}
    waiting = {u for u, path in built.items() if path.is_file()}
    deadline = time.monotonic() + timeout
    while waiting and time.monotonic() < deadline:
        waiting = {u for u in waiting if live_bytes(u) != built[u].read_bytes()}
        if waiting:
            time.sleep(20)
    return waiting


def submit(host, key, urls):
    body = json.dumps({"host": host, "key": key, "urlList": sorted(urls)}).encode()
    request = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--site", type=Path, default=ROOT / "_site", help="the built site")
    parser.add_argument("--old", type=Path, help="the sitemap that was live before this deploy")
    parser.add_argument("--changed", type=Path, help="the files the push changed, one per line")
    parser.add_argument("--all", action="store_true", help="send every address in the sitemap")
    parser.add_argument("--wait", type=int, default=900, help="seconds to wait for GitHub Pages")
    parser.add_argument("--dry-run", action="store_true", help="print the addresses and stop")
    args = parser.parse_args()

    urls = sitemap_urls(args.site / "sitemap.xml")
    if not urls:
        sys.exit(f"no sitemap in {args.site}")
    if args.all:
        send = set(urls)
    else:
        old = sitemap_urls(args.old) if args.old else None
        if old is None:
            print("the old sitemap is missing: sending only pages whose sources changed")
        send = set(urls) - set(old or urls)
        changed = args.changed.read_text().split() if args.changed and args.changed.is_file() else []
        send |= changed_paths(changed, urls)
    if not send:
        print("no new or changed pages")
        return
    print(f"{len(send)} pages:", *sorted(send), sep="\n  ")
    if args.dry_run:
        return

    key = read_key()
    host = urlparse(urls[0]).netloc
    stale = wait_until_live(args.site, send | {f"https://{host}/{key}.txt"}, args.wait)
    if stale:
        print("still not live, sent anyway:", *sorted(stale), sep="\n  ")
    status, text = submit(host, key, send)
    print(f"IndexNow answered {status} {text}".strip())
    if status >= 400:
        sys.exit(1)


if __name__ == "__main__":
    main()
