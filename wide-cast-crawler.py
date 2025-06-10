#!/usr/bin/env python3
"""
Wide-Net Web Spider (Perl → Python 3 rewrite)
─────────────────────────────────────────────
• Starts from URLs in seeds.txt
• Respects robots.txt (configurable)
• Extracts <title>, <meta description>, <meta keywords>
  - If meta tags are missing, builds a summary & autokeywords
• “Wide-cast net” keyword expansion:
  adds WordNet synonyms so related searches hit the same page
• Polite: configurable delay, size limits, max iterations
• Writes tab-separated DB files spider_#.dat (URL,KEYWORDS,DESCRIPTION,TITLE)
• Generates helper files: keywords.dat, urls.dat, descriptions.dat, title.dat
"""

import re
import csv
import sys
import time
import queue
import json
import html
import hashlib
import urllib.parse as urlparse
from pathlib import Path
from collections import Counter, deque
from typing import List, Set

import requests
from bs4 import BeautifulSoup, Comment
import tldextract
import nltk
from nltk.corpus import stopwords, wordnet as wn
from urllib import robotparser

# ──────────────────────────────────────────────
# CONFIG – tweak as needed
# ──────────────────────────────────────────────
SEEDS_FILE               = "seeds.txt"
DB_TEMPLATE              = "spider_#.dat"           # # → 1,2,3…
DB_LAYOUT                = "{url}\t{keywords}\t{desc}\t{title}\n"
LINES_PER_FILE           = 50_000                   # split files for huge crawls
MAX_DB_SIZE              = 100_000                  # total records
MAX_ITERATIONS           = 4
FETCH_TIMEOUT            = 5                        # seconds
USER_AGENT               = "WideNet Crawler (Mozilla compatible)"
MAX_REDIRECTS            = 7
FETCH_PAUSE              = 1.5                      # polite delay, seconds
PAGE_MAX_SIZE            = 200_000                  # bytes
CONTENT_TYPES            = ("text/html", "text/plain")
INCLUDE_URL_RX           = re.compile(
    r"^(https?://)"
    r"[a-z0-9\-\.]+\.[a-z]{2,5}"
    r"(:[0-9]{1,5})?(/.*)?$",
    re.IGNORECASE,
)
EXCLUDE_URL_RX           = re.compile(r"\.(jpg|jpeg|png|gif|pdf|zip|exe)$", re.I)
RESPECT_ROBOTS           = True

MAX_DESCRIPTION_LEN      = 197
MAX_KEYWORDS_LEN         = 800
GENERATE_KEYWORDS        = True
WIDEN_KEYWORDS           = True                    # add WordNet synonyms
MIN_WORD_LEN             = 5
STOPWORDS                = set(stopwords.words("english"))

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def normalize_url(url: str) -> str:
    if url.endswith("/"):
        return url
    parsed = urlparse.urlparse(url)
    if parsed.path == "":
        return f"{url}/"
    return url

def should_process_url(url: str) -> bool:
    if EXCLUDE_URL_RX.search(url):
        return False
    return bool(INCLUDE_URL_RX.match(url))

def is_content_type_ok(ct_header: str) -> bool:
    return any(ct_header.startswith(ok) for ok in CONTENT_TYPES)

def polite_get(url: str, session: requests.Session) -> requests.Response | None:
    try:
        resp = session.get(
            url,
            timeout=FETCH_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            stream=True,    # so we can enforce max length
        )
        # Hard size cap
        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > PAGE_MAX_SIZE:
                raise ValueError("Page exceeds max size")
        resp._content = content  # type: ignore
        return resp
    except Exception as exc:
        print(f"      ! fetch error: {exc}")
        return None

def strip_html(page: str) -> str:
    soup = BeautifulSoup(page, "html.parser")
    # remove scripts, styles, comments
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return html.unescape(text)

def extract_meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    return tag["content"].strip() if tag and tag.has_attr("content") else ""

def generate_summary(text: str, max_len: int = MAX_DESCRIPTION_LEN) -> str:
    return text[:max_len].rstrip()

def autokeywords(text: str, max_len: int = MAX_KEYWORDS_LEN) -> str:
    words = [
        w.lower() for w in re.findall(r"[a-zA-Z0-9@\-]{%d,}" % MIN_WORD_LEN, text)
        if w.lower() not in STOPWORDS
    ]
    freq = Counter(words)
    ranked = [w for w, _ in freq.most_common()]
    if WIDEN_KEYWORDS:
        ranked = widen_with_synonyms(ranked)
    kw = ", ".join(ranked)
    return kw[:max_len]

def widen_with_synonyms(words: List[str]) -> List[str]:
    """Add WordNet synonyms (lemmas) for each word."""
    expanded: Set[str] = set(words)
    for w in words:
        for syn in wn.synsets(w):
            for lemma in syn.lemmas():
                term = lemma.name().replace("_", " ").lower()
                if len(term) >= MIN_WORD_LEN and term not in STOPWORDS:
                    expanded.add(term)
    # simple heuristic: keep shortest unique spelling if duplicates
    return sorted(expanded, key=lambda s: (len(s), s))

def write_record(db_handle, url: str, kw: str, desc: str, title: str):
    db_handle.write(DB_LAYOUT.format(
        url=url,
        keywords=kw,
        desc=desc.replace("\t", " "),
        title=title.replace("\t", " "),
    ))

# ──────────────────────────────────────────────
# MAIN SPIDER
# ──────────────────────────────────────────────
def crawl():
    # Seed queue ---------------------------------------------------
    seeds = [normalize_url(u.strip()) for u in Path(SEEDS_FILE).read_text().splitlines() if u.strip()]
    url_queue: deque[str] = deque(seeds)
    known: Set[str] = set(url_queue)
    next_queue: deque[str] = deque()

    # Robots parsers cache
    robots_cache: dict[str, robotparser.RobotFileParser] = {}

    # Output DB handling -------------------------------------------
    db_index = 1
    created_total = 0
    created_this_file = 0
    output_files: List[Path] = []

    def open_db(idx: int):
        fname = DB_TEMPLATE.replace("#", str(idx))
        fh = open(fname, "a", encoding="utf-8")
        output_files.append(Path(fname))
        return fh

    db = open_db(db_index)

    # HTTP session
    session = requests.Session()

    for iteration in range(1, MAX_ITERATIONS + 1):
        if not url_queue:
            break
        print(f"\n=== Iteration {iteration} | {len(url_queue)} pages queued ===")
        processed = 0

        while url_queue:
            url = url_queue.popleft()
            processed += 1
            print(f"-> [{processed}/{len(url_queue)+processed}] {url}")

            if not should_process_url(url):
                print("      skipped (rule)")
                continue

            # robots.txt check
            if RESPECT_ROBOTS:
                ext = tldextract.extract(url)
                host = f"{ext.subdomain+'.' if ext.subdomain else ''}{ext.domain}.{ext.suffix}"
                if host not in robots_cache:
                    rp = robotparser.RobotFileParser()
                    rp.set_url(f"https://{host}/robots.txt")
                    try:
                        rp.read()
                    except Exception:
                        pass  # treat unreadable as allowed
                    robots_cache[host] = rp
                if not robots_cache[host].can_fetch(USER_AGENT, url):
                    print("      disallowed by robots.txt")
                    continue

            resp = polite_get(url, session)
            if not resp or not resp.ok:
                continue
            ct = resp.headers.get("content-type", "")
            if not is_content_type_ok(ct):
                print(f"      skipped (content-type {ct})")
                continue

            page_bytes = resp.content
            page = ""
            try:
                page = page_bytes.decode(resp.encoding or "utf-8", errors="replace")
            except Exception:
                page = page_bytes.decode("utf-8", errors="replace")

            soup = BeautifulSoup(page, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            description = extract_meta(soup, "description") or generate_summary(strip_html(page))
            keywords = (
                extract_meta(soup, "keywords")
                or (autokeywords(strip_html(page)) if GENERATE_KEYWORDS else "")
            )

            # write DB record
            write_record(db, url, keywords, description, title)
            created_total += 1
            created_this_file += 1
            if created_total >= MAX_DB_SIZE:
                print("### Max DB size reached, stopping crawl.")
                url_queue.clear()
                break
            if created_this_file >= LINES_PER_FILE:
                db.close()
                db_index += 1
                created_this_file = 0
                db = open_db(db_index)

            # enqueue links for next iteration
            if iteration < MAX_ITERATIONS:
                links = [
                    urlparse.urljoin(url, a.get("href"))
                    for a in soup.find_all("a", href=True)
                ]
                added = 0
                for link in links:
                    link = normalize_url(link)
                    if link not in known and should_process_url(link):
                        next_queue.append(link)
                        known.add(link)
                        added += 1
                print(f"      +{added} new links")
            time.sleep(FETCH_PAUSE)

        # prepare next round
        url_queue, next_queue = next_queue, deque()
        print(f"=== Iteration {iteration} done | total records: {created_total} ===")

    db.close()
    print("\nCrawl finished. Building helper lists…")
    build_helper_lists(output_files)
    print("All done.")

# ──────────────────────────────────────────────
# POST-PROCESSING: split columns for convenience
# ──────────────────────────────────────────────
def build_helper_lists(files: List[Path]):
    aggregate: List[List[str]] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            aggregate.extend(list(reader))

    cols = {
        "urls.dat":         0,
        "keywords.dat":     1,
        "descriptions.dat": 2,
        "title.dat":        3,
    }
    for fname, idx in cols.items():
        with open(fname, "w", encoding="utf-8") as out:
            for row in aggregate:
                if len(row) > idx:
                    out.write(row[idx] + "\n")

    # reversed order
    for fname in cols:
        lines = Path(fname).read_text(encoding="utf-8").splitlines()
        Path(f"rev_{fname}").write_text("\n".join(reversed(lines)), encoding="utf-8")
        Path(f"sort_{fname}").write_text("\n".join(sorted(lines)), encoding="utf-8")
        Path(f"revsort_{fname}").write_text("\n".join(sorted(lines, reverse=True)), encoding="utf-8")

# ──────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        crawl()
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting…")
        sys.exit(1)
