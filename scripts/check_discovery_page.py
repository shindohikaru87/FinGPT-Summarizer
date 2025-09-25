# scripts/check_discovery_page.py
#!/usr/bin/env python3
import re, sys, requests
from bs4 import BeautifulSoup

def main(url, allow_csv, deny_csv):
    allow = [re.compile(p) for p in allow_csv.split("|||") if p.strip()]
    deny  = [re.compile(p) for p in deny_csv.split("|||") if p.strip()]
    html = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for a in soup.select("a[href]"):
        hrefs.append(a.get("href"))
    total = len(hrefs)
    abs_or_rel = hrefs
    kept = []
    rejected = []
    for h in abs_or_rel:
        s = h
        if any(rx.search(s) for rx in deny):
            rejected.append((s,"deny"))
            continue
        if allow and not any(rx.search(s) for rx in allow):
            rejected.append((s,"no-allow-match"))
            continue
        kept.append(s)
    print(f"Total links: {total}")
    print(f"Kept: {len(kept)}  Rejected: {len(rejected)}")
    for s, why in rejected[:20]:
        print(" REJECT", why, s)
    for s in kept[:20]:
        print(" KEEP", s)

if __name__ == "__main__":
    # example:
    # python scripts/check_discovery_page.py "https://www.cnbc.com/markets/" "^/202[45]/\\d{2}/\\d{2}/|||^https?://www\\.cnbc\\.com/202[45]/\\d{2}/\\d{2}/" "/video/|||/live/|||/pro/|||/make-it/|||/select/|||/events/"
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "")
