import asyncio
import json
import concurrent.futures
from playwright.sync_api import sync_playwright
from loguru import logger

try:
    from langsmith import traceable
except ImportError:
    traceable = lambda **kwargs: (lambda f: f)

BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _make_browser(p):
    return p.chromium.launch(
        headless=True,
        executable_path=BRAVE_PATH,
        args=LAUNCH_ARGS,
    )


def _make_page(browser):
    context = browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
    )
    page = context.new_page()
    page.route(
        "**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf,svg,mp4}",
        lambda r: r.abort()
    )
    return page


# ── RemoteOK ──────────────────────────────────────────────────────────────────

def _search_remoteok_sync(query: str) -> list[dict]:
    """
    RemoteOK has a public JSON API at /api — returns real job data with real URLs.
    We filter by query keywords client-side.
    """
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)
            page.goto("https://remoteok.com/api", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(1000)

            # Extract JSON text from page
            body = page.locator("pre").inner_text() if page.locator("pre").count() > 0 else page.content()
            browser.close()

            data = json.loads(body)
            keywords = query.lower().split()
            results = []

            for job in data:
                if not isinstance(job, dict) or "position" not in job:
                    continue

                # Filter by keyword match in title or tags
                title = job.get("position", "").lower()
                tags = [t.lower() for t in job.get("tags", [])]
                combined = title + " " + " ".join(tags)

                if any(kw in combined for kw in keywords):
                    results.append({
                        "title": job.get("position", ""),
                        "company": job.get("company", ""),
                        "location": "Remote",
                        "salary": job.get("salary", ""),
                        "skills": job.get("tags", [])[:10],
                        "description": job.get("description", "")[:300],
                        "url": job.get("url", f"https://remoteok.com/l/{job.get('id', '')}"),
                        "source": "remoteok",
                    })

                if len(results) >= 10:
                    break

            logger.info(f"RemoteOK: {len(results)} jobs matched '{query}'")
            return results

    except Exception as e:
        logger.error(f"RemoteOK failed: {e}")
        return []


# ── LinkedIn ──────────────────────────────────────────────────────────────────

def _search_linkedin_sync(query: str) -> list[dict]:
    """
    Scrapes LinkedIn public job listings.
    Extracts real job URLs from <a> tags — no LLM URL generation.
    """
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            encoded = query.replace(" ", "%20")
            url = (
                f"https://www.linkedin.com/jobs/search/"
                f"?keywords={encoded}&f_WT=2&sortBy=DD"
            )
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(4000)

            results = []

            # LinkedIn job cards have a consistent structure
            cards = page.locator("div.base-card").all()
            logger.info(f"LinkedIn: found {len(cards)} cards")

            for card in cards[:15]:
                try:
                    title = card.locator("h3.base-search-card__title").inner_text(timeout=2000).strip()
                    company = card.locator("h4.base-search-card__subtitle").inner_text(timeout=2000).strip()
                    location = card.locator("span.job-search-card__location").inner_text(timeout=2000).strip()

                    # Get REAL URL from the anchor tag
                    link = card.locator("a.base-card__full-link")
                    job_url = link.get_attribute("href", timeout=2000) or ""

                    # Clean tracking params — keep only the base URL
                    if "?" in job_url:
                        job_url = job_url.split("?")[0]

                    if not job_url.startswith("http"):
                        continue

                    results.append({
                        "title": title,
                        "company": company,
                        "location": location,
                        "salary": "",
                        "skills": [],
                        "description": "",
                        "url": job_url,
                        "source": "linkedin",
                    })
                except Exception:
                    continue

            browser.close()
            logger.info(f"LinkedIn: extracted {len(results)} real jobs")
            return results

    except Exception as e:
        logger.error(f"LinkedIn failed: {e}")
        return []


# ── Indeed ────────────────────────────────────────────────────────────────────

def _search_indeed_sync(query: str) -> list[dict]:
    """
    Indeed via their RSS feed — much more reliable than scraping HTML.
    RSS gives clean structured data with real URLs.
    """
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            # Indeed RSS feed — reliable, no JS rendering needed
            encoded = query.replace(" ", "+")
            url = f"https://www.indeed.com/rss?q={encoded}&l=remote&sort=date"
            logger.debug(f"Indeed RSS: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2000)

            content = page.content()
            browser.close()

            # Parse RSS XML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "lxml-xml")
            items = soup.find_all("item")

            if not items:
                # RSS not available — try HTML with updated selectors
                logger.warning("Indeed RSS empty, trying HTML fallback")
                return _search_indeed_html_sync(query)

            results = []
            for item in items[:15]:
                title = item.find("title")
                link = item.find("link")
                desc = item.find("description")

                title_text = title.get_text(strip=True) if title else ""
                job_url = link.get_text(strip=True) if link else ""
                description = desc.get_text(strip=True)[:300] if desc else ""

                # Extract company from title (format: "Job Title - Company")
                company = ""
                if " - " in title_text:
                    parts = title_text.rsplit(" - ", 1)
                    title_text = parts[0].strip()
                    company = parts[1].strip() if len(parts) > 1 else ""

                if job_url.startswith("http") and title_text:
                    results.append({
                        "title": title_text,
                        "company": company,
                        "location": "Remote",
                        "salary": "",
                        "skills": [],
                        "description": description,
                        "url": job_url,
                        "source": "indeed",
                    })

            logger.info(f"Indeed RSS: {len(results)} jobs")
            return results

    except Exception as e:
        logger.error(f"Indeed RSS failed: {e}")
        return []


def _search_indeed_html_sync(query: str) -> list[dict]:
    """Indeed HTML fallback with updated selectors."""
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            encoded = query.replace(" ", "+")
            page.goto(
                f"https://www.indeed.com/jobs?q={encoded}&l=remote&sort=date",
                wait_until="domcontentloaded", timeout=20000
            )
            page.wait_for_timeout(4000)

            results = []
            # Updated Indeed selectors for 2024/2025
            cards = page.locator("div.job_seen_beacon, div.jobsearch-ResultsList > li").all()
            logger.info(f"Indeed HTML: {len(cards)} cards")

            for card in cards[:15]:
                try:
                    # Multiple selector attempts for title
                    title, job_url, company, location = "", "", "", ""

                    for sel in ["h2.jobTitle a", "a[data-jk]", "h2 a span[title]"]:
                        el = card.locator(sel)
                        if el.count() > 0:
                            title = el.first.inner_text(timeout=1500).strip()
                            href = el.first.get_attribute("href", timeout=1500) or ""
                            if href:
                                job_url = f"https://www.indeed.com{href}" if href.startswith("/") else href
                            break

                    for sel in ["[data-testid='company-name']", "span.companyName"]:
                        el = card.locator(sel)
                        if el.count() > 0:
                            company = el.first.inner_text(timeout=1500).strip()
                            break

                    if title and job_url.startswith("http"):
                        results.append({
                            "title": title, "company": company,
                            "location": "Remote", "salary": "",
                            "skills": [], "description": "",
                            "url": job_url, "source": "indeed",
                        })
                except Exception:
                    continue

            browser.close()
            return results
    except Exception as e:
        logger.error(f"Indeed HTML fallback failed: {e}")
        return []

# ── Naukri ────────────────────────────────────────────────────────────────────

def _search_naukri_sync(query: str) -> list[dict]:
    """
    Naukri.com with updated 2024 selectors.
    """
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            encoded = query.replace(" ", "-").lower()
            url = f"https://www.naukri.com/{encoded}-jobs"
            logger.debug(f"Naukri URL: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(5000)

            results = []

            # Try multiple selector strategies — Naukri updates their DOM frequently
            selectors = [
                "div.cust-job-tuple",       # current 2024 selector
                "article.jobTuple",          # older selector
                "div.job-tuple-wrapper",     # alternate
                "div[class*='jobTuple']",    # wildcard
                "div[class*='tuple']",       # broad fallback
            ]

            cards = []
            for selector in selectors:
                cards = page.locator(selector).all()
                if cards:
                    logger.info(f"Naukri: matched selector '{selector}' — {len(cards)} cards")
                    break

            for card in cards[:15]:
                try:
                    # Try multiple title selectors
                    title_el = None
                    for sel in ["a.title", "a[class*='title']", "a.jobTitle", "a[title]"]:
                        if card.locator(sel).count() > 0:
                            title_el = card.locator(sel).first
                            break

                    if not title_el:
                        continue

                    title = title_el.inner_text(timeout=2000).strip()
                    job_url = title_el.get_attribute("href", timeout=2000) or ""

                    # Company
                    company = ""
                    for sel in ["a.subTitle", "a[class*='company']", "span[class*='company']"]:
                        if card.locator(sel).count() > 0:
                            company = card.locator(sel).first.inner_text(timeout=2000).strip()
                            break

                    # Experience
                    exp = ""
                    for sel in ["span[class*='expwdth']", "li.experience span", "span[class*='exp']"]:
                        if card.locator(sel).count() > 0:
                            exp = card.locator(sel).first.inner_text(timeout=2000).strip()
                            break

                    if not job_url.startswith("http"):
                        continue

                    results.append({
                        "title": title,
                        "company": company,
                        "location": "India",
                        "salary": "",
                        "skills": [],
                        "description": f"Experience: {exp}",
                        "url": job_url,
                        "source": "naukri",
                    })
                except Exception as e:
                    logger.debug(f"Naukri card parse error: {e}")
                    continue

            browser.close()
            logger.info(f"Naukri: extracted {len(results)} jobs")
            return results

    except Exception as e:
        logger.error(f"Naukri failed: {e}")
        return []

# ── Wellfound ─────────────────────────────────────────────────────────────────

def _search_wellfound_sync(query: str) -> list[dict]:
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            encoded = query.replace(" ", "%20")
            url = f"https://wellfound.com/jobs?q={encoded}"
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(4000)

            results = []
            cards = page.locator("div[data-test='StartupResult']").all()
            logger.info(f"Wellfound: found {len(cards)} cards")

            for card in cards[:10]:
                try:
                    title_el = card.locator("a[data-test='job-title']")
                    title = title_el.inner_text(timeout=2000).strip()
                    href = title_el.get_attribute("href", timeout=2000) or ""
                    job_url = f"https://wellfound.com{href}" if href.startswith("/") else href

                    company_el = card.locator("a[data-test='startup-link']")
                    company = company_el.inner_text(timeout=2000).strip()

                    if not job_url.startswith("http"):
                        continue

                    results.append({
                        "title": title,
                        "company": company,
                        "location": "Remote",
                        "salary": "",
                        "skills": [],
                        "description": "",
                        "url": job_url,
                        "source": "wellfound",
                    })
                except Exception:
                    continue

            browser.close()
            logger.info(f"Wellfound: extracted {len(results)} real jobs")
            return results

    except Exception as e:
        logger.error(f"Wellfound failed: {e}")
        return []


# ── YC Jobs ───────────────────────────────────────────────────────────────────

def _search_ycombinator_sync(query: str) -> list[dict]:
    try:
        with sync_playwright() as p:
            browser = _make_browser(p)
            page = _make_page(browser)

            encoded = query.replace(" ", "%20")
            url = f"https://www.ycombinator.com/jobs/search?query={encoded}"
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)

            results = []
            cards = page.locator("a._job_").all()
            logger.info(f"YC: found {len(cards)} cards")

            for card in cards[:10]:
                try:
                    title = card.locator("div._jobTitle_").inner_text(timeout=2000).strip()
                    company = card.locator("div._company_").inner_text(timeout=2000).strip()
                    href = card.get_attribute("href", timeout=2000) or ""
                    job_url = f"https://www.ycombinator.com{href}" if href.startswith("/") else href

                    if not job_url.startswith("http"):
                        continue

                    results.append({
                        "title": title,
                        "company": company,
                        "location": "Remote",
                        "salary": "",
                        "skills": [],
                        "description": "",
                        "url": job_url,
                        "source": "ycombinator",
                    })
                except Exception:
                    continue

            browser.close()
            logger.info(f"YC: extracted {len(results)} real jobs")
            return results

    except Exception as e:
        logger.error(f"YC failed: {e}")
        return []


# ── Public async API ──────────────────────────────────────────────────────────

class BrowserTools:
    """
    Now returns List[dict] with REAL URLs extracted from DOM.
    The LLM never generates URLs anymore — it only enriches existing data.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def search_remoteok(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_remoteok_sync, query)

    async def search_linkedin(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_linkedin_sync, query)

    async def search_indeed(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_indeed_sync, query)

    async def search_naukri(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_naukri_sync, query)

    async def search_wellfound(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_wellfound_sync, query)

    async def search_ycombinator(self, query: str) -> list[dict]:
        return await asyncio.to_thread(_search_ycombinator_sync, query)

    def _decorate(cls):
        for name in ("search_remoteok", "search_linkedin", "search_indeed",
                     "search_naukri", "search_wellfound", "search_ycombinator"):
            fn = getattr(cls, name)
            setattr(cls, name, traceable(name=name, run_type="tool")(fn))
        return cls


BrowserTools._decorate(BrowserTools)