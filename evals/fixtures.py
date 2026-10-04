"""Hermetic browser fixtures for agent/tool-call evals.

FakeBrowser stands in for `tools.browser_tools.BrowserTools` so evals run
deterministically (no network/Playwright) while still recording *which* tool
was invoked and with which query — exactly what tool-dispatch evals need.
"""

import copy

FIXTURE_JOBS: dict[str, list[dict]] = {
    "remoteok": [
        {"title": "Backend Engineer (Python)", "company": "Acme Remote", "location": "Remote",
         "salary": "$120k", "skills": ["python", "fastapi", "postgres"],
         "description": "Build APIs for a remote-first product company.",
         "url": "https://remoteok.com/l/remoteok-1", "source": "remoteok"},
        {"title": "Senior Software Engineer", "company": "Globex", "location": "Remote",
         "salary": "$140k", "skills": ["python", "aws", "sql"],
         "description": "Platform team, distributed systems.",
         "url": "https://remoteok.com/l/remoteok-2", "source": "remoteok"},
        {"title": "Full Stack Developer", "company": "Initech", "location": "Remote",
         "salary": "$110k", "skills": ["react", "node", "postgres"],
         "description": "Ship customer-facing features.",
         "url": "https://remoteok.com/l/remoteok-3", "source": "remoteok"},
    ],
    "linkedin": [
        {"title": "Python Developer", "company": "Umbrella Systems", "location": "Remote",
         "salary": "", "skills": [], "description": "",
         "url": "https://www.linkedin.com/jobs/view/linkedin-1", "source": "linkedin"},
        {"title": "DevOps Engineer", "company": "Stark Industries", "location": "Remote",
         "salary": "", "skills": [], "description": "",
         "url": "https://www.linkedin.com/jobs/view/linkedin-2", "source": "linkedin"},
    ],
    "indeed": [
        {"title": "Backend Software Engineer", "company": "Wayne Enterprises", "location": "Remote",
         "salary": "", "skills": [], "description": "Build scalable backend services.",
         "url": "https://www.indeed.com/viewjob?jk=indeed-1", "source": "indeed"},
    ],
    "naukri": [
        {"title": "Python - Node Js Developer", "company": "Tata Consultancy", "location": "India",
         "salary": "", "skills": [], "description": "Experience: 3-8 years",
         "url": "https://www.naukri.com/job-listings-naukri-1", "source": "naukri"},
    ],
    "wellfound": [
        {"title": "Senior Backend Engineer", "company": "Rocket Startup", "location": "Remote",
         "salary": "", "skills": [], "description": "",
         "url": "https://wellfound.com/jobs/wellfound-1", "source": "wellfound"},
    ],
    "ycombinator": [
        {"title": "Software Engineer (Backend)", "company": "YC Unicorn", "location": "Remote",
         "salary": "", "skills": [], "description": "",
         "url": "https://www.ycombinator.com/jobs/yc-1", "source": "ycombinator"},
    ],
}

SITE_METHODS = (
    "search_remoteok", "search_linkedin", "search_indeed",
    "search_naukri", "search_wellfound", "search_ycombinator",
)


class FakeBrowser:
    """
    Deterministic stand-in for BrowserTools.
    `recorder` grows one entry per tool call: {"tool": <method>, "query": <query>}.
    """

    def __init__(self, recorder: list | None = None):
        self.recorder = recorder if recorder is not None else []
        self._decorate()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def _call(self, site: str, query: str) -> list[dict]:
        self.recorder.append({"tool": f"search_{site}", "query": query})
        return copy.deepcopy(FIXTURE_JOBS.get(site, []))

    async def search_remoteok(self, query):    return await self._call("remoteok", query)
    async def search_linkedin(self, query):    return await self._call("linkedin", query)
    async def search_indeed(self, query):      return await self._call("indeed", query)
    async def search_naukri(self, query):      return await self._call("naukri", query)
    async def search_wellfound(self, query):   return await self._call("wellfound", query)
    async def search_ycombinator(self, query): return await self._call("ycombinator", query)

    def _decorate(self):
        try:
            from langsmith import traceable
        except ImportError:
            return
        for name in SITE_METHODS:
            setattr(self, name, traceable(name=name, run_type="tool")(
                getattr(self, name)
            ))