"""
Keyword library for PM skills/experience.

How it works
------------
1. A curated vocabulary of common PM keywords is maintained in _VOCABULARY.
2. For each new JD, text-matching against the vocabulary increments keyword
   counts — no LLM tokens consumed.
3. Only keywords with count >= popularity_threshold (default: 2) are
   considered "popular" and eligible for inclusion in generated CVs/cover
   letters. This filters out niche, company-specific jargon.
4. LLM-based extraction is used *only* for discovering keywords that are not
   yet in the vocabulary, and only when explicitly called.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from rich import print as rprint

_LIBRARY_PATH = Path(__file__).parent.parent / "data" / "keyword_library.json"

CATEGORIES = ("methodology", "tool", "skill", "practice", "domain")

# ── Curated PM keyword vocabulary ────────────────────────────────────────────
# Each entry: "Canonical Name": ("category", [match_patterns...])
# Patterns are matched case-insensitively as substrings inside the JD text.

_VOCABULARY: dict[str, tuple[str, list[str]]] = {
    # Methodologies / Frameworks
    "Agile":                    ("methodology", ["agile"]),
    "Scrum":                    ("methodology", ["scrum"]),
    "Kanban":                   ("methodology", ["kanban"]),
    "Lean":                     ("methodology", ["lean methodology", "lean product"]),
    "SAFe":                     ("methodology", ["safe framework", "scaled agile"]),
    "OKRs (Objectives and Key Results)": ("methodology", ["okr", "objectives and key results"]),
    "KPIs (Key Performance Indicators)": ("methodology", ["kpi", "key performance indicator"]),
    "Jobs to Be Done (JTBD)":   ("methodology", ["jtbd", "jobs to be done"]),
    "Continuous Discovery":     ("methodology", ["continuous discovery"]),
    "Design Thinking":          ("methodology", ["design thinking"]),
    "Shape Up":                 ("methodology", ["shape up"]),
    "Dual-track Agile":         ("methodology", ["dual-track"]),
    "CI/CD":                    ("methodology", ["ci/cd", "continuous integration", "continuous delivery"]),
    # Skills
    "Product strategy":         ("skill", ["product strategy", "product vision"]),
    "Roadmapping":              ("skill", ["roadmap", "road map"]),
    "Prioritisation":           ("skill", ["prioriti"]),
    "Stakeholder management":   ("skill", ["stakeholder"]),
    "User research":            ("skill", ["user research", "user interview", "usability"]),
    "Product discovery":        ("skill", ["product discovery", "discovery process"]),
    "Go-to-market":             ("skill", ["go-to-market", "gtm", "launch strategy"]),
    "Backlog management":       ("skill", ["backlog"]),
    "Requirements definition":  ("skill", ["requirements", "acceptance criteria", "user stor"]),
    "Product analytics":        ("skill", ["product analytics", "product metrics"]),
    "Competitive analysis":     ("skill", ["competitive analysis", "competitive research", "market research"]),
    "Cross-functional collaboration": ("skill", ["cross-functional", "cross functional"]),
    "Metrics & success criteria": ("skill", ["success metrics", "north star", "metric", "metrics tracking", "metric tracking"]),
    "Problem definition":       ("skill", ["problem definition", "problem statement"]),
    "Hypothesis validation":    ("skill", ["hypothesis", "validate assumption"]),
    "Backend systems":          ("skill", ["backend system", "back-end system", "server-side system", "backend architecture", "backend engineer", "backend service", "backend platform"]),
    "API & system integrations": ("skill", ["third-party integration", "third party integration", "api integration", "system integration", "partner integration", "external integration", "vendor integration"]),
    "Machine learning":         ("skill", ["machine learning", "deep learning", "ml model", "neural network", "natural language processing", "nlp model"]),
    "AI-driven products":       ("skill", ["ai-driven", "ai-powered", "ai-assisted", "ai-enhanced", "llm-powered", "generative ai"]),
    "Travel domain expertise":  ("domain", ["travel knowledge", "travel domain", "travel industry", "travel tech", "travel platform", "flight distribution", "gds", "global distribution system", "online travel agency", "ota "]),
    # Practices
    "A/B testing":              ("practice", ["a/b test", "ab test", "split test"]),
    "Experimentation":          ("practice", ["experiment"]),
    "Data-driven decision-making": ("practice", ["data-driven", "data driven"]),
    "Product analytics":        ("practice", ["funnel analysis", "retention analysis", "cohort"]),
    "User journey mapping":     ("practice", ["user journey", "customer journey"]),
    "UX design":                ("practice", ["ux", "user experience design"]),
    "Prototyping":              ("practice", ["prototype", "wireframe", "mockup"]),
    "Growth hacking":           ("practice", ["growth loop", "growth hack", "growth experiment"]),
    "Workflow automation":      ("practice", ["workflow automation", "automate workflow", "process automation", "automation workflow", "automated workflow"]),
    # Tools
    "Jira":                     ("tool", ["jira"]),
    "Confluence":               ("tool", ["confluence"]),
    "Figma":                    ("tool", ["figma"]),
    "Miro":                     ("tool", ["miro"]),
    "Amplitude":                ("tool", ["amplitude"]),
    "Mixpanel":                 ("tool", ["mixpanel"]),
    "Looker":                   ("tool", ["looker"]),
    "Google Analytics":         ("tool", ["google analytics"]),
    "SQL":                      ("tool", ["sql"]),
    "Python":                   ("tool", ["python"]),
    "Tableau":                  ("tool", ["tableau"]),
    "Power BI":                 ("tool", ["power bi"]),
    "Pendo":                    ("tool", ["pendo"]),
    "ProductBoard":             ("tool", ["productboard"]),
    "Linear":                   ("tool", ["linear.app", " linear "]),
    "Notion":                   ("tool", ["notion"]),
    "Intercom":                 ("tool", ["intercom"]),
    # Domains
    "B2B SaaS":                 ("domain", ["b2b saas", "b2b software"]),
    "B2C":                      ("domain", ["b2c", "consumer product"]),
    "Fintech":                  ("domain", ["fintech", "financial technology"]),
    "API products":             ("domain", ["api product", "developer platform", "developer tool"]),
    "Enterprise software":      ("domain", ["enterprise software", "enterprise product"]),
    "Marketplace":              ("domain", ["marketplace"]),
    "Mobile":                   ("domain", ["mobile app", "ios", "android"]),
    "AI/ML products":           ("domain", ["ai product", "ml product", "machine learning product", "llm", "ai-native", "artificial intelligence"]),
    "Travel":                   ("domain", ["online travel", "travel tech"]),
    "Booking platforms":        ("domain", ["booking platform", "booking system", "reservation system", "inventory management"]),
    "Payments":                 ("domain", ["payment", "transaction", "checkout"]),
    "E-commerce":               ("domain", ["e-commerce", "ecommerce"]),
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_library() -> dict:
    if not _LIBRARY_PATH.exists():
        return _empty_library()
    with _LIBRARY_PATH.open() as f:
        return json.load(f)


def save_library(lib: dict) -> None:
    _LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lib["metadata"]["last_updated"] = str(date.today())
    with _LIBRARY_PATH.open("w") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False)


def _empty_library() -> dict:
    return {
        "metadata": {
            "last_updated": "",
            "total_jds_processed": 0,
            "popularity_threshold": 2,
        },
        "keywords": {},
    }


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _find_existing(lib: dict, canonical: str) -> Optional[str]:
    target = _normalise(canonical)
    for key in lib["keywords"]:
        if _normalise(key) == target:
            return key
    return None


# ── Text-matching extraction (no LLM, no tokens) ─────────────────────────────

def match_keywords_from_jd(jd_text: str) -> list[dict]:
    """
    Match PM keywords in jd_text using the curated vocabulary.
    Returns a list of {"keyword": str, "category": str} for every keyword found.
    Zero LLM token cost.
    """
    jd_lower = jd_text.lower()
    found = []
    for canonical, (category, patterns) in _VOCABULARY.items():
        if any(pat in jd_lower for pat in patterns):
            found.append({"keyword": canonical, "category": category})
    return found


# ── Post-generation keyword enforcement ──────────────────────────────────────

def check_missing_jd_keywords(cv_sections: dict, jd_matched: list[dict]) -> list[dict]:
    """Return JD-matched keywords absent from all text in cv_sections.

    Presence is checked via vocabulary patterns (not just the canonical name)
    so a bullet containing "third-party integration" satisfies the
    "API & system integrations" keyword, etc.
    """
    texts: list[str] = []
    for job in cv_sections.get("work_experience", []):
        texts.extend(job.get("bullets", []))
    for proj in cv_sections.get("projects", []):
        texts.extend(proj.get("bullets", []))
    skills = cv_sections.get("skills", {})
    if isinstance(skills, dict):
        texts.append(skills.get("proficiency", "") or "")
        texts.append(skills.get("tools", "") or "")
    elif isinstance(skills, list):
        texts.extend(str(s) for s in skills)
    combined = " ".join(texts).lower()

    missing = []
    for entry in jd_matched:
        canonical = entry.get("keyword", "")
        category = entry.get("category", "skill")
        vocab = _VOCABULARY.get(canonical)
        patterns = vocab[1] if vocab else []
        # Keyword is present if any pattern matches OR the canonical name itself appears
        present = any(pat in combined for pat in patterns) or canonical.lower() in combined
        if not present:
            missing.append({"keyword": canonical, "category": category})
    return missing


# ── LLM extraction (used only to discover new keywords not in vocabulary) ────

def _is_503_error(e: Exception) -> bool:
    return "503" in str(e) or "service_unavailable" in str(e).lower()

@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(15),
    retry=retry_if_exception(_is_503_error),
    before_sleep=lambda retry_state: rprint(f"  [yellow]API Error. Retrying in 2 seconds... (Attempt {retry_state.attempt_number}/15)[/yellow]"),
    reraise=True
)
def discover_new_keywords(jd_text: str, client: OpenAI, model: str) -> list[dict]:
    """
    Ask the LLM for keywords that might not be in the vocabulary yet.
    Call sparingly — consumes Groq tokens.
    """
    known = set(_VOCABULARY.keys())
    prompt = f"""Extract PM skills/methodologies/tools NOT in this known list.

Known list (skip these): {', '.join(sorted(known))}

Return ONLY a JSON array — no markdown:
[{{"keyword": "canonical 1-4 word name", "category": "methodology|tool|skill|practice|domain"}}]

Rules:
- Only include if genuinely useful for a PM CV and NOT already in the known list
- Skip generic phrases ("team player", "fast-paced", "passion for")
- Canonical forms only ("A/B tests" → "A/B testing")
- Return [] if nothing new found

JOB DESCRIPTION:
{jd_text[:3000]}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        result = json.loads(content)
        return [e for e in result if isinstance(e, dict) and "keyword" in e]
    except json.JSONDecodeError:
        return []


# ── Library update ────────────────────────────────────────────────────────────

def update_library_from_jd(
    jd_text: str,
    client: OpenAI | None = None,
    model: str | None = None,
    lib: Optional[dict] = None,
    save: bool = True,
    discover: bool = False,
) -> dict:
    """
    Match keywords in jd_text and update counts in the library.

    - Always uses text-matching (free, no tokens).
    - If discover=True and client is provided, also calls the LLM to find
      keywords not yet in the vocabulary (use sparingly due to token cost).
    """
    if lib is None:
        lib = load_library()

    entries = match_keywords_from_jd(jd_text)

    if discover and client and model:
        new_entries = discover_new_keywords(jd_text, client, model)
        entries.extend(new_entries)

    for entry in entries:
        canonical = entry.get("keyword", "").strip()
        category = entry.get("category", "skill")
        if not canonical:
            continue
        existing_key = _find_existing(lib, canonical)
        if existing_key:
            lib["keywords"][existing_key]["count"] += 1
        else:
            lib["keywords"][canonical] = {"category": category, "count": 1}

    lib["metadata"]["total_jds_processed"] += 1
    if save:
        save_library(lib)
    return lib


# ── Querying ──────────────────────────────────────────────────────────────────

def get_popular_keywords(min_count: Optional[int] = None) -> dict[str, list[str]]:
    """Return popular keywords grouped by category."""
    lib = load_library()
    threshold = min_count if min_count is not None else lib["metadata"].get("popularity_threshold", 2)
    result: dict[str, list[str]] = {}
    for kw, meta in lib["keywords"].items():
        if meta["count"] >= threshold:
            cat = meta.get("category", "skill")
            result.setdefault(cat, []).append(kw)
    return result


def get_popular_keywords_flat(min_count: Optional[int] = None) -> list[str]:
    """Return a flat sorted list of popular keywords."""
    grouped = get_popular_keywords(min_count)
    out = []
    for cat in CATEGORIES:
        out.extend(sorted(grouped.get(cat, [])))
    return out


def get_jd_required_keywords(jd_text: str, min_occurrences: int = 2) -> list[str]:
    """
    Return niche keywords (count < popularity_threshold in the library) that
    appear >= min_occurrences times in this specific JD.

    These are role-specific requirements that the employer clearly emphasises,
    so they should appear in the CV even though they're not broadly popular
    across the market.
    """
    lib = load_library()
    threshold = lib["metadata"].get("popularity_threshold", 2)
    jd_lower = jd_text.lower()

    required = []
    for canonical, (category, patterns) in _VOCABULARY.items():
        # Only consider niche (non-popular) keywords
        entry = _find_existing(lib, canonical)
        if entry and lib["keywords"][entry]["count"] >= threshold:
            continue  # already popular — handled by get_popular_keywords_flat()

        # Count how many times any pattern appears in this JD
        occurrences = sum(jd_lower.count(pat) for pat in patterns)
        if occurrences >= min_occurrences:
            required.append(canonical)

    return sorted(required)


def get_library_summary() -> str:
    lib = load_library()
    threshold = lib["metadata"].get("popularity_threshold", 2)
    total = lib["metadata"].get("total_jds_processed", 0)
    all_kws = lib["keywords"]
    popular = {k: v for k, v in all_kws.items() if v["count"] >= threshold}
    niche = {k: v for k, v in all_kws.items() if v["count"] < threshold}

    lines = [
        f"Keyword Library — {total} JDs processed",
        f"Popularity threshold: {threshold}+ occurrences",
        f"Popular keywords: {len(popular)} | Niche (excluded from CVs): {len(niche)}",
        "",
    ]
    for cat in CATEGORIES:
        cat_kws = sorted(
            [(k, v["count"]) for k, v in popular.items() if v.get("category") == cat],
            key=lambda x: -x[1],
        )
        if cat_kws:
            lines.append(f"  {cat.upper()}")
            for kw, cnt in cat_kws:
                lines.append(f"    {kw:<45} ×{cnt}")
    lines.append("")
    lines.append(f"  NICHE / EXCLUDED (count < {threshold})")
    for cat in CATEGORIES:
        cat_niche = sorted(
            [(k, v["count"]) for k, v in niche.items() if v.get("category") == cat],
            key=lambda x: x[0],
        )
        for kw, cnt in cat_niche:
            lines.append(f"    [{cat[:3]}] {kw:<42} ×{cnt}")
    return "\n".join(lines)
