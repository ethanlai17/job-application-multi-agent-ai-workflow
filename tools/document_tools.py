import json
import re
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from rich import print as rprint

try:
    from config.cv_data import (
        APPLICANT_NAME, APPLICANT_TITLE, CONTACT, EDUCATION,
        PROJECTS, WORK_EXPERIENCE_FIXED, EXTRA_BULLETS,
    )
except ImportError:
    from config.cv_data_example import (
        APPLICANT_NAME, APPLICANT_TITLE, CONTACT, EDUCATION,
        PROJECTS, WORK_EXPERIENCE_FIXED, EXTRA_BULLETS,
    )

try:
    from config.experience_library import EXPERIENCE_LIBRARY
except ImportError:
    EXPERIENCE_LIBRARY = {}


def _clean_bullet(text: str) -> str:
    """Collapse irregular whitespace in bullet text to single spaces."""
    return re.sub(r"[ \t]+", " ", text).strip()


def _experience_library_block() -> str:
    if not EXPERIENCE_LIBRARY:
        return ""
    lines = [
        "EXPERIENCE LIBRARY — PRIMARY source for bullet selection.",
        "Each section groups bullets by capability. Multiple framings of the same achievement",
        "are provided (reframed + original). Always pick the version whose language most",
        "closely mirrors the JD's exact wording, preserving all metrics.",
    ]
    for category, bullets in EXPERIENCE_LIBRARY.items():
        lines.append(f"\n--- {category} ---")
        current_company = None
        for b in bullets:
            if b["company"] != current_company:
                current_company = b["company"]
                lines.append(f'{current_company}:')
            lines.append(f'  • {b["text"]}')
    return "\n".join(lines)


def _extra_bullets_block() -> str:
    if not EXTRA_BULLETS:
        return ""
    lines = [
        "\nADDITIONAL VERIFIED BULLETS (confirmed real experience — not yet in original CV PDF):",
        "Use these EXACTLY as written when the JD calls for the corresponding skill.",
    ]
    for b in EXTRA_BULLETS:
        lines.append(f'  {b["company"]}: "{b["text"]}"')
        lines.append(f'  [Include when: {b["use_when"]}]')
    return "\n".join(lines)


def _work_experience_rules() -> str:
    """Generate the work experience prompt rules from WORK_EXPERIENCE_FIXED."""
    mandatory = WORK_EXPERIENCE_FIXED[:3]
    optional = WORK_EXPERIENCE_FIXED[3] if len(WORK_EXPERIENCE_FIXED) > 3 else None
    top_two_companies = " and ".join(j["company"] for j in mandatory[:2])

    lines = [
        f"1. work_experience — MUST include the first {len(mandatory)} jobs in this EXACT order "
        f"with these EXACT titles:"
    ]
    for i, job in enumerate(mandatory, 1):
        lines.append(f'   {i}. "{job["title"]}" at "{job["company"]}",  {job["dates"]}')

    if optional:
        lines.append(
            f'   The {len(mandatory) + 1}th job "{optional["title"]}" at "{optional["company"]}"'
            f' ({optional["dates"]}) is OPTIONAL:'
        )
        lines.append(
            f"   include it ONLY if it adds clear value for this specific JD.\n"
            f"   If {top_two_companies} bullets already cover the JD requirements comprehensively,\n"
            f"   OMIT {optional['company']} and still use the standard bullet counts (TUI UK: 5, Inspectorio: 4, FPT Software: 2)."
        )

    return "\n".join(lines)


def _projects_prompt_rules() -> str:
    """Generate the projects prompt section from PROJECTS."""
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]
    lines = [
        f"2. projects — include EXACTLY these {len(PROJECTS)} project(s) in this order (use verbatim):"
    ]
    for i, proj in enumerate(PROJECTS):
        ord_str = ordinals[i] if i < len(ordinals) else f"{i + 1}th"
        subtitle = proj.get("subtitle", "")
        bullets_str = json.dumps(proj["bullets"], ensure_ascii=False)
        lines.append(
            f'   {ord_str}: {{"name": "{proj["name"]}", "subtitle": "{subtitle}", '
            f'"dates": "{proj["dates"]}",\n'
            f"     \"bullets\": {bullets_str}}}"
        )
    return "\n".join(lines)


# ── CV reading ────────────────────────────────────────────────────────────────

def read_cv(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CV file not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in (".docx", ".doc"):
        return _read_docx(path)
    raise ValueError(f"Unsupported CV format: {suffix}. Use PDF or DOCX.")


def _read_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── LLM generation ────────────────────────────────────────────────────────────

def _popular_keywords_rule(popular_keywords: list[str] | None) -> str:
    """Build the skills-filter rule injected into the CV prompt."""
    if not popular_keywords:
        return (
            "   - Include ALL relevant skills from the JD, even if not in the applicant's CV.\n"
        )
    kw_list = ", ".join(popular_keywords)
    return (
        f"   - APPROVED KEYWORD LIST (market-popular skills + every skill explicitly required in this JD):\n"
        f"     {kw_list}\n"
        f"   - Only include skills/keywords from this approved list OR from {APPLICANT_NAME}'s original CV or ADDITIONAL VERIFIED BULLETS.\n"
        f"   - Do NOT add niche, company-specific jargon from this JD that is not on the approved list\n"
        f"     (e.g. one-off phrases like 'thinking in bets', 'thin/vertical slicing', 'test-learn-iterate'\n"
        f"     should only appear if they are already in the approved list above).\n"
    )


def _is_503_error(e: Exception) -> bool:
    return "503" in str(e)


@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(15),
    retry=retry_if_exception(_is_503_error),
    before_sleep=lambda retry_state: rprint(f"  [yellow]API Error. Retrying in 2 seconds... (Attempt {retry_state.attempt_number}/15)[/yellow]"),
    reraise=True
)
def generate_tailored_cv(
    cv_text: str,
    job_description: str,
    job_title: str,
    company: str,
    client: OpenAI,
    model: str,
    popular_keywords: list[str] | None = None,
) -> dict:
    """Ask the LLM for tailored CV sections. Returns a dict with work_experience, projects, skills."""
    prompt = f"""You are an expert CV writer tailoring {APPLICANT_NAME}'s CV for a specific job.

Return ONLY a valid JSON object — no markdown fences, no explanation — matching this schema exactly:
{{
  "work_experience": [
    {{"title": "job title", "company": "company name", "dates": "date range", "bullets": ["bullet 1", "bullet 2", ...]}}
  ],
  "projects": [
    {{"name": "project name", "subtitle": "role label", "dates": "date range", "bullets": ["bullet 1", "bullet 2"]}},
    {{"name": "project name", "subtitle": "role label", "dates": "date range", "bullets": ["bullet 1", "bullet 2"]}}
  ],
  "skills": {{
    "proficiency": "comma-separated list of competencies",
    "tools": "comma-separated list of tools"
  }}
}}

RULES:

{_work_experience_rules()}
   - Generate bullet CANDIDATES per job, ranked most-to-least relevant to this JD:
       TUI UK:       5–8 bullets  (minimum 5, maximum 8)
       Inspectorio:  4–7 bullets  (minimum 4, maximum 7)
       FPT Software: 2–5 bullets  (minimum 2, maximum 5)
     Put the most JD-relevant bullets first — the renderer will automatically select the
     optimal count to fill one A4 page, always keeping TUI > Inspectorio > FPT.
   - BULLET SELECTION PROCESS — follow these steps in order:
       Step 1: Extract every responsibility and required qualification from the JD.
       Step 2: For each extracted requirement, identify the matching capability category in the EXPERIENCE LIBRARY,
               then scan all bullets in that category for direct or adjacent matches.
               "Adjacent" means the underlying skill/activity is the same even if the wording differs.
               EXAMPLE — JD says "Track adoption, usage, behaviour change and feedback; define leading indicators."
                         Match category: "Root Cause Analysis & Evidence Synthesis"
                         Adjacent bullet: "Developed real-time logging & dashboards for search microservices,
                         reducing incident resolution time by 30%" → INCLUDE IT (it is about tracking and measurement).
       Step 3: ALWAYS include an adjacent bullet when found. Pick the framing (reframed or original) whose language
               most closely mirrors the JD's wording, while preserving the original metric.
               If neither framing fits naturally, adapt the closest one — never invent a new bullet.
       Step 4: Fill the remaining bullet slots with the next-best matching bullets.
   - Keep each bullet to at most 2 lines. If a bullet wraps to a 2nd line, the last line MUST
     contain at least 5 words. NEVER end a bullet with a short time span, number, or 1-2 word
     phrase alone on the last line (e.g. "in 2 mos.", "in Q3", "across 3 teams", "via Agile"
     are all too short). Fix by moving the short phrase earlier in the sentence:
       BAD:  "…rolling out a booking amendment feature across 5 microservices in 2 mos."
       GOOD: "…rolling out a booking amendment feature in 2 mos. across 5 microservices"
     Do NOT shorten or remove content — always preserve the full detail and metric.
   - ALWAYS preserve the original metric/outcome (€ amount, %, hours saved, etc.) — never strip numbers.
   - Keep the original sentence structure; you may:
       • swap a word for a JD keyword (e.g. "iterative" → "test-learn-iterate", "data" → "data-driven")
       • insert a JD keyword phrase mid-sentence (e.g. "…via Continuous Discovery", "…using Agile sprints")
   - NEVER append "…demonstrating/reflecting/showcasing/highlighting X" at the end — forbidden.
   - Do NOT truncate bullets to just an action verb without context or metric.
   - Do NOT invent metrics, companies, or outcomes not in the original CV.
   - Do NOT fabricate bullets — only use or adapt bullets from the EXPERIENCE LIBRARY or ADDITIONAL VERIFIED BULLETS below.
   - Only weave in a JD keyword if it fits naturally. If it sounds forced, leave that bullet unchanged.
   - Target style: "Drove €1.6M revenue by launching global booking fee across 5 microservices via test-learn-iterate"
                   "Designed AI-powered personalisation engine via Continuous Discovery, boosting CTR 15% & conversion 3%"

{_projects_prompt_rules()}

3. skills — this is CRITICAL:
   - Read the FULL job description carefully and extract relevant skills, methodologies, frameworks, and tools.
   - Include ALL keywords from the APPROVED KEYWORD LIST below that are relevant to this JD — do not skip any approved keyword that appears in the JD.
   - Merge with {APPLICANT_NAME}'s existing skills, placing JD-matched skills first.
   - "proficiency" = core competencies: methodologies, frameworks, soft skills, PM skills (rendered as "Core Competencies" on the CV — an ATS-standard label)
   - "tools" = named software tools and platforms. Selection process:
       Step 1: For every responsibility and qualification in the JD, ask which tool from {APPLICANT_NAME}'s
               toolkit would be used to perform that task — add it.
       Step 2: Always prefer tools from this PRIORITY LIST (in order) before adding anything else:
               Python, SQL, Excel, Tableau, Power BI, Datadog, Grafana, AWS services, Google Analytics,
               Postman, Swagger, Bruno
       Step 3: Fill the tools line to its character limit — keep adding lower-priority tools from the
               original CV until you reach the limit. Never leave the tools line half-empty.
   - Keep each value (proficiency / tools) to ONE compact comma-separated line — no repetition, no padding.
     Fill to the character limit: proficiency target 90–110 characters, tools target 80–100 characters.
     Stop only when the next item would push the line over the limit or orphan a single word.
   - Spell out abbreviations on first use:
       write "Objectives and Key Results (OKRs)" not just "OKRs"
       write "Key Performance Indicators (KPIs)" not just "KPIs"
       write "Jobs to Be Done (JTBD)" not just "JTBD"
   - NEVER abbreviate skill names — write them in full:
       write "Stakeholder management" not "Stakeholder mgnt" or "Stakeholder mgmt"
   - Copy skill names EXACTLY as they appear in the APPROVED KEYWORD LIST — do not paraphrase or substitute words:
       write "Cross-functional collaboration" not "Cross-functional leadership" or "Cross-functional alignment"
   - Avoid vague language: never write "various", "several", "multiple", or "etc."
     Replace with specific counts or named examples from the original CV.
{_popular_keywords_rule(popular_keywords)}
4. Return ONLY the JSON — nothing else.

{_experience_library_block()}
{_extra_bullets_block()}

{APPLICANT_NAME.upper()}'S ORIGINAL CV (use for skills context; bullets above take precedence):
{cv_text}

TARGET ROLE:
Title: {job_title}
Company: {company}

Full Job Description:
{job_description}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown fences."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    # Strip markdown code fences if the model wraps with them
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    result = json.loads(content)

    # Enforce fixed titles/companies/dates — LLM controls bullets only.
    # First 3 jobs are mandatory; any beyond index 2 are optional — only added if LLM included them.
    jobs = result.get("work_experience", [])
    mandatory_count = min(3, len(WORK_EXPERIENCE_FIXED))
    for i, fixed in enumerate(WORK_EXPERIENCE_FIXED):
        if i < len(jobs):
            jobs[i]["title"] = fixed["title"]
            jobs[i]["company"] = fixed["company"]
            jobs[i]["dates"] = fixed["dates"]
        elif i < mandatory_count:
            jobs.append({**fixed, "bullets": []})
    result["work_experience"] = jobs

    # Normalise whitespace in every bullet so irregular LLM spacing never
    # causes inconsistent gaps when rendered (especially under justification).
    for job in result.get("work_experience", []):
        job["bullets"] = [_clean_bullet(b) for b in job.get("bullets", [])]
    for proj in result.get("projects", []):
        proj["bullets"] = [_clean_bullet(b) for b in proj.get("bullets", [])]
    skills = result.get("skills", {})
    if isinstance(skills, dict):
        if skills.get("proficiency"):
            skills["proficiency"] = _clean_bullet(skills["proficiency"])
        if skills.get("tools"):
            skills["tools"] = _clean_bullet(skills["tools"])

    return result


@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(15),
    retry=retry_if_exception(_is_503_error),
    before_sleep=lambda retry_state: rprint(f"  [yellow]API Error. Retrying in 2 seconds... (Attempt {retry_state.attempt_number}/15)[/yellow]"),
    reraise=True
)
def generate_cover_letter(
    cv_text: str,
    job_description: str,
    job_title: str,
    company: str,
    client: OpenAI,
    model: str,
) -> str:
    """Ask the LLM for cover letter body paragraphs only (no salutation/sign-off)."""
    system_instruction = (
        f"You are writing a cover letter body for {APPLICANT_NAME}, a {APPLICANT_TITLE}. "
        f"Write exactly 3 paragraphs, total under 280 words, in a formal but human style. "
        f"Focus on specific measurable impacts from {APPLICANT_NAME}'s experience and how they "
        "match what this company and role need. "
        "Do NOT include: a salutation (no 'Dear'), sign-off, subject line, or date — "
        "those are added separately. "
        "Do NOT use clichés like 'I am writing to express my interest' or "
        "'I am a passionate'. "
        "Do NOT use '--' (double hyphen) anywhere — use a comma, semicolon, or rewrite the sentence instead. "
        f"Open with what specifically draws {APPLICANT_NAME} to this company and role. "
        "Middle paragraph: 2-3 specific achievements with metrics that directly address the JD. "
        "Prioritise examples from TUI UK (most recent, UK-based) first; "
        "use Inspectorio only as a secondary source when it adds relevant evidence not covered by TUI. "
        "Closing: concise statement of contribution and confidence.\n\n"
        f"{APPLICANT_NAME.upper()}'S CV:\n{cv_text}"
    )

    prompt = (
        f"Write the cover letter body for:\n"
        f"Job Title: {job_title}\n"
        f"Company: {company}\n\n"
        f"Job Description:\n{job_description[:2000]}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
    )
    return (response.choices[0].message.content or "").strip()


# ── PDF rendering ─────────────────────────────────────────────────────────────

def render_cv_pdf(sections: dict, output_path: str) -> str:
    """Render a professional single-page A4 CV PDF using reportlab.

    Automatically scales line spacing upward (binary search) so the content
    always fills the full A4 page without overflowing to a second page.
    Bullet points are justified.
    """
    import io
    import fitz  # PyMuPDF — already a project dependency
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

    _DOC_KWARGS = dict(
        pagesize=A4, leftMargin=38, rightMargin=38, topMargin=28, bottomMargin=24,
    )

    def _build_story(scale: float, we_override=None) -> list:
        """Build the platypus story with vertical spacing multiplied by scale.

        we_override: if provided, replaces sections["work_experience"] (used by greedy optimizer).

        ATS compliance:
        - All body text >= 10pt (ATS minimum)
        - No Table elements — title rows use a single Paragraph with a right-aligned tab stop
        - Standard section headings (WORK EXPERIENCE, PROJECTS, EDUCATION, SKILLS)
        - Helvetica font (ATS-approved)
        - Justified bullets for clean text extraction
        """
        work_experience = we_override if we_override is not None else sections.get("work_experience", [])

        def s(name, **kw):
            # Single base font (Helvetica) throughout — bold applied via <b> markup only.
            # 11.5pt leading for 10pt text (1.15× — slightly looser than the ATS floor of 1.0×).
            defaults = dict(fontName="Helvetica", fontSize=10, leading=11.5 * scale)
            defaults.update(kw)
            return ParagraphStyle(name, **defaults)

        # Usable content width inside the page margins
        content_w = A4[0] - 38 - 38

        # Header — fixed, not scaled
        name_s    = s("Name",    fontSize=16, leading=20,
                      alignment=TA_CENTER, spaceAfter=1)
        contact_s = s("Contact", fontSize=9,  leading=11, alignment=TA_CENTER, spaceAfter=3)

        # Scaled body styles — all >= 10pt per ATS guidelines
        section_s   = s("Section",  fontSize=11, leading=13,
                         spaceBefore=3 * scale, spaceAfter=0.5 * scale)
        # Title row: bold title left, dates right — same line via tab stop (no Table needed)
        title_row_s = s("TitleRow", fontSize=10, leading=11.5 * scale, spaceAfter=0.5 * scale,
                         tabStops=[(content_w, TA_RIGHT)])
        # Hanging indent: • at the left margin, tab stop at 12pt snaps text to the continuation
        # column — so the first-line text start and all wrapped lines are at exactly the same x.
        # Tab (&#9;) is a fixed-width stop, so justification never stretches the gap after •.
        bullet_s    = s("Bullet",   fontSize=10, leading=11.5 * scale, leftIndent=12,
                         firstLineIndent=-12, spaceAfter=0.3 * scale, alignment=TA_JUSTIFY,
                         tabStops=[(12, TA_LEFT)])
        edu_s       = s("Edu",      fontSize=10, leading=11.5 * scale)
        edu_det_s   = s("EduDet",   fontSize=10, leading=11.5 * scale, leftIndent=12,
                         firstLineIndent=-12, spaceAfter=1.5 * scale, alignment=TA_JUSTIFY,
                         tabStops=[(12, TA_LEFT)])

        story = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph(f"<b>{APPLICANT_NAME.upper()}</b>", name_s))
        story.append(Paragraph(CONTACT, contact_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black,
                                spaceAfter=2 * scale))

        def section_header(text):
            story.append(Paragraph(f"<b>{text}</b>", section_s))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black,
                                    spaceAfter=1 * scale))

        def title_row(label, dates):
            # Single Paragraph with right-aligned tab stop — title left, dates right on the same line.
            # ATS-friendly (no Table) and reads naturally in linear text extraction.
            story.append(Paragraph(
                f'<b>{label}</b>&#9;{dates}',
                title_row_s,
            ))

        def add_bullets(bullets):
            for b in bullets:
                story.append(Paragraph(f"•&#9;{b}", bullet_s))

        # ── Work Experience ───────────────────────────────────────────────────
        section_header("WORK EXPERIENCE")
        for job in work_experience:
            label = f"{job['title']} - {job['company']}"
            title_row(label, job.get("dates", ""))
            add_bullets(job.get("bullets", []))
            story.append(Spacer(1, 2 * scale))

        # ── Projects ─────────────────────────────────────────────────────────
        section_header("PROJECTS")
        for proj in sections.get("projects", PROJECTS[:1]):
            label = proj["name"]
            if proj.get("subtitle"):
                label = f"{proj['subtitle']} - {proj['name']}"
            title_row(label, proj.get("dates", ""))
            add_bullets(proj.get("bullets", []))
            story.append(Spacer(1, 2 * scale))

        # ── Education (fixed) ────────────────────────────────────────────────
        section_header("EDUCATION")
        for edu in EDUCATION:
            story.append(Paragraph(f"<b>{edu['title']}</b>", edu_s))
            if edu["detail"]:
                story.append(Paragraph(f"•&#9;{edu['detail']}", edu_det_s))
            else:
                story.append(Spacer(1, 2 * scale))

        # ── Skills ───────────────────────────────────────────────────────────
        section_header("SKILLS")
        skills = sections.get("skills", {})
        if isinstance(skills, dict):
            if skills.get("proficiency"):
                story.append(Paragraph(
                    f"•&#9;<b>Core Competencies:</b> {skills['proficiency']}", bullet_s))
            if skills.get("tools"):
                story.append(Paragraph(
                    f"•&#9;<b>Tools:</b> {skills['tools']}", bullet_s))
        elif isinstance(skills, list):
            add_bullets(skills)

        return story

    def _page_count(scale: float, we_override=None) -> int:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, **_DOC_KWARGS)
        doc.build(_build_story(scale, we_override))
        buf.seek(0)
        pdf = fitz.open(stream=buf.read(), filetype="pdf")
        n = pdf.page_count
        pdf.close()
        return n

    def _greedy_trim_work_experience() -> list:
        """Maximize bullet counts at scale=1.0 subject to TUI > Inspectorio > FPT and page fit.

        Constraints:  TUI >= 5,  Inspectorio >= 4,  FPT >= 2
                      TUI count > Inspectorio count > FPT count always
        """
        we = sections.get("work_experience", [])

        def company_slot(marker):
            for i, job in enumerate(we):
                if marker in job.get("company", ""):
                    return i, len(job["bullets"])
            return None, 0

        tui_i,  tui_max  = company_slot("TUI")
        insp_i, insp_max = company_slot("Inspectorio")
        fpt_i,  fpt_max  = company_slot("FPT")

        counts = {}
        if tui_i  is not None: counts[tui_i]  = min(5, tui_max)
        if insp_i is not None: counts[insp_i] = min(4, insp_max)
        if fpt_i  is not None: counts[fpt_i]  = min(2, fpt_max)

        def make_we():
            return [
                {**job, "bullets": job["bullets"][:counts[i]]} if i in counts else job
                for i, job in enumerate(we)
            ]

        # If minimums already overflow one page, return as-is and let binary search handle it.
        if _page_count(1.0, make_we()) > 1:
            return make_we()

        # order: (index, upper_bound_index) — upper_bound_index must stay strictly greater
        order = [
            (tui_i,  None),
            (insp_i, tui_i),
            (fpt_i,  insp_i),
        ]

        while True:
            progress = False
            for idx, upper_idx in order:
                if idx is None or counts.get(idx, 0) >= {tui_i: tui_max, insp_i: insp_max, fpt_i: fpt_max}.get(idx, 0):
                    continue
                if upper_idx is not None and counts.get(idx, 0) + 1 >= counts.get(upper_idx, 0):
                    continue
                counts[idx] += 1
                if _page_count(1.0, make_we()) == 1:
                    progress = True
                else:
                    counts[idx] -= 1
            if not progress:
                break

        return make_we()

    def _trim_no_orphan(value: str, label: str, min_last_words: int = 4) -> str:
        """Remove trailing comma-separated items until the last rendered line has >= min_last_words."""
        from reportlab.pdfbase.pdfmetrics import stringWidth as sw
        cw = A4[0] - 38 - 38 - 12  # content width minus bullet leftIndent
        label_w = sw(label, "Helvetica-Bold", 10)
        avail_first = cw - label_w
        items = [x.strip() for x in value.split(",")]
        for n in range(len(items), 0, -1):
            candidate = ", ".join(items[:n])
            if sw(candidate, "Helvetica", 10) <= avail_first:
                return candidate  # fits on one line — no orphan possible
            # Simulate word-wrap to find last line word count
            words = candidate.split()
            lines, cur, cur_w = [], [], avail_first
            for word in words:
                ww = sw(word + " ", "Helvetica", 10)
                if cur_w - ww < 0 and cur:
                    lines.append(cur); cur = [word]; cur_w = cw - ww
                else:
                    cur.append(word); cur_w -= ww
            if cur:
                lines.append(cur)
            if not lines or len(lines[-1]) >= min_last_words:
                return candidate
        return items[0] if items else value

    # Trim skill lines to prevent orphaned words before rendering.
    skills = sections.get("skills", {})
    if isinstance(skills, dict):
        if skills.get("proficiency"):
            skills["proficiency"] = _trim_no_orphan(skills["proficiency"], "Core Competencies: ")
        if skills.get("tools"):
            skills["tools"] = _trim_no_orphan(skills["tools"], "Tools: ")

    # Phase 1: greedily maximize bullet counts to fill the page at the tightest spacing.
    sections["work_experience"] = _greedy_trim_work_experience()

    # Phase 2: binary-search for the largest spacing scale that still fits on one page.
    # Range 1.0–2.0; 8 iterations → precision ~0.004, well below any visible threshold.
    bounds = [1.0, 2.0]
    best = 1.0
    for _ in range(8):
        mid = (bounds[0] + bounds[1]) / 2
        if _page_count(mid) == 1:
            best = mid
            bounds[0] = mid
        else:
            bounds[1] = mid

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, **_DOC_KWARGS)
    doc.build(_build_story(best))
    return output_path


def render_cover_letter_pdf(body_text: str, output_path: str) -> str:
    """Render a single-page A4 cover letter PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    def s(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=11, leading=16)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    salut_s = s("Salut", alignment=TA_LEFT, spaceAfter=14)
    body_s  = s("Body", alignment=TA_JUSTIFY, spaceAfter=10)
    sign_s  = s("Sign", alignment=TA_LEFT, spaceBefore=14)

    story = []
    story.append(Paragraph("Dear the hiring manager,", salut_s))

    for para in body_text.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para.replace("\n", " "), body_s))

    story.append(Paragraph(f"Kind regards,<br/>{APPLICANT_NAME}", sign_s))

    doc.build(story)
    return output_path
