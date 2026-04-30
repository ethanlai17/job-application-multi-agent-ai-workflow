import json
import re
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from rich import print as rprint


def _clean_bullet(text: str) -> str:
    """Collapse irregular whitespace in bullet text to single spaces."""
    return re.sub(r"[ \t]+", " ", text).strip()

# ── Fixed CV content (never modified by LLM) ─────────────────────────────────

_CONTACT = (
    "(+44)7466675957 | ethanlaipm@gmail.com | "
    "linkedin.com/in/ethan-lai17 | github.com/ethanlai17 | London, UK"
)

_EDUCATION = [
    {
        "title": "MSc. Information Systems (IS) - The University of Sheffield",
        "detail": (
            "Postgraduate merit scholarship. "
            "Coursework: User-centred design, database design, IS modelling, Python"
        ),
    },
    {
        "title": "Erasmus+ exchange scholarship - University of Tampere, Finland",
        "detail": (
            "Coursework: Project management, Leadership, "
            "Strategic management, Management of change"
        ),
    },
    {
        "title": "B.A Business Administration (Distinction) - University of Economics HCMC",
        "detail": "",
    },
]

_NEW_PROJECT = {
    "name": "Job Application AI Multi-Agent Workflow",
    "subtitle": "Vibe Coding",
    "dates": "Apr 2026",
    "bullets": [
        (
            "Built a multi-agent Python system orchestrating LinkedIn job search, "
            "Google Sheets tracking, tailored CV/cover letter generation, and automated "
            "form filling via Playwright and Groq LLM"
        ),
        (
            "Designed a human-in-the-loop pipeline: agents search and generate documents, "
            "user reviews and approves roles in Google Sheets, then agents auto-apply"
        ),
    ],
}

_LANGNOTE_PROJECT = {
    "name": "LangNote",
    "subtitle": "Product Manager",
    "dates": "May - Dec 2021",
    "bullets": [
        "Shaped the product vision and conducted qualitative & quantitative market research to identify target audience",
        "Developed a user pilot plan including A/B tests on hypotheses of nice-to-have features",
    ],
}

# Fixed job title/company/dates — the LLM only controls bullet selection, never these fields.
_WORK_EXPERIENCE_FIXED = [
    {"title": "Senior Technical Product Manager", "company": "TUI UK",        "dates": "Oct 2022 – Present"},
    {"title": "Technical Product Manager",        "company": "Inspectorio",   "dates": "Dec 2020 – Feb 2022"},
    {"title": "Technical Business Analyst",       "company": "FPT Software",  "dates": "Jan – Dec 2020"},
    {"title": "Account Manager",                  "company": "Carousell",     "dates": "Jul – Dec 2019"},
]

# Verified extra bullets — real experience not yet written into the original CV PDF.
# The LLM may use these exactly as written when the JD matches the described skill.
_EXTRA_BULLETS = [
    {
        "company": "TUI UK",
        "text": (
            "Defined conversion, bookability, and API latency as needle-moving KPIs at TUI UK, "
            "applying a data-driven approach to product ownership to translate metric insights "
            "into engineering priorities"
        ),
        "use_when": (
            "JD explicitly mentions data-driven product ownership, defining or tracking metrics, "
            "conversion/bookability/latency, or performance KPIs"
        ),
    },
]


def _extra_bullets_block() -> str:
    if not _EXTRA_BULLETS:
        return ""
    lines = [
        "\nADDITIONAL VERIFIED BULLETS (confirmed real experience — not yet in original CV PDF):",
        "Use these EXACTLY as written when the JD calls for the corresponding skill.",
    ]
    for b in _EXTRA_BULLETS:
        lines.append(f'  {b["company"]}: "{b["text"]}"')
        lines.append(f'  [Include when: {b["use_when"]}]')
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
            "   - Include ALL relevant skills from the JD, even if not in Ethan's CV.\n"
        )
    kw_list = ", ".join(popular_keywords)
    return (
        f"   - APPROVED KEYWORD LIST (market-popular skills + every skill explicitly required in this JD):\n"
        f"     {kw_list}\n"
        f"   - Only include skills/keywords from this approved list OR from Ethan's original CV or ADDITIONAL VERIFIED BULLETS.\n"
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
    prompt = f"""You are an expert CV writer tailoring Ethan Lai's CV for a specific job.

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

1. work_experience — MUST include the first 3 jobs in this EXACT order with these EXACT titles:
   1. "Senior Technical Product Manager" at "TUI UK",  Oct 2022 – Present
   2. "Technical Product Manager" at "Inspectorio",    Dec 2020 – Feb 2022
   3. "Technical Business Analyst" at "FPT Software",  Jan – Dec 2020
   The 4th job "Account Manager" at "Carousell" (Jul – Dec 2019) is OPTIONAL:
   include it ONLY if it adds clear value for this specific JD.
   If TUI UK and Inspectorio bullets already cover the JD requirements comprehensively,
   OMIT Carousell and give TUI UK and Inspectorio 4–5 bullets each instead.
   - Select 3–5 bullets per job that best match the JD.
   - Keep each bullet to at most 2 lines. If a bullet wraps to a 2nd line, ensure the last line
     has at least 5 words — never leave a single word or very short phrase alone on the last line.
     Trim or rephrase the end of any bullet that would otherwise orphan a word.
   - ALWAYS preserve the original metric/outcome (€ amount, %, hours saved, etc.) — never strip numbers.
   - Keep the original sentence structure; you may:
       • swap a word for a JD keyword (e.g. "iterative" → "test-learn-iterate", "data" → "data-driven")
       • insert a JD keyword phrase mid-sentence (e.g. "…via Continuous Discovery", "…using Agile sprints")
   - NEVER append "…demonstrating/reflecting/showcasing/highlighting X" at the end — forbidden.
   - Do NOT truncate bullets to just an action verb without context or metric.
   - Do NOT invent metrics, companies, or outcomes not in the original CV.
   - Do NOT fabricate bullets — only use or adapt bullets from the original CV or ADDITIONAL VERIFIED BULLETS below.
     Carousell has EXACTLY 2 bullets in the original. Do not add a third.
   - Only weave in a JD keyword if it fits naturally. If it sounds forced, leave that bullet unchanged.
   - Target style: "Drove €1.6M revenue by launching global booking fee across 5 microservices via test-learn-iterate"
                   "Designed AI-powered personalisation engine via Continuous Discovery, boosting CTR 15% & conversion 3%"

2. projects — include EXACTLY these 2 projects in this order (use verbatim):
   First: {{"name": "Job Application AI Multi-Agent Workflow", "subtitle": "Vibe Coding", "dates": "Apr 2026",
     "bullets": [
       "Built a multi-agent Python system orchestrating LinkedIn job search, Google Sheets tracking, tailored CV/cover letter generation, and automated form filling via Playwright and Groq LLM",
       "Designed a human-in-the-loop pipeline: agents search and generate documents, user reviews and approves roles in Google Sheets, then agents auto-apply"
     ]}}
   Second: {{"name": "LangNote", "subtitle": "Product Manager", "dates": "May - Dec 2021",
     "bullets": [
       "Shaped the product vision and conducted qualitative & quantitative market research to identify target audience",
       "Developed a user pilot plan including A/B tests on hypotheses of nice-to-have features"
     ]}}

3. skills — this is CRITICAL:
   - Read the FULL job description carefully and extract relevant skills, methodologies, frameworks, and tools.
   - Merge with Ethan's existing skills, placing JD-matched skills first.
   - "proficiency" = core competencies: methodologies, frameworks, soft skills, PM skills (rendered as "Core Competencies" on the CV — an ATS-standard label)
   - "tools" = named software tools and platforms from both the JD and Ethan's original CV
   - Keep each value (proficiency / tools) to ONE compact comma-separated line — no repetition, no padding.
     Limit to 5–7 items each; stop adding items before the line would orphan a single word on the next line.
     Rule of thumb: proficiency ≤ 100 characters, tools ≤ 85 characters.
   - Spell out abbreviations on first use:
       write "Objectives and Key Results (OKRs)" not just "OKRs"
       write "Key Performance Indicators (KPIs)" not just "KPIs"
       write "Jobs to Be Done (JTBD)" not just "JTBD"
   - Avoid vague language: never write "various", "several", "multiple", or "etc."
     Replace with specific counts or named examples from the original CV.
{_popular_keywords_rule(popular_keywords)}
4. Return ONLY the JSON — nothing else.

ETHAN'S ORIGINAL CV:
{cv_text}
{_extra_bullets_block()}

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
    # First 3 jobs are mandatory; Carousell (4th) is optional — only added if LLM included it.
    jobs = result.get("work_experience", [])
    for i, fixed in enumerate(_WORK_EXPERIENCE_FIXED):
        if i < len(jobs):
            jobs[i]["title"] = fixed["title"]
            jobs[i]["company"] = fixed["company"]
            jobs[i]["dates"] = fixed["dates"]
        elif i < 3:
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
        "You are writing a cover letter body for Ethan Lai, a Technical Product Manager. "
        "Write exactly 3 paragraphs, total under 280 words, in a formal but human style. "
        "Focus on specific measurable impacts from Ethan's experience and how they "
        "match what this company and role need. "
        "Do NOT include: a salutation (no 'Dear'), sign-off, subject line, or date — "
        "those are added separately. "
        "Do NOT use clichés like 'I am writing to express my interest' or "
        "'I am a passionate'. "
        "Open with what specifically draws Ethan to this company and role. "
        "Middle paragraph: 2-3 specific achievements with metrics that directly address the JD. "
        "Closing: concise statement of contribution and confidence.\n\n"
        f"ETHAN'S CV:\n{cv_text}"
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
        max_tokens=800,
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
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

    _DOC_KWARGS = dict(
        pagesize=A4, leftMargin=38, rightMargin=38, topMargin=28, bottomMargin=24,
    )

    def _build_story(scale: float) -> list:
        """Build the platypus story with vertical spacing multiplied by scale.

        ATS compliance:
        - All body text >= 10pt (ATS minimum)
        - No Table elements — title rows use a single Paragraph with a right-aligned tab stop
        - Standard section headings (WORK EXPERIENCE, PROJECTS, EDUCATION, SKILLS)
        - Helvetica font (ATS-approved)
        - Justified bullets for clean text extraction
        """

        def s(name, **kw):
            # Base defaults: 10pt body, 12pt leading (ATS minimum font size)
            defaults = dict(fontName="Helvetica", fontSize=10, leading=12 * scale)
            defaults.update(kw)
            return ParagraphStyle(name, **defaults)

        # Usable content width inside the page margins
        content_w = A4[0] - 38 - 38

        # Header — fixed, not scaled
        name_s    = s("Name",    fontName="Helvetica-Bold", fontSize=16, leading=20,
                      alignment=TA_CENTER, spaceAfter=1)
        contact_s = s("Contact", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=3)

        # Scaled body styles — all >= 10pt per ATS guidelines
        section_s = s("Section", fontName="Helvetica-Bold", fontSize=11, leading=13,
                      spaceBefore=5 * scale, spaceAfter=1 * scale)
        # Title row: bold title left, dates right — same line via tab stop (no Table needed)
        title_row_s = s("TitleRow", fontName="Helvetica-Bold", fontSize=10,
                        leading=12 * scale, spaceAfter=1 * scale,
                        tabStops=[(content_w, TA_RIGHT)])
        # Hanging indent: • at the left margin, tab stop at 12pt snaps text to the continuation
        # column — so the first-line text start and all wrapped lines are at exactly the same x.
        # Tab (&#9;) is a fixed-width stop, so justification never stretches the gap after •.
        bullet_s  = s("Bullet",  fontSize=10, leading=12 * scale, leftIndent=12,
                      firstLineIndent=-12, spaceAfter=0.5 * scale, alignment=TA_LEFT,
                      tabStops=[(12, TA_LEFT)])
        edu_s     = s("Edu",     fontName="Helvetica-Bold", fontSize=10, leading=12 * scale)
        edu_det_s = s("EduDet",  fontSize=10, leading=12 * scale, leftIndent=12,
                      firstLineIndent=-12, spaceAfter=2 * scale, alignment=TA_LEFT,
                      tabStops=[(12, TA_LEFT)])

        story = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph("ETHAN LAI", name_s))
        story.append(Paragraph(_CONTACT, contact_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black,
                                spaceAfter=3 * scale))

        def section_header(text):
            story.append(Paragraph(text, section_s))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black,
                                    spaceAfter=2 * scale))

        def title_row(label, dates):
            # Single Paragraph with right-aligned tab stop — title left, dates right on the same line.
            # ATS-friendly (no Table) and reads naturally in linear text extraction.
            story.append(Paragraph(
                f'<b>{label}</b>&#9;<font name="Helvetica">{dates}</font>',
                title_row_s,
            ))

        def add_bullets(bullets):
            for b in bullets:
                story.append(Paragraph(f"•&#9;{b}", bullet_s))

        # ── Work Experience ───────────────────────────────────────────────────
        section_header("WORK EXPERIENCE")
        for job in sections.get("work_experience", []):
            label = f"{job['title']} - {job['company']}"
            title_row(label, job.get("dates", ""))
            add_bullets(job.get("bullets", []))
            story.append(Spacer(1, 3 * scale))

        # ── Projects ─────────────────────────────────────────────────────────
        section_header("PROJECTS")
        for proj in sections.get("projects", [_NEW_PROJECT]):
            label = proj["name"]
            if proj.get("subtitle"):
                label = f"{proj['subtitle']} - {proj['name']}"
            title_row(label, proj.get("dates", ""))
            add_bullets(proj.get("bullets", []))
            story.append(Spacer(1, 3 * scale))

        # ── Education (fixed) ────────────────────────────────────────────────
        section_header("EDUCATION")
        for edu in _EDUCATION:
            story.append(Paragraph(edu["title"], edu_s))
            if edu["detail"]:
                story.append(Paragraph(f"•&#9;{edu['detail']}", edu_det_s))
            else:
                story.append(Spacer(1, 4 * scale))

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

    def _page_count(scale: float) -> int:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, **_DOC_KWARGS)
        doc.build(_build_story(scale))
        buf.seek(0)
        pdf = fitz.open(stream=buf.read(), filetype="pdf")
        n = pdf.page_count
        pdf.close()
        return n

    # Binary-search for the largest scale that still fits on one page.
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

    story.append(Paragraph("Kind regards,<br/>Ethan Lai", sign_s))

    doc.build(story)
    return output_path
