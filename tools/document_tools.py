import json
import re
from pathlib import Path

from openai import OpenAI

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
            "Coursework: User-centred design (UI/UX), Database design, IS modelling, Python"
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
    "dates": "Apr 2025",
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
        f"   - APPROVED KEYWORD LIST (from market-wide keyword library — appear in 2+ JDs):\n"
        f"     {kw_list}\n"
        f"   - Only include skills/keywords from this approved list OR from Ethan's original CV.\n"
        f"   - Do NOT add niche, company-specific jargon from this JD that is not on the approved list\n"
        f"     (e.g. one-off phrases like 'thinking in bets', 'thin/vertical slicing', 'test-learn-iterate'\n"
        f"     should only appear if they are already in the approved list above).\n"
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

1. work_experience — include ALL 4 jobs (TUI UK, Inspectorio, FPT Software, Carousell), newest first.
   - Select 3–5 bullets per job that best match the JD.
   - ALWAYS preserve the original metric/outcome (€ amount, %, hours saved, etc.) — never strip numbers.
   - Keep the original sentence structure; you may:
       • swap a word for a JD keyword (e.g. "iterative" → "test-learn-iterate", "data" → "data-driven")
       • insert a JD keyword phrase mid-sentence (e.g. "…via Continuous Discovery", "…using Agile sprints")
   - NEVER append "…demonstrating/reflecting/showcasing/highlighting X" at the end — forbidden.
   - Do NOT truncate bullets to just an action verb without context or metric.
   - Do NOT invent metrics, companies, or outcomes not in the original CV.
   - Do NOT fabricate bullets — only use or adapt bullets that literally exist in the original CV.
     Carousell has EXACTLY 2 bullets in the original. Do not add a third.
   - Only weave in a JD keyword if it fits naturally. If it sounds forced, leave that bullet unchanged.
   - Target style: "Drove €1.6M revenue by launching global booking fee across 5 microservices via test-learn-iterate"
                   "Designed AI-powered personalisation engine via Continuous Discovery, boosting CTR 15% & conversion 3%"

2. projects — include EXACTLY these 2 projects in this order (use verbatim):
   First: {{"name": "Job Application AI Multi-Agent Workflow", "subtitle": "Vibe Coding", "dates": "Apr 2025",
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
   - "proficiency" = methodologies, frameworks, soft skills, PM competencies
   - "tools" = named software tools and platforms from both the JD and Ethan's original CV
   - Keep each value (proficiency / tools) to ONE compact comma-separated line — no repetition, no padding
{_popular_keywords_rule(popular_keywords)}
4. Return ONLY the JSON — nothing else.

ETHAN'S ORIGINAL CV:
{cv_text}

TARGET ROLE:
Title: {job_title}
Company: {company}

Full Job Description:
{job_description}"""

    response = client.chat.completions.create(
        model=model,
        max_tokens=3500,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown fences."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps with them
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def generate_cover_letter(
    cv_text: str,
    job_description: str,
    job_title: str,
    company: str,
    client: OpenAI,
    model: str,
) -> str:
    """Ask the LLM for cover letter body paragraphs only (no salutation/sign-off)."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": (
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
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Write the cover letter body for:\n"
                    f"Job Title: {job_title}\n"
                    f"Company: {company}\n\n"
                    f"Job Description:\n{job_description[:2000]}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


# ── PDF rendering ─────────────────────────────────────────────────────────────

def render_cv_pdf(sections: dict, output_path: str) -> str:
    """Render a professional single-page A4 CV PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=28,
        bottomMargin=24,
    )

    W = A4[0] - 76  # usable content width

    def s(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=8.8, leading=10.5)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    name_s    = s("Name", fontName="Helvetica-Bold", fontSize=15, leading=19,
                  alignment=TA_CENTER, spaceAfter=1)
    contact_s = s("Contact", fontSize=7.8, leading=9.5, alignment=TA_CENTER, spaceAfter=3)
    section_s = s("Section", fontName="Helvetica-Bold", fontSize=9.2, leading=11,
                  spaceBefore=4, spaceAfter=1)
    title_s   = s("Title", fontName="Helvetica-Bold", fontSize=8.8, leading=10.5)
    dates_s   = s("Dates", fontSize=8.2, leading=10.5, alignment=TA_RIGHT)
    bullet_s  = s("Bullet", fontSize=8.2, leading=10.5, leftIndent=9, spaceAfter=0.3)
    edu_s     = s("Edu", fontName="Helvetica-Bold", fontSize=8.8, leading=10.5)
    edu_det_s = s("EduDet", fontSize=8.2, leading=10.5, leftIndent=9, spaceAfter=1.5)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("ETHAN LAI", name_s))
    story.append(Paragraph(_CONTACT, contact_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=3))

    def section_header(text):
        story.append(Paragraph(text, section_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=2))

    _table_style = TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ALIGN",         (1, 0), (1, 0),   "RIGHT"),
    ])

    def title_row(label, dates):
        t = Table(
            [[Paragraph(f"<b>{label}</b>", title_s), Paragraph(dates, dates_s)]],
            colWidths=[W * 0.73, W * 0.27],
            hAlign="LEFT",
        )
        t.setStyle(_table_style)
        story.append(t)

    def add_bullets(bullets):
        for b in bullets:
            story.append(Paragraph(f"• {b}", bullet_s))

    # ── Work Experience ───────────────────────────────────────────────────────
    section_header("WORK EXPERIENCE")
    for job in sections.get("work_experience", []):
        label = f"{job['title']} - {job['company']}"
        title_row(label, job.get("dates", ""))
        add_bullets(job.get("bullets", []))
        story.append(Spacer(1, 2))

    # ── Project Experience ────────────────────────────────────────────────────
    section_header("PROJECT EXPERIENCE")
    for proj in sections.get("projects", _NEW_PROJECT):
        label = proj["name"]
        if proj.get("subtitle"):
            label = f"{proj['subtitle']} - {proj['name']}"
        title_row(label, proj.get("dates", ""))
        add_bullets(proj.get("bullets", []))
        story.append(Spacer(1, 2))

    # ── Education (fixed) ────────────────────────────────────────────────────
    section_header("EDUCATION")
    for edu in _EDUCATION:
        story.append(Paragraph(edu["title"], edu_s))
        if edu["detail"]:
            story.append(Paragraph(f"• {edu['detail']}", edu_det_s))
        else:
            story.append(Spacer(1, 3))

    # ── Skills ────────────────────────────────────────────────────────────────
    section_header("SKILLS")
    skills = sections.get("skills", {})
    if isinstance(skills, dict):
        if skills.get("proficiency"):
            story.append(Paragraph(
                f"• <b>Proficiency:</b> {skills['proficiency']}", bullet_s))
        if skills.get("tools"):
            story.append(Paragraph(
                f"• <b>Tools:</b> {skills['tools']}", bullet_s))
    elif isinstance(skills, list):
        add_bullets(skills)

    doc.build(story)
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
