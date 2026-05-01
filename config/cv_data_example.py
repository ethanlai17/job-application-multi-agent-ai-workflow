APPLICANT_NAME = "Your Name"
APPLICANT_TITLE = "Your Job Title (e.g. Product Manager)"

CONTACT = "Your Phone | your.email@example.com | linkedin.com/in/your-profile | github.com/your-github | Your City, Country"

EDUCATION = [
    {
        "title": "Your Degree - Your University",
        "detail": "Relevant coursework or achievements",
    },
]

PROJECTS = [
    {
        "name": "Your Project Name",
        "subtitle": "Your Role",
        "dates": "Month Year",
        "bullets": [
            "Brief description of what you built and the outcome",
        ],
    },
]

# Fixed job title/company/dates — the LLM only controls bullet selection, never these fields.
# First 3 are mandatory; anything beyond index 2 is optional (included only if it adds JD value).
WORK_EXPERIENCE_FIXED = [
    {"title": "Your Most Recent Job Title", "company": "Company A", "dates": "Month Year – Present"},
    {"title": "Previous Job Title",         "company": "Company B", "dates": "Month Year – Month Year"},
    {"title": "Earlier Job Title",          "company": "Company C", "dates": "Month Year – Month Year"},
]

# Add extra bullet points here that are real experience not yet in your CV PDF.
EXTRA_BULLETS = []
