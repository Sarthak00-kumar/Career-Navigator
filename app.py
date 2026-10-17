import io
import re
import json
import html
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "indian_tech_jobs_2026.csv"

# ---------------------------------------------------------------------
# Visual system: premium dark SaaS
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --bg:#080d19; --panel:#10182a; --panel2:#141f35; --line:#25334d; --text:#eef3ff; --muted:#9aa9c4; --blue:#6ea8ff; --purple:#a78bfa; --cyan:#67e8f9; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 75% -15%, #1b2d55 0%, #0a1020 36%, #080d19 70%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,13,25,.85); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10182a,#0b1120); border-right:1px solid #202d45; }
[data-testid="stSidebar"] * { color:#eaf0ff; }
.block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:3rem; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:#f3f6ff; }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
h2 { font-size:1.55rem !important; font-weight:750 !important; }
h3 { font-size:1.12rem !important; font-weight:700 !important; }
p, li, label, .stMarkdown { color:#e1e9fa; }
small, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#151f34,#101728); border:1px solid #283752; border-radius:16px; padding:17px 18px; box-shadow:0 8px 24px #00000016; }
div[data-testid="stMetricLabel"] { color:#9eafd0 !important; font-size:.82rem; }
div[data-testid="stMetricValue"] { color:#f4f7ff !important; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stMetricDelta"] { color:#9ae6c6 !important; }
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  border-radius:11px; border:1px solid #3c5a8a; color:#f6f8ff;
  background:linear-gradient(115deg,#3158a8,#6246a8); font-weight:700;
  min-height:2.55rem; transition:transform .15s ease, filter .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(1.13); transform:translateY(-1px); border-color:#8aaeff; }
button[kind="secondary"] { background:#151f33 !important; border:1px solid #2c3b57 !important; }
input, textarea, [data-baseweb="select"] > div { background:#121b2d !important; color:#f1f5ff !important; border-color:#2b3a55 !important; border-radius:10px !important; }
[data-testid="stForm"] { background:#10182a; border:1px solid #263651; border-radius:16px; padding:18px; }
[data-testid="stExpander"] { border:1px solid #273651; border-radius:12px; background:#10182a; }
hr { border-color:#26344e !important; }
.hero {
  position:relative; overflow:hidden; padding:30px 32px; border-radius:22px;
  background:linear-gradient(112deg,rgba(31,59,111,.97),rgba(66,56,139,.92) 60%,rgba(20,112,151,.78));
  border:1px solid #5775b6; box-shadow:0 18px 48px #02061155; margin-bottom:22px;
}
.hero:after { content:""; position:absolute; width:250px; height:250px; border-radius:50%; right:-60px; top:-110px; background:radial-gradient(circle,#9cc7ff44,transparent 68%); }
.eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; color:#c8d8ff; font-weight:800; }
.hero h1 { margin:.4rem 0 .55rem; color:white !important; font-size:2.35rem !important; }
.hero p { color:#e0eaff; margin:0; font-size:1rem; max-width:700px; }
.hero-pill { display:inline-block; margin-top:18px; padding:6px 11px; border-radius:99px; background:#ffffff17; border:1px solid #ffffff2b; font-size:.78rem; color:#eef4ff; }
.section-head { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin:25px 0 12px; }
.section-head p { color:#9aa9c4; margin:0; font-size:.88rem; }
.panel { background:linear-gradient(145deg,#131d31,#0e1627); border:1px solid #263651; border-radius:17px; padding:20px; height:100%; box-shadow:0 12px 32px #0206111c; }
.panel h3 { margin-top:0; }
.panel-muted { color:#9aa9c4; font-size:.88rem; }
.big-number { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:#f4f7ff; line-height:1.2; }
.gradient-text { background:linear-gradient(90deg,#8bc4ff,#c5a7ff,#72e2ee); -webkit-background-clip:text; background-clip:text; color:transparent; }
.tag { display:inline-block; padding:5px 9px; border-radius:99px; margin:3px 4px 3px 0; font-size:.76rem; font-weight:700; border:1px solid #344766; background:#192640; color:#c8dcff; }
.tag.good { background:#12372f; border-color:#226b58; color:#8df0cb; }
.tag.warn { background:#3b2a1a; border-color:#79522b; color:#ffd39a; }
.job-card { border:1px solid #2a3a57; border-radius:16px; padding:18px; background:linear-gradient(140deg,#121d31,#0f1728); margin:10px 0; }
.job-top { display:flex; justify-content:space-between; align-items:start; gap:12px; }
.score { min-width:85px; text-align:center; border:1px solid #3f5592; border-radius:13px; padding:8px; background:linear-gradient(145deg,#1b2c55,#29234f); }
.score strong { display:block; font-size:1.35rem; color:#c9d7ff; font-family:'Manrope',sans-serif; }
.score span { font-size:.68rem; color:#9daed6; }
.mini-label { color:#8fa2c4; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:800; }
.roadmap-item { display:flex; gap:14px; padding:16px; margin:10px 0; border:1px solid #293955; border-radius:14px; background:#111a2b; }
.roadmap-num { width:36px; height:36px; flex:0 0 36px; display:grid; place-items:center; border-radius:11px; background:linear-gradient(140deg,#3158a8,#7550b6); font-weight:800; color:white; }
.cv-preview { background:#f8fafc; color:#18263e; border-radius:13px; padding:28px; border:1px solid #d9e2f0; box-shadow:0 14px 35px #0002; }
.cv-preview h2, .cv-preview h3, .cv-preview p, .cv-preview li { color:#18263e !important; }
.cv-preview h2 { font-size:1.65rem !important; margin:0; }
.cv-preview h3 { font-size:.92rem !important; color:#3157a2 !important; border-bottom:1px solid #d7e1f0; padding-bottom:5px; margin-top:20px; }
.cv-sub { color:#5f6f86 !important; font-size:.87rem; }
.empty-state { border:1px dashed #344866; border-radius:16px; padding:24px; color:#9aa9c4; background:#10182a; text-align:center; }
.foot { color:#7283a3; font-size:.76rem; text-align:center; padding-top:16px; }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#5d8eff,#a78bfa,#67e8f9); }
@media(max-width:800px) {
 .block-container { padding-left:1rem; padding-right:1rem; }
 .hero { padding:23px 20px; }
 .hero h1 { font-size:1.8rem !important; }
 .panel { padding:15px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sample fallback data. Clearly marked as illustrative.
# ---------------------------------------------------------------------
DEMO_JOBS = [
    {"job_id":"DEMO-01","job_title":"Junior Python Developer","company_name":"Northstar Digital (demo)","target_role":"Python / Software Developer","experience_level":"Entry level","location":"India / Hybrid","skills_required":"python, git, sql, api development, debugging, data structures","description_text":"Build and maintain Python applications, work with REST APIs, write tests and debug issues.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-02","job_title":"Graduate Software Engineer","company_name":"BluePeak Systems (demo)","target_role":"Python / Software Developer","experience_level":"Fresher","location":"India","skills_required":"python, javascript, git, data structures, problem solving, testing","description_text":"Entry-level engineering role involving application development, testing and code reviews.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-03","job_title":"Frontend Developer Trainee","company_name":"PixelTrail Labs (demo)","target_role":"Python / Software Developer","experience_level":"Trainee","location":"India / Remote","skills_required":"html, css, javascript, react, git, api integration","description_text":"Create responsive interfaces, integrate APIs and work with a frontend team.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-04","job_title":"Junior Data Analyst","company_name":"InsightSpring (demo)","target_role":"Data Analyst","experience_level":"Entry level","location":"India","skills_required":"sql, excel, python, pandas, data visualization, statistics","description_text":"Clean datasets, answer business questions, build reports and communicate insights.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-05","job_title":"Graduate Reporting Analyst","company_name":"Cedar Analytics (demo)","target_role":"Data Analyst","experience_level":"Fresher","location":"India / Hybrid","skills_required":"excel, sql, power bi, dashboarding, data cleaning, communication","description_text":"Prepare reports and dashboards, validate data and summarize trends for stakeholders.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-06","job_title":"Python Data Intern","company_name":"OpenField Research (demo)","target_role":"Data Analyst","experience_level":"Internship","location":"India / Remote","skills_required":"python, pandas, numpy, matplotlib, statistics, sql","description_text":"Support data preparation, exploratory analysis and visual reporting in Python.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-07","job_title":"Software Development Intern","company_name":"BrightLoop (demo)","target_role":"Python / Software Developer","experience_level":"Internship","location":"India / Remote","skills_required":"python, oop, git, sql, testing, problem solving","description_text":"Contribute to small software features, fix bugs, test code and learn team practices.","source_url":"","data_source":"Illustrative sample"},
    {"job_id":"DEMO-08","job_title":"Business Intelligence Trainee","company_name":"MetricWorks (demo)","target_role":"Data Analyst","experience_level":"Trainee","location":"India","skills_required":"sql, excel, power bi, data visualization, communication, data cleaning","description_text":"Create business reports, maintain dashboards and explain findings to teams.","source_url":"","data_source":"Illustrative sample"},
]

ALIASES = {
    "js":"javascript", "react.js":"react", "reactjs":"react", "mongo db":"mongodb",
    "powerbi":"power bi", "data visualisation":"data visualization", "apis":"api development",
    "rest api":"api development", "oops":"oop", "c++":"cpp", "ms excel":"excel",
}

TASK_LIBRARY = {
    "python": ("Strengthen Python fundamentals", "Write small functions, use lists/dictionaries, handle files, and add error handling.", "Build a command-line task tracker with input validation."),
    "git": ("Practise Git and GitHub", "Create a repository, make meaningful commits, use a branch, and write a useful README.", "Publish your next project with setup instructions and screenshots."),
    "sql": ("Practise SQL querying", "Work on SELECT, JOIN, GROUP BY, aggregate functions, and subqueries.", "Answer five questions using a small relational dataset."),
    "api development": ("Build an API-powered feature", "Call a public API and handle loading, empty, and error states.", "Add an API-powered search feature to a small web app."),
    "debugging": ("Build a debugging habit", "Reproduce a bug, isolate the cause, fix it, and record the test that verifies the fix.", "Document one bug and its fix in your project README."),
    "data structures": ("Practise core data structures", "Solve beginner problems on arrays, strings, hash maps, stacks, and queues.", "Solve and explain three problems with time complexity."),
    "javascript": ("Strengthen JavaScript", "Practise functions, arrays, objects, async/await, and error handling.", "Build a small interactive feature with clear state handling."),
    "react": ("Build a React feature", "Create reusable components, manage state, and call an API with loading/error states.", "Add a responsive search or watchlist feature to StreamFlix."),
    "html": ("Practise semantic HTML", "Build accessible pages with headings, labels, forms, and meaningful structure.", "Audit one page for semantic and accessible markup."),
    "css": ("Practise responsive CSS", "Use Flexbox/Grid and test narrow, medium, and wide viewports.", "Make one project page responsive and capture screenshots."),
    "testing": ("Add meaningful tests", "Test a normal input, invalid input, and an edge case.", "Write a short test checklist for one project feature."),
    "pandas": ("Practise pandas data handling", "Load a CSV, inspect missing values, clean columns, and produce a short summary.", "Create a reproducible data-cleaning notebook."),
    "excel": ("Practise spreadsheet analysis", "Use formulas, sorting, filtering, pivot tables, and a concise summary.", "Build a small report with three useful findings."),
    "power bi": ("Create a Power BI dashboard", "Use clear labels, appropriate visuals, filters, and a short insight summary.", "Publish screenshots and explain two decisions in the design."),
    "data visualization": ("Improve data visualization", "Choose charts that fit the question, label axes, and explain what each chart shows.", "Create three charts from a public dataset with written takeaways."),
    "statistics": ("Practise applied statistics", "Calculate mean, median, spread, and understand how outliers affect summaries.", "Write a short analysis explaining the limits of your findings."),
    "numpy": ("Practise NumPy", "Create arrays and use vectorized calculations on a small dataset.", "Compare a vectorized operation with a simple loop."),
    "matplotlib": ("Create readable charts", "Use titles, axis labels, legends, and readable scales.", "Build and explain two charts from a cleaned dataset."),
    "data cleaning": ("Practise data cleaning", "Handle missing values, duplicates, inconsistent categories, and document choices.", "Create a before/after cleaning summary."),
    "communication": ("Practise explaining your work", "Explain the problem, your contribution, the outcome, and one limitation clearly.", "Prepare a 60-second explanation of one project."),
    "mongodb": ("Review document databases", "Practise CRUD operations, filters, and document structure.", "Create a small collection and document a few example queries."),
    "oop": ("Review object-oriented programming", "Practise classes, encapsulation, composition, and readable method design.", "Refactor a small script into a few cohesive classes."),
    "problem solving": ("Practise problem solving", "Break a problem into smaller steps and test boundary cases.", "Solve three beginner problems and explain the approach."),
    "react": ("Build a React feature", "Practise components, props, state, and API integration.", "Add one polished, responsive feature to your existing project."),
}

def norm_skill(value):
    value = re.sub(r"[^a-zA-Z0-9+#. ]", " ", str(value or "").lower())
    value = re.sub(r"\s+", " ", value).strip()
    return ALIASES.get(value, value)

def parse_skills(value):
    return sorted({norm_skill(s) for s in re.split(r"[,;\n|]+", str(value or "")) if norm_skill(s)})

def esc(value):
    return html.escape(str(value or "").strip())

def find_column(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None

@st.cache_data(show_spinner=False)
def load_jobs_from_path(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame(DEMO_JOBS), "Illustrative sample data"
    try:
        raw = pd.read_csv(path, low_memory=False)
        return standardize_jobs(raw), "Bundled job-posting dataset"
    except Exception:
        return pd.DataFrame(DEMO_JOBS), "Illustrative sample data (CSV could not be parsed)"

def standardize_jobs(raw):
    aliases = {
        "job_id": ["job_id", "id"],
        "job_title": ["job_title", "title", "position", "job"],
        "company_name": ["company_name", "company", "employer"],
        "target_role": ["target_role", "role_category", "category", "role", "skill_domain"],
        "experience_level": ["experience_level", "experience_raw", "experience", "seniority", "experience_tier"],
        "location": ["location", "job_location", "city", "primary_city"],
        "skills_required": ["skills_required", "skills", "required_skills", "skill_list"],
        "description_text": ["description_text", "job_description", "description"],
        "source_url": ["source_url", "job_url", "url", "link"],
        "data_source": ["data_source", "source"],
    }
    out = pd.DataFrame(index=raw.index)
    for target, options in aliases.items():
        col = find_column(raw, options)
        out[target] = raw[col].fillna("").astype(str) if col else ""
    out["skill_list"] = out["skills_required"].apply(parse_skills)
    out["search_text"] = (
        out["job_title"] + " " + out["target_role"] + " " +
        out["skills_required"] + " " + out["description_text"]
    ).str.lower()
    out = out[(out["job_title"].str.strip() != "") & (out["skills_required"].str.strip() != "")]
    out = out.drop_duplicates(subset=["job_title", "company_name", "skills_required"]).reset_index(drop=True)
    return out

def role_filter(jobs, role):
    text = (
        jobs["job_title"].fillna("").astype(str) + " " +
        jobs["role_category"].fillna("").astype(str) + " " +
        jobs["skills_required"].fillna("").astype(str) + " " +
        jobs["job_description"].fillna("").astype(str)
    ).str.lower()

    if role == "Data Analyst":
        mask = text.str.contains(
            r"data analyst|business intelligence|reporting analyst|analytics|data engineer|data intern|business analyst",
            regex=True,
            na=False
        )
    else:
        mask = text.str.contains(
            r"software|python|developer|web application|frontend|front end engineering intern|react",
            regex=True,
            na=False
        )

    return jobs[mask].copy() if mask.any() else jobs.copy()
def match_jobs(profile, jobs, role):
    if jobs.empty:
        return pd.DataFrame()
    subset = role_filter(jobs, role)
    if subset.empty:
        subset = jobs.copy()
    profile_text = " ".join([
        role, profile.get("skills", ""), profile.get("projects", ""),
        profile.get("interests", ""), profile.get("objective", "")
    ])
    corpus = subset["search_text"].fillna("").tolist() + [profile_text]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        scores = [0.0] * len(subset)
    subset["similarity"] = (scores * 100).round().astype(int)
    have = set(parse_skills(profile.get("skills", "")))
    subset["matched"] = subset["skill_list"].apply(lambda x: sorted(set(x) & have))
    subset["missing"] = subset["skill_list"].apply(lambda x: sorted(set(x) - have))
    subset["overlap_pct"] = subset.apply(
        lambda r: round(100 * len(r["matched"]) / max(1, len(r["skill_list"]))), axis=1
    )
    return subset.sort_values(["similarity", "overlap_pct"], ascending=False).head(10).reset_index(drop=True)

def build_roadmap(gaps, role):
    items = []
    for skill in gaps[:5]:
        title, action, project = TASK_LIBRARY.get(
            skill, (f"Learn and practise {skill}", f"Study the fundamentals of {skill}, complete 2–3 exercises, and record evidence of your work.", f"Use {skill} in a small role-relevant project.")
        )
        items.append({
            "title": title,
            "why": f"'{skill}' appeared in the skill lists of relevant job-posting matches and was not listed in your profile.",
            "action": action,
            "deliverable": project,
            "skill": skill,
        })
    items.append({
        "title": "Turn learning into project evidence",
        "why": "A demonstrable project helps reviewers understand how you applied your skills.",
        "action": f"Choose a small project aligned with your target role ({role}); document the problem, tools, your contribution, and limitations.",
        "deliverable": "A working repository with README, screenshots, and truthful project details.",
        "skill": "project",
    })
    items.append({
        "title": "Prepare a role-specific application",
        "why": "Clear evidence and concise explanations make your application easier to review.",
        "action": "Update your CV with skills you can demonstrate, add project links, and practise explaining one technical decision.",
        "deliverable": "One reviewed CV draft and three practice interview answers.",
        "skill": "application",
    })
    return items

def run_career_agent(gaps, role, done_tasks):
    """A transparent, rule-based agent loop; no LLM or external API is used.

    Observe: current skill gaps and completed milestones.
    Decide: select the highest-priority incomplete milestone.
    Act: return a concrete task and deliverable.
    Reassess: called again after progress changes.
    """
    tasks = build_roadmap(gaps, role)
    completed = [t for t in tasks if done_tasks.get(t["title"], False)]
    remaining = [t for t in tasks if not done_tasks.get(t["title"], False)]
    if not remaining:
        return {
            "status": "complete",
            "observation": f"All {len(tasks)} current milestones are marked complete.",
            "decision": "Move to a fresh matching run or review the evidence you created.",
            "action": "Review your CV and project repository, then rerun job matching after updating your skills.",
            "deliverable": "Updated profile and reviewed project/CV evidence.",
            "next_task": None,
            "completed_count": len(completed),
            "total_count": len(tasks),
        }
    next_task = remaining[0]
    return {
        "status": "active",
        "observation": f"{len(gaps)} skill gaps are being tracked; {len(completed)} of {len(tasks)} milestones are complete.",
        "decision": f"Prioritise '{next_task['skill']}' because it is the first incomplete item in the current evidence-based plan.",
        "action": next_task["action"],
        "deliverable": next_task["deliverable"],
        "next_task": next_task["title"],
        "completed_count": len(completed),
        "total_count": len(tasks),
    }

def make_resume_pdf(profile, resume_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Name", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#18335c"), alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="Contact", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#53647d"), spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionX", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#3157a2"), spaceBefore=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9.3, leading=13, textColor=colors.HexColor("#243249"), spaceAfter=4))
    story = []
    name = resume_data.get("name") or "Your Name"
    contact = " | ".join([x for x in [resume_data.get("email"), resume_data.get("location")] if x])
    story += [Paragraph(esc(name), styles["Name"]), Paragraph(esc(contact), styles["Contact"]), HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#7999d5")), Spacer(1, 5)]
    def section(title, body):
        story.append(Paragraph(title.upper(), styles["SectionX"]))
        safe_body = esc(body).replace("\n", "<br/>")
        story.append(Paragraph(safe_body or "Add verified details before using this CV.", styles["BodyX"]))
    section("Profile", resume_data.get("objective", ""))
    section("Education", f'{profile.get("degree","")}<br/>{profile.get("college","")}')
    section("Skills", ", ".join(parse_skills(resume_data.get("skills", ""))))
    section("Projects", resume_data.get("projects", ""))
    section("Target role", profile.get("role", ""))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Review this draft carefully. Include only skills, experience, and achievements you can honestly demonstrate.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
defaults = {
    "profile": {
        "name": "", "email": "", "degree": "B.Tech Computer Science Engineering",
        "college": "Lovely Professional University", "location": "India",
        "role": "Python / Software Developer", "skills": "", "projects": "",
        "interests": "web development, problem solving", "objective": ""
    },
    "matched_jobs": None,
    "gaps": [],
    "done_tasks": {},
    "resume": {},
    "uploaded_jobs": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, dict) else value

# ---------------------------------------------------------------------
# Navigation callbacks: update the radio value before the next rerun.
# This avoids Streamlit's "widget already instantiated" session-state error.
# ---------------------------------------------------------------------
def go_to_page(target):
    st.session_state["workspace_nav"] = target

if "workspace_nav" not in st.session_state:
    st.session_state["workspace_nav"] = "Overview"

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Career Navigator")
    st.markdown('<div class="panel-muted">Your next step, made clearer.</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Overview", "Find opportunities", "Career roadmap", "CV studio", "My progress", "Data settings"],
        key="workspace_nav",
        label_visibility="collapsed",
    )
    st.markdown("---")
    p = st.session_state.profile
    st.markdown('<div class="mini-label">YOUR CAREER TARGET</div>', unsafe_allow_html=True)
    st.markdown(f"**{esc(p.get('role','Choose a role'))}**")
    skill_count = len(parse_skills(p.get("skills", "")))
    st.caption(f"{skill_count} skills currently listed")
    if st.button("Edit my profile", use_container_width=True, key="edit_profile_btn"):
        st.session_state["show_profile_editor"] = True
        st.rerun()
    st.markdown("---")
    st.caption("SDG 8 · Decent Work and Economic Growth")
    st.caption("Prototype · Job similarity is not a hiring probability.")

jobs, data_label = load_jobs_from_path(str(DATA_PATH))
if st.session_state.uploaded_jobs is not None:
    try:
        jobs = standardize_jobs(st.session_state.uploaded_jobs)
        data_label = "User-uploaded dataset"
    except Exception:
        pass

# ---------------------------------------------------------------------
# Header / profile editor
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">CAREER INTELLIGENCE · SDG 8</div>
  <h1>Make your next move <span style="color:#b9c9ff">count.</span></h1>
  <p>Turn your current skills into a practical plan for your first technology role. Explore job-posting matches, identify skill gaps, and build evidence of your progress.</p>
  <span class="hero-pill">✦ ML-assisted matching &nbsp; · &nbsp; Actionable roadmap &nbsp; · &nbsp; CV studio</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("show_profile_editor", False):
    with st.expander("Your career profile", expanded=True):
        with st.form("profile_form", border=False):
            current = st.session_state.profile
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full name", current["name"], placeholder="Your name")
                email = st.text_input("Email (optional)", current["email"], placeholder="you@example.com")
                degree = st.text_input("Degree / branch", current["degree"])
                college = st.text_input("College / university", current["college"])
                location = st.text_input("Preferred location", current["location"])
            with c2:
                role = st.selectbox("Target role", ["Python / Software Developer", "Data Analyst"], index=0 if current["role"] == "Python / Software Developer" else 1)
                skills_text = st.text_area("Skills you can currently demonstrate", current["skills"], placeholder="Python, SQL, Git, React...", height=95)
                projects = st.text_area("Projects and experience", current["projects"], placeholder="Project name, tools, and what you built...", height=95)
                interests = st.text_input("Career interests", current["interests"])
                objective = st.text_input("Career objective (optional)", current["objective"])
            submitted = st.form_submit_button("Save profile", type="primary", use_container_width=True)
            if submitted:
                st.session_state.profile = {"name":name.strip(),"email":email.strip(),"degree":degree.strip(),"college":college.strip(),"location":location.strip(),"role":role,"skills":skills_text.strip(),"projects":projects.strip(),"interests":interests.strip(),"objective":objective.strip()}
                st.session_state["show_profile_editor"] = False
                st.session_state.matched_jobs = None
                st.session_state.gaps = []
                st.success("Profile updated. Run matching again to refresh your recommendations.")
                st.rerun()

p = st.session_state.profile
have = parse_skills(p.get("skills", ""))

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    matched = st.session_state.matched_jobs
    col1.metric("Roles analysed", f"{len(matched) if matched is not None else 0}", help="Ranked results from the selected job dataset.")
    col2.metric("Skills listed", len(have))
    col3.metric("Skill gaps found", len(st.session_state.gaps))
    total_tasks = len(build_roadmap(st.session_state.gaps, p["role"])) if st.session_state.gaps else 0
    completed = sum(bool(st.session_state.done_tasks.get(t["title"])) for t in build_roadmap(st.session_state.gaps, p["role"]))
    col4.metric("Roadmap progress", f"{round(100*completed/max(1,total_tasks)) if total_tasks else 0}%")

    st.markdown('<div class="section-head"><div><h2>Welcome back. Let’s build momentum.</h2><p>A focused workspace for your first-role preparation.</p></div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="mini-label">YOUR NEXT MILESTONE</div><h3 style="margin-top:10px">Find the skills that matter for your target role.</h3><p class="panel-muted">Compare your profile with job-posting text, see which skills overlap, and identify the most repeated missing skills.</p></div>', unsafe_allow_html=True)
        if st.button("Analyse my profile →", type="primary", use_container_width=True, key="overview_match"):
            if not p.get("skills","").strip() and not p.get("projects","").strip():
                st.warning("Add at least some skills or project details first. Use “Edit my profile” in the sidebar.")
            else:
                with st.spinner("Comparing your profile with job-posting text…"):
                    result = match_jobs(p, jobs, p["role"])
                    st.session_state.matched_jobs = result
                    gaps = []
                    for _, row in result.head(10).iterrows():
                        gaps.extend(row["missing"])
                    from collections import Counter
                    counts = Counter(gaps)
                    st.session_state.gaps = [s for s, _ in counts.most_common(8)]
                st.success("Analysis complete. Open Find opportunities or Career roadmap to explore the results.")
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="mini-label">HOW IT WORKS</div><h3 style="margin-top:10px">From profile to action</h3><p class="panel-muted">① Describe what you can demonstrate<br>② Rank related job-posting text with TF-IDF + cosine similarity<br>③ Compare parsed skill lists transparently<br>④ Turn repeated gaps into learning actions</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="mini-label">PROFILE SNAPSHOT</div><h3 style="margin-top:10px">Your career direction</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-number gradient-text">{esc(p["role"])}</div><p class="panel-muted">{esc(p["degree"])} · {esc(p["college"])}</p>', unsafe_allow_html=True)
        if have:
            st.markdown("".join(f'<span class="tag">{esc(s)}</span>' for s in have[:12]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">Add your current skills to personalise recommendations.</div>', unsafe_allow_html=True)
        if p.get("projects"):
            st.markdown('<div class="mini-label" style="margin-top:15px">PROJECT EVIDENCE</div>', unsafe_allow_html=True)
            st.write(p["projects"][:240] + ("…" if len(p["projects"]) > 240 else ""))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><h2>Explore your workspace</h2><p>Each area is designed to answer one practical career question.</p></div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><div class="mini-label">01 · DISCOVER</div><h3 style="margin-top:10px">Find opportunities</h3><p class="panel-muted">See ranked role matches, overlap skills, and skills mentioned but not listed in your profile.</p></div>', unsafe_allow_html=True)
        st.button("Explore job matches", use_container_width=True, on_click=go_to_page, args=("Find opportunities",), key="go_matches")
    with c2:
        st.markdown('<div class="panel"><div class="mini-label">02 · BUILD</div><h3 style="margin-top:10px">Career roadmap</h3><p class="panel-muted">Turn skill gaps into bite-sized learning actions and project deliverables.</p></div>', unsafe_allow_html=True)
        st.button("View my roadmap", use_container_width=True, on_click=go_to_page, args=("Career roadmap",), key="go_roadmap")
    with c3:
        st.markdown('<div class="panel"><div class="mini-label">03 · PRESENT</div><h3 style="margin-top:10px">CV studio</h3><p class="panel-muted">Create a clean CV draft and export it as a PDF for further editing.</p></div>', unsafe_allow_html=True)
        st.button("Open CV studio", use_container_width=True, on_click=go_to_page, args=("CV studio",), key="go_cv")
    st.caption(f"Dataset status: {data_label} · {len(jobs):,} usable rows after cleaning. Data quality and vacancy status should be independently verified.")

# ---------------------------------------------------------------------
# Find opportunities
# ---------------------------------------------------------------------
if page == "Find opportunities":
    st.markdown('<div class="section-head"><div><h2>Opportunities aligned with your profile</h2><p>Text similarity ranks related postings; it does not estimate hiring probability.</p></div></div>', unsafe_allow_html=True)
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown('<div class="panel"><div class="mini-label">MATCHING METHOD</div><h3 style="margin-top:8px">TF-IDF + cosine similarity</h3><p class="panel-muted">The model represents job and profile text as weighted terms, then ranks their textual similarity. Skill overlap and missing-skill lists are calculated separately from the parsed skills column.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="mini-label">CURRENT TARGET</div><h3 style="margin-top:8px">'+esc(p["role"])+'</h3><p class="panel-muted">'+str(len(jobs))+' usable records · '+esc(data_label)+'</p></div>', unsafe_allow_html=True)
    if st.button("Run job matching & skill-gap analysis", type="primary", use_container_width=True):
        if not p.get("skills","").strip() and not p.get("projects","").strip():
            st.warning("Add skills or project details using Edit my profile in the sidebar.")
        else:
            with st.spinner("Ranking job-posting text…"):
                result = match_jobs(p, jobs, p["role"])
                st.session_state.matched_jobs = result
                from collections import Counter
                all_gaps = []
                for _, row in result.head(10).iterrows():
                    all_gaps.extend(row["missing"])
                counts = Counter(all_gaps)
                st.session_state.gaps = [s for s, _ in counts.most_common(8)]
            st.rerun()
    result = st.session_state.matched_jobs
    if result is not None and not result.empty:
        st.markdown(f'<div class="section-head"><div><h2>Top matches <span class="gradient-text">{len(result)}</span></h2><p>Sorted by text similarity, then parsed skill overlap.</p></div></div>', unsafe_allow_html=True)
        avg = int(result["similarity"].mean())
        c1,c2,c3 = st.columns(3)
        c1.metric("Results shown", len(result))
        c2.metric("Average text similarity", f"{avg}%")
        c3.metric("Repeated skill gaps", len(st.session_state.gaps))
        for _, row in result.iterrows():
            title = esc(row["job_title"])
            company = esc(row["company_name"] or "Company not specified")
            experience = esc(row["experience_level"] or "Experience not specified")
            location = esc(row["location"] or "Location not specified")
            score = int(row["similarity"])
            st.markdown(f"""
            <div class="job-card">
              <div class="job-top"><div><div class="mini-label">{esc(row['target_role'] or p['role'])}</div><h3 style="margin:6px 0 4px">{title}</h3><div class="panel-muted">{company} &nbsp;·&nbsp; {experience} &nbsp;·&nbsp; {location}</div></div>
              <div class="score"><strong>{score}%</strong><span>TEXT SIMILARITY</span></div></div>
            </div>""", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Skills overlapping your profile**")
                st.markdown("".join(f'<span class="tag good">{esc(s)}</span>' for s in row["matched"]) if row["matched"] else '<span class="panel-muted">No parsed skill overlap found.</span>', unsafe_allow_html=True)
            with c2:
                st.markdown("**Skills to review**")
                st.markdown("".join(f'<span class="tag warn">{esc(s)}</span>' for s in row["missing"][:10]) if row["missing"] else '<span class="panel-muted">All parsed skills are listed; verify your evidence.</span>', unsafe_allow_html=True)
            with st.expander("Role description and source"):
                st.write(row["description_text"] or "No description text available in this record.")
                if str(row["source_url"]).startswith(("https://","http://")):
                    st.link_button("Open source listing", row["source_url"])
                else:
                    st.caption("No source URL is available for this record. Verify vacancy status independently.")
        st.markdown('<div class="section-head"><div><h2>Most repeated missing skills</h2><p>Derived from the top ranked results, not from an external skills benchmark.</p></div></div>', unsafe_allow_html=True)
        if st.session_state.gaps:
            st.markdown("".join(f'<span class="tag warn">{esc(s)}</span>' for s in st.session_state.gaps), unsafe_allow_html=True)
        else:
            st.info("No repeated missing skills were detected from the parsed records.")
    else:
        st.markdown('<div class="empty-state"><h3>Ready when you are</h3><p>Add your current skills in the profile, then run the analysis to see ranked results here.</p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Career roadmap
# ---------------------------------------------------------------------
if page == "Career roadmap":
    st.markdown('<div class="section-head"><div><h2>Your action plan + career agent</h2><p>A transparent rule-based agent observes your gaps, chooses a next action, and reassesses when progress changes. No LLM/API is used.</p></div></div>', unsafe_allow_html=True)
    gaps = st.session_state.gaps
    if not gaps:
        starter = ["python","git","api development","testing"] if "Developer" in p["role"] else ["sql","excel","python","data visualization"]
        gaps = [s for s in starter if s not in set(have)] or starter[:2]
        st.info("This is a starter roadmap because matching has not been run yet. Run matching for dataset-derived skill suggestions.")
    tasks = build_roadmap(gaps, p["role"])
    done = sum(bool(st.session_state.done_tasks.get(t["title"])) for t in tasks)
    agent = run_career_agent(gaps, p["role"], st.session_state.done_tasks)
    st.markdown('<div class="panel"><div class="mini-label">CAREER AGENT · OBSERVE → DECIDE → ACT → REASSESS</div>', unsafe_allow_html=True)
    st.markdown(f"**Observe:** {esc(agent['observation'])}")
    st.markdown(f"**Decide:** {esc(agent['decision'])}")
    st.markdown(f"**Next action:** {esc(agent['action'])}")
    st.markdown(f"**Expected deliverable:** {esc(agent['deliverable'])}")
    st.caption("The agent uses explicit rules and your current progress. Mark a task complete below; the app reruns and selects the next incomplete action.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="mini-label">YOUR MOMENTUM</div><div class="big-number gradient-text">{round(100*done/max(1,len(tasks)))}%</div><p class="panel-muted">{done} of {len(tasks)} milestones completed · progress is stored in this browser session only.</p></div>', unsafe_allow_html=True)
    st.progress(done / max(1, len(tasks)))
    for i, task in enumerate(tasks, start=1):
        c1, c2 = st.columns([0.88,0.12])
        with c1:
            st.markdown(f'<div class="roadmap-item"><div class="roadmap-num">{i:02d}</div><div><div class="mini-label">{esc(task["skill"])}</div><h3 style="margin:4px 0 8px">{esc(task["title"])}</h3><p class="panel-muted"><b>Why:</b> {esc(task["why"])}</p><p>{esc(task["action"])}</p><p><span style="color:#a9baff;font-weight:700">Deliverable:</span> {esc(task["deliverable"])}</p></div></div>', unsafe_allow_html=True)
        with c2:
            st.write("")
            new_val = st.toggle("Done", key=f"roadmap_done_{i}_{task['title']}", value=bool(st.session_state.done_tasks.get(task["title"])))
            st.session_state.done_tasks[task["title"]] = new_val
    st.caption("Agent limitation: this is a lightweight rule-based agent, not an LLM-powered agent. It does not browse the web, enrol you in courses, or guarantee employment. Progress is session-based.")

# ---------------------------------------------------------------------
# CV studio
# ---------------------------------------------------------------------
if page == "CV studio":
    st.markdown('<div class="section-head"><div><h2>CV studio</h2><p>Draft a clean, truthful CV from details you provide. Review every line before applying.</p></div></div>', unsafe_allow_html=True)
    old = st.session_state.resume
    with st.form("resume_form", border=False):
        c1,c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", old.get("name", p["name"]))
            email = st.text_input("Email", old.get("email", p["email"]))
            location = st.text_input("Location", old.get("location", p["location"]))
            objective = st.text_area("Professional profile / objective", old.get("objective", p["objective"] or f"{p['role']} candidate studying {p['degree']}. Interested in applying practical skills to build reliable, useful software."), height=110)
        with c2:
            skills_text = st.text_area("Skills you can demonstrate", old.get("skills", p["skills"]), height=110)
            projects_text = st.text_area("Projects (include tools and your contribution)", old.get("projects", p["projects"]), height=145)
            links = st.text_input("Portfolio / GitHub link (optional)", old.get("links", ""))
        save_resume = st.form_submit_button("Update CV preview", type="primary", use_container_width=True)
        if save_resume:
            st.session_state.resume = {"name":name.strip(),"email":email.strip(),"location":location.strip(),"objective":objective.strip(),"skills":skills_text.strip(),"projects":projects_text.strip(),"links":links.strip()}
            st.rerun()
    data = st.session_state.resume or {
        "name":p["name"], "email":p["email"], "location":p["location"],
        "objective":p["objective"] or f"{p['role']} candidate studying {p['degree']}.",
        "skills":p["skills"], "projects":p["projects"], "links":""
    }
    contact = " | ".join(x for x in [data.get("email"), data.get("location"), data.get("links")] if x)
    st.markdown('<div class="section-head"><div><h2>Live preview</h2><p>Preview content below will be exported to a PDF.</p></div></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cv-preview">
      <h2>{esc(data.get('name') or 'Your Name')}</h2>
      <p class="cv-sub">{esc(contact)}</p>
      <h3>PROFILE</h3><p>{esc(data.get('objective'))}</p>
      <h3>EDUCATION</h3><p><b>{esc(p.get('degree'))}</b><br/>{esc(p.get('college'))}</p>
      <h3>SKILLS</h3><p>{esc(', '.join(parse_skills(data.get('skills',''))))}</p>
      <h3>PROJECTS</h3><p>{esc(data.get('projects') or 'Add project details and your contribution.')}</p>
      <h3>TARGET ROLE</h3><p>{esc(p.get('role'))}</p>
    </div>
    """, unsafe_allow_html=True)
    pdf_bytes = make_resume_pdf(p, data)
    st.download_button("⬇ Download CV as PDF", data=pdf_bytes, file_name="AI_Career_Navigator_CV.pdf", mime="application/pdf", type="primary", use_container_width=True)
    st.caption("The PDF is a draft. Add dates, verifiable achievements, project links, and any missing sections before using it for applications.")

# ---------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------
if page == "My progress":
    st.markdown('<div class="section-head"><div><h2>Your progress snapshot</h2><p>Track completed actions and export a small progress report.</p></div></div>', unsafe_allow_html=True)
    gaps = st.session_state.gaps or (["python","git"] if "Developer" in p["role"] else ["sql","excel"])
    tasks = build_roadmap(gaps, p["role"])
    done = sum(bool(st.session_state.done_tasks.get(t["title"])) for t in tasks)
    c1,c2,c3 = st.columns(3)
    c1.metric("Completed milestones", f"{done}/{len(tasks)}")
    c2.metric("Completion", f"{round(100*done/max(1,len(tasks)))}%")
    c3.metric("Skills to review", len(st.session_state.gaps))
    st.progress(done/max(1,len(tasks)))
    for task in tasks:
        completed = bool(st.session_state.done_tasks.get(task["title"]))
        st.markdown(f'<div class="panel" style="padding:13px 16px;margin:8px 0"><span style="color:{"#83edc3" if completed else "#8da1c4"};font-weight:800">{"✓ COMPLETED" if completed else "○ IN PROGRESS"}</span><br/><b>{esc(task["title"])}</b><div class="panel-muted">{esc(task["deliverable"])}</div></div>', unsafe_allow_html=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "target_role": p["role"],
        "skills_listed": have,
        "skill_gaps_from_matching": st.session_state.gaps,
        "completed_tasks": [t["title"] for t in tasks if st.session_state.done_tasks.get(t["title"])],
        "total_tasks": len(tasks),
        "completion_percent": round(100*done/max(1,len(tasks))),
        "storage_note": "Prototype state is session-based; this file is an export, not a permanent student account."
    }
    st.download_button("Export progress summary (JSON)", json.dumps(summary, indent=2), file_name="career_progress_summary.json", mime="application/json")
    st.caption("Progress is not currently saved to a permanent student account or database.")

# ---------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------
if page == "Data settings":
    st.markdown('<div class="section-head"><div><h2>Data & transparency</h2><p>Inspect the dataset used by the prototype and optionally upload a compatible CSV.</p></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Usable records", f"{len(jobs):,}")
    c2.metric("Distinct job titles", f"{jobs['job_title'].nunique():,}" if not jobs.empty else "0")
    c3.metric("Records with URLs", f"{jobs['source_url'].astype(str).str.startswith(('http://','https://')).sum():,}" if not jobs.empty else "0")
    st.markdown('<div class="panel"><h3>Current data source</h3><p class="panel-muted">The bundled CSV is a job-posting dataset intended for demonstration and analysis. Some records may be duplicated, stale, mislabeled, or no longer available. The app does not verify active vacancies, and matching results are not hiring probabilities.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Upload a job-posting CSV")
    uploaded = st.file_uploader("CSV with job title and skills columns", type=["csv"], help="Recognised column names include job_title/title, skills_required/skills, job_description/description, job_url/source_url.")
    if uploaded is not None:
        try:
            candidate = pd.read_csv(uploaded, low_memory=False)
            normalized = standardize_jobs(candidate)
            if normalized.empty:
                st.error("The file was read, but no rows with both a job title and skill list were found.")
            else:
                st.session_state.uploaded_jobs = candidate
                st.success(f"Loaded {len(normalized):,} usable rows for this session. Run matching again to use them.")
                st.dataframe(normalized[["job_title","target_role","experience_level","location","skills_required"]].head(10), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    st.markdown("#### Preview")
    if not jobs.empty:
        st.dataframe(jobs[["job_title","target_role","experience_level","location","skills_required"]].head(12), use_container_width=True, hide_index=True)
    st.caption(f"Source label: {data_label}. Uploads are held in the current app session and are not written back to the repository.")

st.markdown("---")
st.markdown('<div class="foot">AI Career Navigator · SDG 8: Decent Work and Economic Growth · Educational prototype, not a hiring service. Verify job details and include only truthful CV information.</div>', unsafe_allow_html=True)
