# AI Career Navigator

A Streamlit prototype for **SDG 8: Decent Work and Economic Growth**. It helps students and fresh graduates explore job-posting text, identify possible skill gaps, plan learning actions, draft a CV, and track progress.

## Features

- Premium dark SaaS-style responsive interface.
- Profile editor for target role, skills, projects, and career interests.
- ML-assisted job ranking using **TF-IDF + cosine similarity**.
- Separate, transparent comparison of parsed skill lists to show overlaps and possible gaps.
- A learning roadmap based on repeated missing skills, with deliverables and progress toggles.
- CV preview and downloadable PDF draft.
- Progress summary export as JSON.
- Dataset preview and optional CSV upload.
- Honest limitations shown in the UI: text similarity is not a hiring probability; vacancy status is not verified; progress is session-based.

## Project structure

```text
AI_Career_Navigator_Pro/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── indian_tech_jobs_2026.csv
```

## Run locally on macOS

Use Python 3.11 or 3.12 if possible.

1. Open Terminal and enter the project directory:

   ```bash
   cd path/to/AI_Career_Navigator_Pro
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Upgrade pip and install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL printed in Terminal, usually `http://localhost:8501`.

If your virtual environment already exists, activate it first. If you see `No module named streamlit`, install packages using the activated environment's `python -m pip install -r requirements.txt`.

## Deploy publicly with Streamlit Community Cloud

1. Create a GitHub repository, for example `ai-career-navigator`.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `data/` folder with its CSV.
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Click **Deploy** and wait for the build to finish.
6. Open the public app URL and test profile editing, job matching, roadmap toggles, CV PDF export, and progress JSON export.

Deployment may take a few minutes. If the build fails, inspect the Streamlit logs and confirm that every dependency used in `app.py` is listed in `requirements.txt`.

## GitHub upload from Terminal

Run these commands from the project folder after creating an empty GitHub repository. Replace the URL with your own repository URL.

```bash
git init
git add app.py requirements.txt README.md .gitignore data/
git commit -m "Build AI Career Navigator prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-career-navigator.git
git push -u origin main
```

Do not commit `.venv/`, credentials, or private user information. This project currently needs no API key.

## How the matching works

1. Job title, role category, required skills, and description are combined into a text document.
2. The student's target role, listed skills, project details, interests, and objective are combined into a profile document.
3. `TfidfVectorizer` converts documents into weighted term vectors.
4. `cosine_similarity` ranks job descriptions by text similarity to the profile.
5. Parsed skill names are compared separately to show overlap and possible missing skills.
6. Repeated missing skills from the ranked results inform a starter roadmap.

The similarity score is **not** a probability of being hired, a validated employability score, or a guarantee. The skill comparison is only as good as the job data and the skills written in the profile.

## Dataset caution

The bundled job-posting CSV is included for demonstration and experimentation. Its records may be duplicated, stale, inconsistent, or not currently active. Verify each listing independently before applying. If you do not have permission to redistribute the original dataset, remove it from the public repository and instead document where users can obtain it lawfully; the app can fall back to clearly labelled illustrative sample records if the CSV is absent.

## Current limitations / future improvements

- Progress and profile data live in Streamlit session state; there is no permanent account, database, or cross-device sync.
- The roadmap is a transparent template-driven task planner, not a connected autonomous agent.
- Resume content is user-provided and needs review; the PDF is a draft, not a guarantee of ATS compatibility.
- The dataset is not live job inventory. A future version could add validated sources, freshness checks, authentication, database persistence, and a genuinely tool-using agent workflow.
