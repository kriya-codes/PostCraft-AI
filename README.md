# ✦ PostCraft AI

**Turn one idea into platform-ready social media content.**

PostCraft AI takes a single piece of text you provide and uses Google's Gemini AI to
reshape it into posts written for each platform you pick. Instead of copy-pasting the
same paragraph into LinkedIn, X, and a blog, you get a purpose-built version of your
idea for each one — then edit, copy, or regenerate any card individually.

---

## ✨ Features

- **Multi-platform generation** — create posts for several platforms from one idea
- **LinkedIn** posts with platform-appropriate length, structure, and hashtags
- **X (Twitter)** posts held to a hard **280-character limit**
  - If Gemini overshoots, the app automatically asks Gemini to shorten it once, then
    verifies the result before returning it
- **Dev.to / Medium** style content — a short blog-style introduction
- **Six writing tones** — Professional, Casual, Friendly, Storytelling, Educational,
  Motivational
- **Platform-specific AI prompts** — every platform gets its own detailed writing
  instructions, not one generic prompt
- **Edit** any generated post before copying
- **Copy** a single post, or **Copy all** successful posts at once
- **Live character and word counts** on every card
- **Individual regeneration** — regenerate one platform without touching the others
- **Partial results and error handling** — if one platform fails, the others still
  succeed, and the failed card shows a clear message with a **Retry** button
- **Secure server-side Gemini integration** — the API key never reaches the browser
- **Responsive frontend** — works on desktop, tablet, and mobile

---

## 🛠 Tech Stack

| Layer | Technology |
| --- | --- |
| Structure | HTML5 |
| Styling | CSS3 (responsive, no framework) |
| Logic | Vanilla JavaScript (no framework, no build step) |
| Language | Python |
| Web framework | Flask |
| AI SDK | Google GenAI SDK (`google-genai`) |
| AI model | Gemini API |
| Secrets | `python-dotenv` |

There is no database, no authentication, and no frontend build step.

---

## 🔄 How It Works

```text
User enters an idea
        ↓
Selects platforms
        ↓
Selects tone
        ↓
Flask backend
        ↓
Gemini AI
        ↓
Platform-specific content
        ↓
Edit / Copy / Regenerate
```

In simple terms: the browser collects your idea, your chosen platforms, and your tone,
then sends them to the Flask backend. The backend reads the API key from `.env`, asks
Gemini to write a post for **each** selected platform using that platform's own prompt
rules, and returns the results. The browser then shows one card per platform, where you
can edit, copy, or regenerate that post on its own.

Each platform is handled independently — one failure never cancels the others.

---

## 📁 Project Structure

```text
PostCraft-AI/
├── app.py                 # Flask routes, validation, error handling
├── gemini_service.py      # Gemini prompts, API calls, X length enforcement
├── templates/
│   └── index.html         # Page markup
├── static/
│   ├── css/
│   │   └── style.css      # All styling, including responsive breakpoints
│   └── js/
│       └── script.js      # Frontend logic: generate, copy, edit, regenerate
├── tests.py               # Automated test suite
├── requirements.txt       # Python dependencies
├── .env.example           # Template for your API key
├── .gitignore
└── README.md
```

`.env` is created by you during setup and is deliberately **not** committed.

---

## 🚀 Installation

Windows-friendly steps:

```bash
git clone https://github.com/kriya-codes/PostCraft-AI.git
cd PostCraft-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Create your `.env` file

Get your key from [Google AI Studio](https://aistudio.google.com/apikey), then create a
file named `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Optionally, override the model (this is genuinely supported by the app):

```env
GEMINI_MODEL=gemini-3.8-flash
```

If you leave `GEMINI_MODEL` out, the app uses its built-in default.

> ⚠️ **Never commit `.env`, and never put the Gemini API key in frontend code.**
> `.env` is already listed in `.gitignore`, so Git will not track it. The key is read
> only by the Python backend and is never sent to the browser.

---

## ▶️ Run the Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The app runs on Flask's development server, which is intended for local use.

---

## 🧪 Testing

The project ships with an automated test suite:

```bash
python -m unittest
```

Expected result:

```text
Ran 125 tests in ...
OK (skipped=7)
```

**No live Gemini API call is required.** Gemini is replaced with a test double, so the
suite runs offline, quickly, and without spending quota.

What the suite covers:

- Input validation for every field
- Successful generation and response shape
- **Partial platform failure** — one platform failing while others succeed
- **All platforms failing** — a single controlled error, never a crash
- The `/api/regenerate` endpoint, including failure and retry
- The X 280-character limit and its shortening retry
- Per-platform prompt and tone rules
- **Security and API-key exposure checks** — the suite asserts the key never appears in
  any HTML page, CSS/JS asset, API response, server log, or error message
- Frontend wiring checks for copy, edit, counts, and error states

The 7 skipped tests are **opt-in live tests**. They only run when you explicitly set an
environment variable and supply your own working key:

```bash
set RUN_LIVE_TESTS=1
python -m unittest
```

---

## 🔐 Security

- The Gemini API key is stored in `.env`, which is **ignored by Git**.
- The key is read **only by the Python backend**.
- The key is **never sent to frontend code** — no key appears in the HTML, CSS, or
  JavaScript served to the browser.
- API responses return a safe, human-readable message; internal details and SDK errors
  stay in the server log.
- The test suite actively verifies the key is not exposed in the page, static assets,
  API responses, logs, or error text.

If you ever commit a real key by accident, treat it as compromised and revoke it in
Google AI Studio immediately.

---

## 🔮 Limitations & Future Improvements

These are ideas, **not** current features.

- Accept an article or URL as input instead of pasted text
- Save drafts and generation history
- User accounts and login
- Publish directly to social media
- Schedule posts
- Engagement and performance analytics
- More output platforms (Instagram, Facebook, Threads, newsletters)
- Custom writing styles saved by the user
- Multiple input ideas in one request

**Current limitations**

- Content is kept in memory for the current page only — refreshing clears it
- No history, drafts, or accounts
- The Dev.to / Medium output is an introduction to a post, not a full article
- `python app.py` uses Flask's development server, not a production WSGI server

---

## 📌 Project Status

**Status: Working MVP**

The core functionality is implemented and tested: multi-platform generation, per-platform
prompt rules, X length enforcement, tone selection, editing, copying, individual
regeneration, partial-failure and error handling, secure server-side API integration, a
responsive interface, and an automated test suite that runs offline.

---

## 👤 Author

**Kriya Parmar**

---

<div align="center">
  <sub>Built with Flask and Gemini AI ✦</sub>
</div>
