# Jev Browser ⚡

> A fast, memory-powered browser agent that executes plain-English tasks in real Chrome tabs.

Give it one goal in plain English. It observes the live page, picks the right operation and element, and acts — no selectors, no scripts, no coordinates hardcoded. After a successful run, it remembers the navigation pattern and replays it at **⚡ 25 ms** on the next visit — skipping the model call entirely.

---

## Features

| | What it does |
|---|---|
| 🧠 **Smart memory** | Saves personal info, credentials, and search patterns — auto-fills fields from context |
| ⚡ **Fast-path replay** | Replays proven navigation sequences at 25 ms — no LLM round-trip |
| 🔐 **2FA / OTP handoff** | Detects login walls, CAPTCHAs, and OTP screens — pauses and lets you take over |
| 📊 **Extraction cards** | Summarises the final page into structured key-value facts and tables |
| 🕸️ **Knowledge graph** | Visualises entities and relations from memory as an interactive force-directed graph |
| 🌐 **Chat interface** | Full chat UI at `http://127.0.0.1:8777` — type a task, watch it execute live |

---

## Quick Start

```bash
git clone https://github.com/nagadurgarao-01/jev-browser.git
cd jev-browser
uv sync
cp .env.example .env
```

Edit `.env` and add your keys:

```env
TYPESAFE_API_KEY=your_key_here
TEXT_MODEL_API_KEY=your_key_here
```

Start the chat interface:

```bash
uv run python examples/chat_interface.py
```

Open **http://127.0.0.1:8777** and type any task — e.g.:

- `search for wireless headphones on amazon`
- `find one-way flights from Zurich to London on google flights`
- `open the Alan Turing article on Wikipedia`

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if Chrome isn't detected.

---

## How It Works

### The Action Space

Every page observation produces an indexed element table:

```
[1] button    Change ticket type  ·  Round trip
[2] combobox  Where from?         ·  empty
[3] combobox  Where to?           ·  empty
[4] textbox   Departure           ·  empty
```

One TypeSafe request picks the **operation** and **target** simultaneously:

```
page → element table ──► operation head      ┐
                         click_target head   ├─ one network call
                         type_text_target    ┘
                              │
              CLICK [2] ──────┤──► browser
          TYPE_TEXT [3] ──────┘
                    │
             small LLM ──► typed text ──► browser
```

- **`CLICK`**, **`TYPE_TEXT`**, **`SELECT`**, **`WAIT`**, **`DONE`**, **`BLOCKED`**
- Only compatible operations and targets are offered
- Text is never hardcoded — a small LLM generates it from context
- Model output never becomes selectors, JS, or shell commands

### Memory & Fast-Path Macro Replay

After every successful task, the agent distills the action sequence into a reusable **search pattern**:

1. **BM25 search** — personal facts, credentials, and entity relations are retrieved at query time to fill `TYPE_TEXT` fields
2. **Pattern matching** — Jaccard token overlap ≥ 0.65 identifies a prior successful run for the same domain and goal
3. **Fast-path execution** — matched steps execute at **25 ms** with no model call; if any element is missing the agent drops out gracefully and resumes normal LLM decisions

### Human-in-the-Loop

If a login wall, OTP screen, or CAPTCHA is detected mid-run, the agent:
- Pauses and shows an amber intervention banner
- Accepts an OTP code you type or waits for you to complete it in the browser
- Resumes automatically once you click **Resume**

---

## Use as a Library

```python
from jev_browser import Agent
from jev_browser.memory import get_memory

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026.",
    memory=get_memory(),
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run a custom task from the CLI:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org \
  --goal "Open the article about Alan Turing"
```

---

## Project Structure

```
jev_browser/
├── agent.py          ← Full agent loop, fast-path macro replay, 2FA handoff
├── memory.py         ← BM25 search, knowledge graph, task history, pattern distillation
├── model.py          ← Dynamic operation/target heads, text generation
├── browser.py        ← Chrome connection, DOM snapshot, execution
├── snapshot.js       ← Atomic DOM reader, indexed controls, freshness guards
├── questions.py      ← Model prompt instructions
└── static/           ← Inspector UI assets

examples/
├── chat_interface.py ← HTTP server: chat, memory, history, extraction endpoints
├── chat.html         ← Chat UI with knowledge graph, fast-path badges, step traces
└── flights.py        ← Flights demo with outcome verification
```

---

## Development

```bash
uv run ruff check .          # lint
uv run pytest                # 47 tests, 100% offline, zero paid API calls
node --check jev_browser/static/app.js
node --check jev_browser/snapshot.js
uv build
```

---

## Limits

- Shadow DOM, iframes, canvas, file uploads, and pop-up tabs are outside the current scope
- `DONE` requires independent outcome verification
- DOM reader covers common HTML and ARIA controls, not the full accessible-name spec
- Owned tabs share the existing Chrome profile

---

## Roadmap

- [x] Phase 1 — Human-in-the-loop / 2FA handoff
- [x] Phase 2 — Structured data extraction cards
- [x] Phase 3 — Knowledge graph visualiser
- [x] Phase 4 — Fast-path macro replay
- [ ] Phase 5 — Auto-dismiss cookie banners & popup tab handler

---

Built on [TypeSafe](https://docs.typesafe.ai) · [Browser Harness](https://github.com/browser-use/browser-harness) · [Browser Use](https://github.com/browser-use/browser-use)
