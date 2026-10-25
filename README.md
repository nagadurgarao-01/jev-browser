# Jev Browser ⚡

**A browser agent with a dynamic, indexed action space, memory, and fast-path macro replay.**

Give it one goal. [TypeSafe's Jev](https://docs.typesafe.ai/introduction) picks an operation and an element. A small LLM writes text only when the operation is `TYPE_TEXT`. Memory stores proven navigation patterns and replays them at 25 ms — skipping the model call entirely.

**Zürich → London on Google Flights in ~7 seconds.** One natural-language goal, actual text generation, and loading waits included.

[Read the loop](jev_browser/agent.py) · [Chat interface](examples/chat_interface.py)

## Features

| Phase | What it does |
|---|---|
| **Core agent** | Dynamic action space: `CLICK`, `TYPE_TEXT`, `SELECT`, `WAIT`, `DONE`, `BLOCKED` |
| **Phase 1** | Human-in-the-loop — detects OTP / 2FA / CAPTCHA and pauses for you |
| **Phase 2** | Structured data extraction — summary cards, key-value grids, and tables on task completion |
| **Phase 3** | Force-directed knowledge graph — visual entity-relation explorer in the chat UI |
| **Phase 4** | Fast-path macro replay — proven search patterns replayed at ⚡ 25 ms, no LLM call |

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. Only supported operations and targets are offered.

```text
                      one TypeSafe request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome.

## Try it

```bash
git clone https://github.com/nagadurgarao-01/jev-browser.git
cd jev-browser
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run python examples/chat_interface.py
```

Open **http://127.0.0.1:8777** — type any task in the chat box and watch it execute live.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from jev_browser import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel's incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.

## Memory & Fast-Path Macro Replay

Every completed task is distilled into a reusable search pattern stored in [`jev_browser/memory.py`](jev_browser/memory.py).

- **BM25 + Knowledge Graph** — personal info, credentials, and entity relations are stored and searched at query time for `TYPE_TEXT` field filling.
- **Pattern Matching** — Jaccard token overlap ≥ 0.65 identifies a prior successful navigation sequence for the same domain and goal.
- **Fast-Path Execution** — matched macro steps run at **⚡ 25 ms** with no TypeSafe round-trip. If any element is missing or the page changes, the agent drops out gracefully and resumes normal LLM-driven decisions.

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** Jev consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **One browser call per snapshot.** Read visible controls, their names, values, and text atomically. Keep references to the actual DOM nodes.
- **Validate the selected target.** Clicks check the document, form values, target, and nearby context. Animation alone does not force another prediction.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node. The executor rechecks page freshness and click occlusion. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](jev_browser/agent.py) | The complete loop, text-helper handoff, and fast-path macro replay |
| [memory.py](jev_browser/memory.py) | BM25 search, knowledge graph, task history, and pattern distillation |
| [snapshot.js](jev_browser/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](jev_browser/browser.py) | Browser connection, current geometry, execution |
| [model.py](jev_browser/model.py) | Dynamic operation/target heads and text generation |
| [questions.py](jev_browser/questions.py) | Model instructions |
| [examples/chat_interface.py](examples/chat_interface.py) | Chat UI server with memory, history, and extraction endpoints |
| [examples/chat.html](examples/chat.html) | Chat UI with knowledge graph, step traces, and fast-path badges |

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_browser/static/app.js
node --check jev_browser/snapshot.js
uv build
```

Tests are 100% offline — 47 passing, zero paid API calls. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls.

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
