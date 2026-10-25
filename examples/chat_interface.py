"""Interactive Chat Interface for Jev browser."""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from jev_browser import Agent
from jev_browser.demo import load_environment
from jev_browser.memory import get_memory
from jev_browser.model import post_json

PORT = int(os.environ.get("JEV_CHAT_PORT", "8777"))
ORIGIN = f"http://127.0.0.1:{PORT}"
HTML_FILE = Path(__file__).with_name("chat.html")

LOCK = threading.Lock()
STOP_EVENT = threading.Event()
ACTIVE_AGENT = None
SESSION_STATE = {
    "running": False,
    "current_step": None,
    "page": None,
    "decision": None,
    "elements": [],
    "elapsed_ms": 0,
    "history": [],
    "human_intervention": None,
    "messages": [
        {
            "role": "assistant",
            "content": (
                "Hi! I am **Jev browser** ⚡. Send me any task in plain English "
                "(e.g., search Google, look up Wikipedia, or check flights) and I will "
                "execute it live in the browser at sub-second speed!"
            ),
            "steps": [],
            "url": None,
            "status": "idle",
        }
    ],
}


def resolve_task(user_text: str) -> tuple[str, str]:
    text = user_text.strip()
    match = re.search(r"https?://[^\s]+", text)
    if match:
        url = match.group(0).rstrip(".,;!?:")
        goal = text.replace(url, "").strip() or f"Navigate and explore {url}"
        return url, goal
    # Dynamically resolve starting URL and natural-language goal using the text LLM
    key = os.environ.get("TEXT_MODEL_API_KEY")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    if key:
        prompt = (
            "You are a URL and goal resolver for a web browser agent.\n"
            "Given any user prompt, return a JSON object with:\n"
            "1. 'url': The absolute starting URL (beginning with https://). "
            "If no specific site is specified, default to 'https://www.google.com'.\n"
            "2. 'goal': The concise natural-language goal for the agent to achieve once on that website. "
            "If the task requires interacting with a post, article, comment, or result in a feed or list that "
            "may require scrolling down to find, phrase the goal to scroll down and perform the action "
            "(e.g. 'Scroll down and like the first post'). "
            "If the task involves typing an email recipient, phrase the goal to select the recipient contact from the "
            "suggestion dropdown to confirm it.\n"
            'Output JSON only: {"url": "...", "goal": "..."}'
        )
        try:
            r = post_json(
                base + "/chat/completions",
                key,
                {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                },
            )
            data = json.loads(r["choices"][0]["message"]["content"])
            resolved_url = data.get("url", "").strip()
            resolved_goal = data.get("goal", "").strip()
            if resolved_url.startswith("http://") or resolved_url.startswith("https://"):
                return resolved_url, resolved_goal or text
        except Exception:
            pass

    return "https://www.google.com", text


def extract_structured_data(goal: str, page_title: str, final_url: str, page_text: str) -> dict | None:
    """Extract structured data (summary, facts/metrics, data table) from final page state."""
    key = os.environ.get("TEXT_MODEL_API_KEY")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    if not key or not page_text:
        return None

    # Truncate page text for extraction prompt
    snippet = page_text[:4000]
    prompt = (
        "You are a structured data extractor for a web browser agent.\n"
        "Given the user's original task goal, the final page title, URL, and visible page text, "
        "extract relevant answers, facts, metrics, prices, specifications, or flight/product details.\n"
        "Return a JSON object with:\n"
        "1. 'title': Short descriptive title of the result (e.g. 'Sony WH-1000XM5 Specs & Price')\n"
        "2. 'summary': 1-3 sentences answering the user goal based on visible text\n"
        "3. 'attributes': Key-value dictionary of extracted metrics/facts "
        "(e.g. {'Price': '₹24,990', 'Rating': '4.5 ★'})\n"
        "4. 'table': Optional list of row objects for tabular data (e.g. flight options, comparisons), or null\n"
        "If the page has no meaningful data for the goal, return null in JSON.\n"
        'Output valid JSON only: {"title": "...", "summary": "...", "attributes": {...}, "table": [...] or null}'
    )
    user_payload = json.dumps({
        "goal": goal,
        "page_title": page_title,
        "url": final_url,
        "visible_text": snippet,
    })
    try:
        r = post_json(
            base + "/chat/completions",
            key,
            {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_payload},
                ],
            },
        )
        content = r["choices"][0]["message"]["content"]
        data = json.loads(content)
        if isinstance(data, dict) and data.get("title") and (data.get("summary") or data.get("attributes")):
            return data
    except Exception:
        pass
    return None


def run_agent_task(prompt: str, starting_url: str, goal: str, memory_enabled: bool = True):
    global SESSION_STATE
    STOP_EVENT.clear()
    assistant_msg = {
        "role": "assistant",
        "content": f"Opening **{starting_url}** with goal: *{goal}*",
        "steps": [],
        "url": starting_url,
        "status": "running",
        "elapsed_ms": 0,
    }
    with LOCK:
        SESSION_STATE["messages"].append(assistant_msg)
        SESSION_STATE["running"] = True
        SESSION_STATE["current_step"] = "Starting browser..."
        SESSION_STATE["page"] = None

    global ACTIVE_AGENT
    try:
        mem_instance = get_memory() if memory_enabled else None
        with Agent(starting_url, goal, screenshots=True, memory=mem_instance) as agent:
            with LOCK:
                ACTIVE_AGENT = agent
            for state in agent.run():
                if STOP_EVENT.is_set():
                    break
                with LOCK:
                    SESSION_STATE["elapsed_ms"] = state["elapsed_ms"]
                    SESSION_STATE["page"] = state["page"]
                    SESSION_STATE["human_intervention"] = state.get("human_intervention")
                    decision = state.get("decision")
                    if not decision and state.get("decisions"):
                        decision = state["decisions"][-1]
                    SESSION_STATE["decision"] = decision
                    try:
                        snap = agent.snapshot()
                        SESSION_STATE["elements"] = snap.get("elements", [])
                    except Exception:
                        pass
                    history = state.get("history", [])
                    assistant_msg["steps"] = history
                    assistant_msg["elapsed_ms"] = state["elapsed_ms"]
                    if state.get("page"):
                        assistant_msg["url"] = state["page"]["url"]

                    if state["status"] == "waiting_for_human":
                        reason = state.get("human_intervention", {}).get("reason", "Verification needed")
                        SESSION_STATE["current_step"] = f"Paused: {reason}. Complete in browser or submit code."
                    elif history:
                        last = history[-1]
                        txt = f' typed "{last["text"]}"' if last.get("text") else ""
                        SESSION_STATE["current_step"] = f'Step {last["step"]}: {last["action"]}{txt}'
                    else:
                        SESSION_STATE["current_step"] = "Observing page & predicting next action..."

                    if state["status"] in {"done", "blocked"}:
                        break

            final_state = agent.snapshot()
            if STOP_EVENT.is_set():
                final_status = "stopped"
            else:
                final_status = final_state.get("status", "done")
            final_url = final_state.get("page", {}).get("url", starting_url)
            page_title = final_state.get("page", {}).get("title") or "Search Results"
            total_actions = len(final_state.get("history", []))
            total_time = final_state.get("elapsed_ms", 0) / 1000

            with LOCK:
                decision = final_state.get("decision")
                if not decision and final_state.get("decisions"):
                    decision = final_state["decisions"][-1]
                SESSION_STATE["decision"] = decision
                SESSION_STATE["elements"] = final_state.get("elements", [])
                assistant_msg["status"] = final_status
                assistant_msg["url"] = final_url
                if final_status == "done":
                    assistant_msg["content"] = (
                        f"Done in **{total_time:.2f}s** ({total_actions} actions).\n\n"
                        f"Destination: [{page_title}]({final_url})"
                    )
                    # Extract structured card from visible page content
                    extracted = extract_structured_data(
                        goal=goal,
                        page_title=page_title,
                        final_url=final_url,
                        page_text=final_state.get("page", {}).get("text", "")
                    )
                    if extracted:
                        assistant_msg["extracted_data"] = extracted
                elif final_status == "stopped":
                    assistant_msg["content"] = (
                        f"Stopped by user after **{total_time:.2f}s** ({total_actions} actions).\n\n"
                        f"Current page: [{page_title}]({final_url})"
                    )
                else:
                    assistant_msg["content"] = (
                        f"Stopped ({final_status}) after **{total_time:.2f}s** ({total_actions} actions).\n\n"
                        f"Current page: [{page_title}]({final_url})"
                    )

            used = final_state.get("used_memories", [])
            if used:
                assistant_msg["used_memories"] = list(dict.fromkeys(used))

            if memory_enabled:
                try:
                    run_record = get_memory().record_task_run(
                        goal=goal,
                        starting_url=starting_url,
                        final_url=final_url,
                        history=final_state.get("history", []),
                        status=final_status,
                        elapsed_seconds=total_time,
                    )
                    if final_status == "done" and run_record.get("actions"):
                        domain = run_record.get("domain", "")
                        assistant_msg["memory_updated"] = f"Saved search pattern for {domain}"
                except Exception:
                    pass
    except Exception as exc:
        with LOCK:
            assistant_msg["status"] = "error"
            assistant_msg["content"] = f"An error occurred during execution: {str(exc)}"
    finally:
        with LOCK:
            ACTIVE_AGENT = None
            SESSION_STATE["running"] = False
            SESSION_STATE["current_step"] = None
            SESSION_STATE["human_intervention"] = None


class ChatHandler(BaseHTTPRequestHandler):
    def send_resp(self, code: int, body: bytes | str, mime: str = "application/json"):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            html = HTML_FILE.read_text(encoding="utf-8")
            return self.send_resp(200, html, "text/html")
        if parsed.path == "/api/status":
            with LOCK:
                return self.send_resp(200, json.dumps(SESSION_STATE))
        if parsed.path == "/api/memory":
            category = parsed.query.split("category=")[1].split("&")[0] if "category=" in parsed.query else None
            mem = get_memory()
            return self.send_resp(200, json.dumps({
                "items": mem.list_all(category=category),
                "graph": mem.get_graph(),
            }))
        if parsed.path == "/api/memory/graph":
            return self.send_resp(200, json.dumps(get_memory().get_graph()))
        if parsed.path == "/api/history":
            mem = get_memory()
            return self.send_resp(200, json.dumps({
                "history": mem.get_task_history(),
                "patterns": mem.get_search_patterns(),
            }))
        return self.send_resp(404, "Not Found", "text/plain")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stop":
            STOP_EVENT.set()
            with LOCK:
                SESSION_STATE["current_step"] = "Stopping..."
            return self.send_resp(200, json.dumps({"status": "stopping"}))

        if parsed.path == "/api/resume":
            with LOCK:
                agent = ACTIVE_AGENT
                SESSION_STATE["human_intervention"] = None
                SESSION_STATE["current_step"] = "Resuming execution after human intervention..."
            if agent:
                try:
                    agent.command("resume", {})
                except Exception as e:
                    return self.send_resp(500, json.dumps({"error": str(e)}))
            return self.send_resp(200, json.dumps({"ok": True}))

        if parsed.path == "/api/otp":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            code = payload.get("code", "").strip()
            with LOCK:
                agent = ACTIVE_AGENT
                SESSION_STATE["human_intervention"] = None
                SESSION_STATE["current_step"] = f"Entering code {code}..."
            if agent:
                try:
                    # Look for input or fill action on the current page
                    page = agent.state.get("page", {})
                    actions = page.get("actions", [])
                    fill_action = next(
                        (a for a in actions if a.get("kind") == "fill" and any(
                            t in (a.get("label") or "").lower() for t in ["otp", "code", "passcode", "security"]
                        )),
                        None
                    )
                    if not fill_action:
                        # Fallback to the first fillable textbox
                        fill_action = next((a for a in actions if a.get("kind") == "fill"), None)

                    if fill_action:
                        agent.browser.act(fill_action, page, text=code)
                    agent.command("resume", {})
                except Exception as e:
                    return self.send_resp(500, json.dumps({"error": str(e)}))
            return self.send_resp(200, json.dumps({"ok": True}))

        if parsed.path == "/api/history/clear":
            get_memory().clear_task_history()
            return self.send_resp(200, json.dumps({"ok": True}))

        if parsed.path == "/api/memory":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text", "").strip()
            category = payload.get("category", "personal")
            if not text:
                return self.send_resp(400, json.dumps({"error": "Memory text cannot be empty"}))
            item_id = get_memory().add(text, category=category)
            return self.send_resp(200, json.dumps({"ok": True, "id": item_id}))

        if parsed.path == "/api/memory/relation":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            source = payload.get("source", "").strip()
            relation = payload.get("relation", "").strip()
            target = payload.get("target", "").strip()
            if not source or not relation or not target:
                return self.send_resp(400, json.dumps({"error": "source, relation, and target are required"}))
            get_memory().add_relation(source, relation, target)
            return self.send_resp(200, json.dumps({"ok": True}))

        if parsed.path == "/api/memory/delete":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            item_id = payload.get("id", "")
            ok = get_memory().delete(item_id)
            return self.send_resp(200, json.dumps({"ok": ok}))

        if parsed.path == "/api/memory/search":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            query = payload.get("query", "").strip()
            results = get_memory().search(query, top_k=5)
            return self.send_resp(200, json.dumps({"results": results}))

        if parsed.path == "/api/chat":
            if SESSION_STATE["running"]:
                return self.send_resp(409, json.dumps({"error": "A task is currently running"}))
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            user_msg = payload.get("message", "").strip()
            memory_enabled = bool(payload.get("memory_enabled", True))
            if not user_msg:
                return self.send_resp(400, json.dumps({"error": "Message cannot be empty"}))

            starting_url, goal = resolve_task(user_msg)
            with LOCK:
                SESSION_STATE["messages"].append({"role": "user", "content": user_msg})

            t = threading.Thread(
                target=run_agent_task,
                args=(user_msg, starting_url, goal, memory_enabled),
                daemon=True,
            )
            t.start()
            return self.send_resp(200, json.dumps({"status": "started"}))

        return self.send_resp(404, "Not Found", "text/plain")

    def log_message(self, *_args):
        pass


def main():
    load_environment()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ChatHandler)
    print(f"Jev browser Chat Interface running at: {ORIGIN}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
