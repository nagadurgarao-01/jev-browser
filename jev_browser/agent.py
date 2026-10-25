"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text
from .questions import MAX_STEPS

# Patterns that signal human intervention is required (OTP, CAPTCHA, 2FA, login, credential)
INTERVENTION_PATTERNS = [
    r"\b(one-time password|otp|verification code|security code|passcode)\b",
    r"\b(enter the code sent to|we sent a code|sms code|email code)\b",
    r"\b(captcha|recaptcha|hcaptcha|cloudflare challenge|turnstile)\b",
    r"\b(two-factor|2-factor|2fa|mfa|authenticate your account)\b",
    r"\b(verify it's you|verify your identity|unusual activity detected)\b",
    r"\b(enter your phone number to continue|get access to your orders|enter your email/mobile)\b",
    r"\b(login to continue|sign in to your account|please sign in|please log in)\b",
]


def detect_human_intervention(page: dict | None) -> dict | None:
    """Detect if the current page requires human-in-the-loop (OTP, 2FA, CAPTCHA)."""
    if not page:
        return None
    text = (page.get("text") or "").lower()
    title = (page.get("title") or "").lower()
    combined = f"{title}\n{text}"

    for pattern in INTERVENTION_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            kind = "captcha" if "captcha" in pattern or "challenge" in pattern else "2fa_otp"
            reason = match.group(0).capitalize()
            return {"kind": kind, "reason": reason}

    # Also check action inputs for OTP / 2FA field labels
    for action in page.get("actions", []):
        lbl = (action.get("label") or "").lower()
        if any(term in lbl for term in ["otp", "verification code", "passcode", "security code", "2fa"]):
            return {"kind": "2fa_otp", "reason": f"Input field: {action.get('label')}"}

    return None


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False, memory=None):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.memory = memory
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        init_intervention = detect_human_intervention(page)
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="waiting_for_human" if init_intervention else "ready",
            human_intervention=init_intervention,
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        self.macro_steps = []
        self.macro_idx = 0
        self.macro_pattern = None
        if self.memory and url:
            parsed_domain = urlparse(url).netloc.lower().replace("www.", "")
            if parsed_domain:
                pattern = self.memory.find_matching_pattern(parsed_domain, self.state["goal"])
                if pattern and pattern.get("replay_actions"):
                    self.macro_steps = list(pattern["replay_actions"])
                    self.macro_pattern = pattern
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def _match_macro_step(self, macro_step: dict, page: dict, goal: str) -> dict | None:
        """Attempt to match a macro replay action against the current page actions."""
        actions = page.get("actions", [])
        if not actions or not macro_step:
            return None

        elements, targets, controls = action_space(actions)
        macro_kind = macro_step.get("kind")
        macro_label = (macro_step.get("action") or "").strip().lower()
        macro_op = macro_step.get("operation")

        matched_action = None

        # 1. Exact match on label and kind
        for act in actions:
            if act.get("kind") == macro_kind:
                act_label = (act.get("label") or "").strip().lower()
                if act_label == macro_label:
                    matched_action = act
                    break

        # 2. Substring containment match if no exact match
        if not matched_action and macro_label:
            for act in actions:
                if act.get("kind") == macro_kind:
                    act_label = (act.get("label") or "").strip().lower()
                    if act_label and (act_label in macro_label or macro_label in act_label):
                        matched_action = act
                        break

        # 3. High token overlap match
        if not matched_action and macro_label:
            macro_tokens = set(macro_label.split())
            best_score = 0.0
            best_act = None
            for act in actions:
                if act.get("kind") == macro_kind:
                    act_label = (act.get("label") or "").strip().lower()
                    act_tokens = set(act_label.split())
                    if act_tokens:
                        overlap = len(macro_tokens.intersection(act_tokens)) / max(
                            1, len(macro_tokens.union(act_tokens))
                        )
                        if overlap > best_score and overlap >= 0.5:
                            best_score = overlap
                            best_act = act
            if best_act:
                matched_action = best_act

        if not matched_action:
            return None

        selected_id = matched_action["id"]

        # Resolve operation
        if macro_op and macro_op in targets:
            op = macro_op
        elif matched_action["kind"] == "fill" and "TYPE_TEXT" in targets:
            op = "TYPE_TEXT"
        elif matched_action["kind"] == "click" and "CLICK" in targets:
            op = "CLICK"
        elif matched_action["kind"] == "select" and "SELECT" in targets:
            op = "SELECT"
        elif selected_id.upper() in controls:
            op = selected_id.upper()
        else:
            op = "CLICK" if "CLICK" in targets else "WAIT"

        # Resolve target index
        target_idx = None
        if op in targets:
            for t_k, t_act in targets[op].items():
                if t_act["id"] == selected_id:
                    target_idx = t_k
                    break
            if target_idx is None:
                return None

        probabilities = {act["id"]: (1.0 if act["id"] == selected_id else 0.0) for act in actions}
        if selected_id not in probabilities:
            probabilities[selected_id] = 1.0

        # Pre-fill text if macro has stored text and matches current goal
        macro_text = macro_step.get("text")
        if matched_action["kind"] == "fill" and macro_text:
            if macro_text.lower() in goal.lower() or (
                self.macro_pattern and self.macro_pattern.get("goal") == goal
            ):
                memory = getattr(self, "memory", None)
                ctx = field_context(goal, matched_action, page, self.state["history"], memory=memory)
                self.pending_text = (ctx, macro_text, {"model": "fast-path-macro", "latency_ms": 5})

        return {
            "choice": selected_id,
            "operation": op,
            "target": target_idx or "1",
            "confidence": 1.0,
            "probabilities": probabilities,
            "operation_probabilities": {op: 1.0},
            "target_probabilities": {target_idx: 1.0} if target_idx else {},
            "target_confidence": 1.0,
            "raw_answers": {"macro": True, "step": self.macro_idx},
            "model": "fast-path-macro",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 25,
            "fast_path": True,
        }

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            except Exception as err:
                # If a model evaluation error occurs (e.g. TypeSafe response parse on complex modal),
                # re-observe the page and check if a human intervention is needed or retry cleanly.
                state["decision"] = None
                try:
                    state["page"] = state["browser"].observe(screenshot=self.screenshots)
                except Exception:
                    pass
                intervention = detect_human_intervention(state.get("page"))
                if intervention:
                    state["human_intervention"] = intervention
                    state["status"] = "waiting_for_human"
                    return self.snapshot()
                raise err
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")

            # Try fast-path macro step if available
            if self.memory and not self.macro_steps and state["page"].get("url"):
                parsed_domain = urlparse(state["page"]["url"]).netloc.lower().replace("www.", "")
                if parsed_domain:
                    pattern = self.memory.find_matching_pattern(parsed_domain, state["goal"])
                    if pattern and pattern.get("replay_actions"):
                        self.macro_steps = list(pattern["replay_actions"])
                        self.macro_idx = 0
                        self.macro_pattern = pattern

            if self.macro_steps and self.macro_idx < len(self.macro_steps):
                macro_step = self.macro_steps[self.macro_idx]
                macro_decision = self._match_macro_step(macro_step, state["page"], state["goal"])
                if macro_decision:
                    state["decision"] = macro_decision
                    self.macro_idx += 1
                else:
                    # Drop out of fast-path cleanly if element missing or changed
                    self.macro_steps = []

            if not state["decision"]:
                state["decision"] = choose(state["page"], state["goal"], state["history"])
                state["decision"].setdefault("fast_path", False)

            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                memory = getattr(self, "memory", None)
                context = field_context(state["goal"], action, page, state["history"], memory=memory)
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                call_entry = {**helper, "field": action["label"], "value": text}
                if context.get("memory"):
                    call_entry["memory"] = context["memory"]
                    state.setdefault("used_memories", []).extend(context["memory"])
                state["text_calls"].append(call_entry)
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                    "fast_path": decision.get("fast_path", False),
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            repeated = state["history"][-3:]
            intervention = detect_human_intervention(state["page"])
            if intervention:
                state["human_intervention"] = intervention
                state["status"] = "waiting_for_human"
            elif len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated):
                state["status"] = "blocked"
            else:
                state["human_intervention"] = None
                state["status"] = "ready"
        elif name == "resume":
            # Resume after human intervention (e.g., user completed OTP/CAPTCHA or submitted code)
            state["human_intervention"] = None
            state["status"] = "ready"
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - (state["started_at"] or time.perf_counter())) * 1000)
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            if self.state["status"] == "waiting_for_human":
                yield self.snapshot()
                time.sleep(0.5)
                continue
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
