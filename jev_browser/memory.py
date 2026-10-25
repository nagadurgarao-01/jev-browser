"""Graph RAG memory module for Jev browser.

Provides an interconnected Knowledge Graph with multi-hop relational retrieval,
hybrid BM25 lexical indexing, and optional vector embeddings.

Graph RAG enables answering relational and multi-hop queries (e.g. 'Jane Doe's passport'
or 'wife's frequent flyer number') by traversing entity nodes and relationship edges,
while keeping token consumption strictly bounded and ultra-cheap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def normalize_id(name: str) -> str:
    """Normalize entity name into a slug id."""
    slug = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    slug = re.sub(r"[\s_]+", "_", slug)
    return slug or "unknown"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


class BM25Index:
    """Zero-dependency Okapi-BM25 index for fast lexical retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: dict[str, int] = {}
        self.doc_term_freqs: dict[str, dict[str, int]] = {}
        self.term_doc_freqs: dict[str, int] = {}
        self.total_docs = 0
        self.avg_dl = 0.0

    def index(self, docs: dict[str, str]) -> None:
        self.doc_lens.clear()
        self.doc_term_freqs.clear()
        self.term_doc_freqs.clear()
        self.total_docs = len(docs)
        if not self.total_docs:
            self.avg_dl = 0.0
            return

        total_len = 0
        for doc_id, text in docs.items():
            tokens = tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs[doc_id] = tf
            for t in tf:
                self.term_doc_freqs[t] = self.term_doc_freqs.get(t, 0) + 1

        self.avg_dl = total_len / self.total_docs if self.total_docs > 0 else 0.0

    def score(self, query: str) -> dict[str, float]:
        scores: dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_lens}
        if not self.total_docs:
            return scores

        query_tokens = tokenize(query)
        informative = [t for t in query_tokens if t not in STOP_WORDS]
        active_tokens = informative if informative else query_tokens

        for term in active_tokens:
            if term not in self.term_doc_freqs:
                continue
            df = self.term_doc_freqs[term]
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            for doc_id, doc_tf in self.doc_term_freqs.items():
                freq = doc_tf.get(term, 0)
                if freq == 0:
                    continue
                d_len = self.doc_lens.get(doc_id, 0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (d_len / (self.avg_dl or 1.0)))
                scores[doc_id] += idf * (freq * (self.k1 + 1.0)) / (denom or 1.0)
        return scores


def extract_triplets(text: str) -> list[tuple[str, str, str, str, str]]:
    """Extract (subject_name, subject_type, relation, object_name, object_type) triplets from text.

    Supports key-value formats, possessive relations, and procedural patterns.
    """
    triplets = []
    cleaned = text.strip()

    # Pattern 1: "Subject's Attribute/Relation is Value" (e.g. "Jane's passport number is X123")
    m_poss = re.match(
        r"([A-Za-z\s]+?)(?:'s|\s+of)\s+([A-Za-z\s]+?)\s+(?:is|=|:)\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if m_poss:
        subj = m_poss.group(1).strip()
        attr = m_poss.group(2).strip()
        val = m_poss.group(3).strip()
        rel = f"HAS_{attr.upper().replace(' ', '_')}"
        triplets.append((subj, "person", rel, f"{attr}: {val}", "credential"))
        return triplets

    # Pattern 2: "Subject is Relation of Target" (e.g. "Jane Doe is my wife" or "Alice is wife")
    m_rel = re.match(
        r"([A-Za-z\s]+?)\s+is\s+(?:my\s+|the\s+)?(wife|husband|spouse|daughter|son|child|mother|father|colleague|friend)(?:\s+of\s+([A-Za-z\s]+))?",
        cleaned,
        re.IGNORECASE,
    )
    if m_rel:
        subj = m_rel.group(1).strip()
        role = m_rel.group(2).strip().lower()
        target = m_rel.group(3).strip() if m_rel.group(3) else "User"
        triplets.append((subj, "person", f"IS_{role.upper()}_OF", target, "person"))
        return triplets

    # Pattern 3: Key: Value (e.g. "Passport: X1234567, exp 2030" or "Email: test@example.com")
    if ":" in cleaned:
        key, val = cleaned.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key and val:
            k_lower = key.lower()
            if any(w in k_lower for w in ["passport", "aadhaar", "id", "license", "ssn", "voter"]):
                rel = f"HAS_{key.upper().replace(' ', '_')}"
                triplets.append(("User", "person", rel, f"{key}: {val}", "credential"))
            elif any(w in k_lower for w in ["email", "phone", "address", "city", "state", "zip"]):
                triplets.append(("User", "person", f"HAS_{key.upper().replace(' ', '_')}", val, "attribute"))
            elif any(w in k_lower for w in ["frequent flyer", "airline", "seat", "hotel", "preference"]):
                triplets.append(("User", "person", f"PREFERS_{key.upper().replace(' ', '_')}", val, "preference"))
            else:
                triplets.append((key, "entity", "HAS_VALUE", val, "attribute"))
            return triplets

    # Pattern 4: "On site X, do Y" (Procedural)
    m_proc = re.match(r"(?:on\s+site\s+|on\s+)?([A-Za-z0-9\.-]+),\s*(.+)", cleaned, re.IGNORECASE)
    if m_proc:
        site = m_proc.group(1).strip()
        proc = m_proc.group(2).strip()
        triplets.append((site, "website", "REQUIRES_PROCEDURE", proc, "procedure"))
        return triplets

    # Fallback heuristic: Extract capital words or primary nouns linked to the fact
    words = re.findall(r"\b[A-Z][a-z]+\b", cleaned)
    if len(words) >= 2:
        subj = words[0]
        obj = words[1]
        triplets.append((subj, "concept", "RELATED_TO", obj, "concept"))
    elif words:
        triplets.append((words[0], "concept", "HAS_FACT", cleaned, "fact"))
    else:
        triplets.append(("User", "person", "ASSOCIATED_WITH", cleaned, "fact"))

    return triplets


class GraphMemoryStore:
    """Interconnected Knowledge Graph Memory Store with multi-hop RAG retrieval."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else self.default_path()
        self.items: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.task_history: list[dict[str, Any]] = []
        self.search_patterns: list[dict[str, Any]] = []
        self.bm25 = BM25Index()
        self.load()

    @staticmethod
    def default_path() -> Path:
        env_path = os.environ.get("JEV_MEMORY_PATH")
        if env_path:
            return Path(env_path)
        base = Path.home() / ".jev"
        return base / "memory.json"

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.items = data.get("items", {})
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                    self.task_history = data.get("task_history", [])
                    self.search_patterns = data.get("search_patterns", [])
                elif isinstance(data, list):
                    self.items = {item["id"]: item for item in data if "id" in item and "text" in item}
            except Exception:
                self.items = {}
                self.nodes = {}
                self.edges = []
                self.task_history = []
                self.search_patterns = []

        # If items exist but graph is empty, reconstruct graph from items
        if self.items and not self.nodes:
            self._reconstruct_graph_from_items()

        self._reindex()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": self.items,
            "nodes": self.nodes,
            "edges": self.edges,
            "task_history": self.task_history,
            "search_patterns": self.search_patterns,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _reconstruct_graph_from_items(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        for item_id, item in self.items.items():
            self._ingest_into_graph(item["text"], item_id=item_id)

    def _reindex(self) -> None:
        # Build document corpus for BM25 from items and rich entity nodes
        docs = {doc_id: item["text"] for doc_id, item in self.items.items()}
        for node_id, node in self.nodes.items():
            desc = f"{node['name']} ({node.get('type', '')}): " + " ".join(
                f"{k} {v}" for k, v in node.get("attributes", {}).items()
            )
            docs[f"node_{node_id}"] = desc
        self.bm25.index(docs)

    def _ensure_node(self, name: str, entity_type: str = "concept", attributes: dict[str, Any] | None = None) -> str:
        nid = normalize_id(name)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "id": nid,
                "name": name.strip(),
                "type": entity_type,
                "attributes": attributes or {},
            }
        else:
            if attributes:
                self.nodes[nid]["attributes"].update(attributes)
        return nid

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        source_type: str = "concept",
        target_type: str = "concept",
        attributes: dict[str, Any] | None = None,
        item_id: str | None = None,
    ) -> None:
        """Add a directed relation between two entities in the knowledge graph."""
        s_id = self._ensure_node(source, entity_type=source_type)
        t_id = self._ensure_node(target, entity_type=target_type)
        rel_clean = relation.strip().upper().replace(" ", "_")

        # Avoid exact duplicate edge
        for edge in self.edges:
            if edge["source"] == s_id and edge["target"] == t_id and edge["relation"] == rel_clean:
                if attributes:
                    edge["attributes"].update(attributes)
                return

        edge_data = {
            "source": s_id,
            "relation": rel_clean,
            "target": t_id,
            "attributes": attributes or {},
            "item_id": item_id,
        }
        self.edges.append(edge_data)

    def _ingest_into_graph(self, text: str, item_id: str) -> None:
        triplets = extract_triplets(text)
        for subj, s_type, rel, obj, o_type in triplets:
            self.add_relation(
                source=subj,
                relation=rel,
                target=obj,
                source_type=s_type,
                target_type=o_type,
                attributes={"source_text": text},
                item_id=item_id,
            )

    def add(
        self,
        text: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Add text snippet to memory and automatically construct knowledge graph entities & edges."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty.")
        item_id = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
        item: dict[str, Any] = {
            "id": item_id,
            "text": cleaned,
            "category": category,
            "metadata": metadata or {},
        }
        if embedding:
            item["embedding"] = embedding

        self.items[item_id] = item
        self._ingest_into_graph(cleaned, item_id=item_id)
        self.save()
        self._reindex()
        return item_id

    def delete(self, item_id: str) -> bool:
        """Delete an item and prune associated edges."""
        if item_id in self.items:
            del self.items[item_id]
            # Prune edges originating from this item
            self.edges = [e for e in self.edges if e.get("item_id") != item_id]
            self.save()
            self._reindex()
            return True
        return False

    def clear(self) -> None:
        """Clear all stored memories, graph nodes, and edges."""
        self.items.clear()
        self.nodes.clear()
        self.edges.clear()
        self.save()
        self._reindex()

    def list_all(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all items, optionally filtered by category."""
        results = list(self.items.values())
        if category:
            results = [r for r in results if r.get("category") == category]
        return results

    def get_graph(self) -> dict[str, Any]:
        """Return the complete Knowledge Graph schema: nodes and edges."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "counts": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "items": len(self.items),
            },
        }

    def traverse(self, seed_node_ids: list[str], max_hops: int = 2) -> list[dict[str, Any]]:
        """Perform multi-hop traversal starting from seed nodes.

        Returns a list of connected relational paths and neighbor entity facts.
        """
        visited_nodes = set(seed_node_ids)
        current_layer = set(seed_node_ids)
        traversed_edges: list[dict[str, Any]] = []

        for _ in range(max_hops):
            next_layer = set()
            for edge in self.edges:
                s, t = edge["source"], edge["target"]
                if s in current_layer or t in current_layer:
                    if edge not in traversed_edges:
                        traversed_edges.append(edge)
                    next_node = t if s in current_layer else s
                    if next_node not in visited_nodes:
                        next_layer.add(next_node)
                        visited_nodes.add(next_node)
            if not next_layer:
                break
            current_layer = next_layer

        # Synthesize traversed edges into high-density relational statements
        paths = []
        for edge in traversed_edges:
            s_name = self.nodes.get(edge["source"], {}).get("name", edge["source"])
            t_name = self.nodes.get(edge["target"], {}).get("name", edge["target"])
            rel = edge["relation"].replace("_", " ").lower()
            text_repr = f"{s_name} {rel} {t_name}"
            paths.append({
                "source": s_name,
                "relation": edge["relation"],
                "target": t_name,
                "text": text_repr,
            })
        return paths

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: str | None = None,
        query_embedding: list[float] | None = None,
        min_score: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Multi-hop Graph RAG retrieval: Combines seed discovery, graph expansion, and BM25 scoring."""
        if not self.items and not self.nodes:
            return []
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # Step 1: Discover Seed Nodes in the Knowledge Graph
        query_tokens = set(tokenize(cleaned_query))
        seed_nodes = []
        for nid, node in self.nodes.items():
            name_tokens = set(tokenize(node["name"]))
            if name_tokens and name_tokens.issubset(query_tokens):
                seed_nodes.append(nid)
            elif name_tokens.intersection(query_tokens):
                seed_nodes.append(nid)

        # Step 2: Multi-Hop Graph Traversal
        graph_paths = self.traverse(seed_nodes, max_hops=2) if seed_nodes else []

        # Step 3: BM25 Lexical and Embedding Scoring on Item Chunks
        bm25_scores = self.bm25.score(cleaned_query)
        max_bm25 = max(bm25_scores.values()) if bm25_scores and max(bm25_scores.values()) > 0 else 1.0

        candidates = []

        # Include direct text items
        for item_id, item in self.items.items():
            if category and item.get("category") != category:
                continue

            bm25_norm = bm25_scores.get(item_id, 0.0) / max_bm25
            score = bm25_norm

            if query_embedding and "embedding" in item:
                sim = cosine_similarity(query_embedding, item["embedding"])
                score = 0.4 * bm25_norm + 0.6 * max(0.0, sim)

            # Boost item if it is linked to any traversed graph edge
            is_graph_linked = any(
                item["text"] in p["text"] or any(term in item["text"].lower() for term in tokenize(p["text"]))
                for p in graph_paths
            )
            if is_graph_linked:
                score = min(1.0, score + 0.35)

            if score >= min_score or (bm25_scores.get(item_id, 0.0) > 0.0):
                candidates.append({
                    "id": item["id"],
                    "text": item["text"],
                    "category": item.get("category", "general"),
                    "metadata": item.get("metadata", {}),
                    "score": round(score, 4),
                })

        # Step 4: Include Synthesized Graph Traversal Relations if high relevance
        for i, path in enumerate(graph_paths[:2]):
            candidates.append({
                "id": f"graph_rel_{i}",
                "text": f"Knowledge Graph: {path['text']}",
                "category": "graph_relation",
                "metadata": {"relation": path["relation"], "source": path["source"], "target": path["target"]},
                "score": 0.95 - (i * 0.05),
            })

        # Deduplicate and sort by score
        seen_texts = set()
        deduped = []
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for c in candidates:
            if c["text"] not in seen_texts:
                seen_texts.add(c["text"])
                deduped.append(c)

        return deduped[:top_k]

    def record_task_run(
        self,
        goal: str,
        starting_url: str,
        final_url: str = "",
        history: list[dict[str, Any]] | None = None,
        status: str = "done",
        elapsed_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Record an execution run into history and distill successful search/interaction patterns."""
        history = history or []
        parsed_domain = ""
        try:
            from urllib.parse import urlparse
            p = urlparse(starting_url or final_url)
            parsed_domain = p.netloc.replace("www.", "")
        except Exception:
            parsed_domain = "web"

        entry_id = hashlib.sha256(
            f"{goal}_{len(self.task_history)}_{len(history)}".encode("utf-8")
        ).hexdigest()[:12]

        action_summary = [
            f"{h.get('operation', h.get('kind', 'action'))}: {h.get('action')}"
            + (f' (typed: "{h.get("text")}")' if h.get("text") else "")
            for h in history
        ]

        # If the task succeeded, distill search pattern into Knowledge Graph
        if status == "done" and len(history) > 0 and parsed_domain:
            pattern_text = f"On {parsed_domain}, to '{goal}': " + " -> ".join(action_summary[:8])
            # Distill replayable action sequence for fast-path execution
            replay_actions = [
                {
                    "action": h.get("action"),
                    "kind": h.get("kind"),
                    "operation": h.get("operation"),
                    "text": h.get("text"),
                }
                for h in history[:8]
                if h.get("action")
            ]
            pattern_entry = {
                "id": f"pattern_{entry_id}",
                "domain": parsed_domain,
                "goal": goal,
                "steps": action_summary,
                "replay_actions": replay_actions,
                "summary": pattern_text,
            }
            if not any(p["domain"] == parsed_domain and p["goal"] == goal for p in self.search_patterns):
                self.search_patterns.insert(0, pattern_entry)
                self.search_patterns = self.search_patterns[:50]
                # Ingest procedural pattern into knowledge graph
                self.add(
                    pattern_text,
                    category="procedure",
                    metadata={"domain": parsed_domain, "type": "search_pattern"},
                )
                self.add_relation(
                    source=parsed_domain,
                    relation="HAS_SEARCH_PATTERN",
                    target=f"{goal} flow",
                    source_type="website",
                    target_type="procedure",
                    attributes={"pattern": pattern_text},
                )

        record = {
            "id": entry_id,
            "goal": goal,
            "starting_url": starting_url,
            "final_url": final_url,
            "status": status,
            "steps_count": len(history),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "actions": action_summary,
            "domain": parsed_domain,
        }
        self.task_history.insert(0, record)
        self.task_history = self.task_history[:100]
        self.save()
        return record

    def get_task_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recorded task execution history."""
        return self.task_history[:limit]

    def clear_task_history(self) -> None:
        """Clear task history log."""
        self.task_history.clear()
        self.save()

    def get_search_patterns(self, domain: str | None = None) -> list[dict[str, Any]]:
        """Return distilled search patterns, optionally filtered by domain."""
        if domain:
            d_clean = domain.lower().replace("www.", "")
            return [p for p in self.search_patterns if d_clean in p["domain"].lower()]
        return list(self.search_patterns)

    def find_matching_pattern(self, domain: str, goal: str) -> dict[str, Any] | None:
        """Find a high-confidence matching pattern for fast-path macro replay."""
        patterns = self.get_search_patterns(domain)
        if not patterns:
            return None
        goal_tokens = set(tokenize(goal))
        for p in patterns:
            # Check for exact or high Jaccard token overlap
            p_goal_tokens = set(tokenize(p.get("goal", "")))
            if not p_goal_tokens:
                continue
            intersection = len(goal_tokens.intersection(p_goal_tokens))
            union = len(goal_tokens.union(p_goal_tokens))
            jaccard = intersection / max(1, union)
            if jaccard >= 0.65 or p_goal_tokens.issubset(goal_tokens):
                if p.get("replay_actions"):
                    return p
        return None


# Alias MemoryStore to GraphMemoryStore for 100% backward compatibility
MemoryStore = GraphMemoryStore

_DEFAULT_MEMORY: GraphMemoryStore | None = None


def get_memory(path: Path | str | None = None) -> GraphMemoryStore:
    """Return the shared GraphMemoryStore instance or initialize with custom path."""
    global _DEFAULT_MEMORY
    if path:
        return GraphMemoryStore(path)
    if _DEFAULT_MEMORY is None:
        _DEFAULT_MEMORY = GraphMemoryStore()
    return _DEFAULT_MEMORY
