"""Offline contracts for RAG memory module. No paid APIs."""

import pytest

from jev_browser.memory import BM25Index, MemoryStore, cosine_similarity, tokenize
from jev_browser.model import field_context


def test_tokenize():
    tokens = tokenize("Hello, World! User's email is test@example.com.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "email" in tokens
    assert "test" in tokens


def test_bm25_index_basic():
    bm25 = BM25Index()
    docs = {
        "d1": "User passport number is P1234567 issued in Berlin",
        "d2": "Frequent flyer number DL9988 for Delta airlines",
        "d3": "Home shipping address is 100 Main Street, San Francisco",
    }
    bm25.index(docs)
    scores = bm25.score("passport")
    assert scores["d1"] > scores["d2"]
    assert scores["d1"] > scores["d3"]

    scores_flyer = bm25.score("delta frequent flyer")
    assert scores_flyer["d2"] > scores_flyer["d1"]
    assert scores_flyer["d2"] > scores_flyer["d3"]


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_memory_store_crud(tmp_path):
    store_file = tmp_path / "memory.json"
    mem = MemoryStore(path=store_file)

    assert mem.list_all() == []
    assert mem.search("anything") == []

    id1 = mem.add("User email is test@example.com", category="personal")
    id2 = mem.add("Passport number: X9876543, valid until 2030", category="personal")
    mem.add("On site X, always click accept cookies first", category="procedure")

    assert len(mem.list_all()) == 3
    assert len(mem.list_all(category="procedure")) == 1

    # Search
    results = mem.search("what is my passport?", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == id2
    assert "X9876543" in results[0]["text"]

    # Persistence verification
    mem2 = MemoryStore(path=store_file)
    assert len(mem2.list_all()) == 3
    persisted_results = mem2.search("email address")
    assert len(persisted_results) > 0
    assert persisted_results[0]["id"] == id1

    # Delete
    assert mem.delete(id1) is True
    assert mem.delete("nonexistent") is False
    assert len(mem.list_all()) == 2

    # Clear
    mem.clear()
    assert mem.list_all() == []


def test_hybrid_search_with_embedding(tmp_path):
    mem = MemoryStore(path=tmp_path / "mem.json")
    emb1 = [0.9, 0.1, 0.0]
    emb2 = [0.0, 0.1, 0.9]

    mem.add("Flight booking preference is aisle seat", category="travel", embedding=emb1)
    mem.add("Billing card ending in 4242", category="billing", embedding=emb2)

    query_emb = [0.85, 0.15, 0.0]
    results = mem.search("seat preference", top_k=1, query_embedding=query_emb)
    assert len(results) == 1
    assert "aisle seat" in results[0]["text"]


def test_field_context_with_memory(tmp_path):
    mem = MemoryStore(path=tmp_path / "mem.json")
    mem.add("Full Name: Johnathan Doe", category="personal")
    mem.add("Passport: A1234567", category="personal")

    goal = "Fill out applicant registration"
    action = {"id": "e1", "kind": "fill", "label": "Passport Number", "role": "textbox", "value": ""}
    page_data = {"title": "Visa Application Form", "text": "Please enter your passport number"}
    history = []

    # With memory
    ctx = field_context(goal, action, page_data, history, memory=mem)
    assert "memory" in ctx
    assert any("A1234567" in m for m in ctx["memory"])

    # Without memory
    ctx_no_mem = field_context(goal, action, page_data, history, memory=None)
    assert "memory" not in ctx_no_mem


def test_extract_triplets():
    from jev_browser.memory import extract_triplets

    # Key-value
    t1 = extract_triplets("Passport: X9876543, valid until 2030")
    assert len(t1) > 0
    assert any("PASSPORT" in t[2] for t in t1)

    # Relational spouse
    t2 = extract_triplets("Jane Doe is my wife")
    assert len(t2) > 0
    assert any("WIFE" in t[2] for t in t2)

    # Possessive
    t3 = extract_triplets("Jane's passport is X9876543")
    assert len(t3) > 0
    assert any("PASSPORT" in t[2] for t in t3)


def test_graph_multi_hop_traversal(tmp_path):
    mem = MemoryStore(path=tmp_path / "graph_mem.json")
    mem.add("Jane Doe is my wife", category="personal")
    mem.add("Jane's passport is X9876543", category="personal")

    graph = mem.get_graph()
    assert graph["counts"]["nodes"] >= 2
    assert graph["counts"]["edges"] >= 2

    # Query for "wife's passport" (requires linking wife -> Jane Doe -> passport)
    results = mem.search("Jane wife passport", top_k=3)
    assert len(results) > 0
    combined = " ".join(r["text"] for r in results)
    assert "Jane" in combined
    assert "X9876543" in combined or "passport" in combined.lower()


def test_explicit_add_relation(tmp_path):
    mem = MemoryStore(path=tmp_path / "graph_rel.json")
    mem.add_relation("Alice", "WORKS_AT", "Acme Corp", source_type="person", target_type="org")
    mem.add_relation("Acme Corp", "LOCATED_IN", "Zurich", source_type="org", target_type="city")

    graph = mem.get_graph()
    assert graph["counts"]["nodes"] == 3
    assert graph["counts"]["edges"] == 2

    results = mem.search("Alice company", top_k=2)
    assert len(results) > 0
    assert any("Acme Corp" in r["text"] for r in results)


def test_record_task_run_and_search_patterns(tmp_path):
    mem = MemoryStore(path=tmp_path / "pattern_mem.json")

    history = [
        {"step": 1, "operation": "CLICK", "action": "Search input", "text": None},
        {"step": 2, "operation": "TYPE_TEXT", "action": "Search input", "text": "Alan Turing"},
        {"step": 3, "operation": "CLICK", "action": "Alan Turing - Wikipedia", "text": None},
    ]

    record = mem.record_task_run(
        goal="Find Alan Turing on Wikipedia",
        starting_url="https://en.wikipedia.org",
        final_url="https://en.wikipedia.org/wiki/Alan_Turing",
        history=history,
        status="done",
        elapsed_seconds=1.85,
    )

    assert record["status"] == "done"
    assert record["steps_count"] == 3
    assert len(mem.get_task_history()) == 1

    patterns = mem.get_search_patterns(domain="wikipedia.org")
    assert len(patterns) == 1
    assert "wikipedia.org" in patterns[0]["domain"]
    assert "TYPE_TEXT" in patterns[0]["summary"]

    # Verify that future search for Wikipedia Alan Turing retrieves this pattern
    results = mem.search("wikipedia Alan Turing search", top_k=3)
    assert len(results) > 0
    assert any("wikipedia.org" in r["text"].lower() for r in results)


def test_task_history_persistence_and_clear(tmp_path):
    p = tmp_path / "hist_mem.json"
    mem = MemoryStore(path=p)
    mem.record_task_run(goal="Test 1", starting_url="https://example.com", status="done")
    mem.record_task_run(goal="Test 2", starting_url="https://google.com", status="blocked")

    assert len(mem.get_task_history()) == 2

    # Reload from disk
    mem2 = MemoryStore(path=p)
    assert len(mem2.get_task_history()) == 2
    assert mem2.get_task_history()[0]["goal"] == "Test 2"

    mem2.clear_task_history()
    assert len(mem2.get_task_history()) == 0


def test_find_matching_pattern_for_macro_replay(tmp_path):
    mem = MemoryStore(path=tmp_path / "macro_mem.json")
    history = [
        {"step": 1, "operation": "TYPE_TEXT", "action": "Search", "kind": "fill", "text": "laptops"},
        {"step": 2, "operation": "CLICK", "action": "Search button", "kind": "click", "text": None},
    ]
    mem.record_task_run(
        goal="search laptops on amazon",
        starting_url="https://www.amazon.com",
        final_url="https://www.amazon.com/s?k=laptops",
        history=history,
        status="done",
    )

    # High match on same domain
    matched = mem.find_matching_pattern("amazon.com", "search laptops on amazon")
    assert matched is not None
    assert len(matched["replay_actions"]) == 2
    assert matched["replay_actions"][0]["action"] == "Search"
    assert matched["replay_actions"][0]["text"] == "laptops"

    # Subset goal tokens match
    matched_subset = mem.find_matching_pattern("amazon.com", "search laptops on amazon today")
    assert matched_subset is not None

    # Unmatched domain
    assert mem.find_matching_pattern("google.com", "search laptops on amazon") is None

    # Completely different goal on same domain
    assert mem.find_matching_pattern("amazon.com", "checkout return orders customer service") is None
