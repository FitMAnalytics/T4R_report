"""
State/Action Taxonomy Builder v4
=================================

Changes from v3:
- Raw transcript text preserved in trajectories.csv (state_text, action_text)
- Dictionary entries store example utterances for lookup
- GPT prompts instruct collapsing of trivial/filler turns into reserved IDs
  (state_id=0 for trivial states, action_id=0 for trivial actions)

Input:  CSV with columns [conversation_id, transcript]
Output:
  - trajectories.csv       : one row per turn with raw text + IDs
  - state_dictionary.csv   : state taxonomy with example utterances
  - action_dictionary.csv  : action taxonomy with example utterances
  - taxonomy_store/        : FAISS indices + metadata for incremental updates

Usage:
    python state_action_taxonomy_v4.py --input transcripts.csv --output ./results/

Requirements:
    pip install faiss-cpu openai numpy pandas
"""

import json
import faiss
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import argparse
import time


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
    api_key="YOUR_KEY",
)
GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

EMBED_DIM = 1536
SIMILARITY_THRESHOLD = 0.82
TOP_K = 5
MAX_EXAMPLE_UTTERANCES = 10  # max examples stored per dictionary entry

# Reserved IDs for trivial/filler turns
TRIVIAL_STATE_ID = 0
TRIVIAL_ACTION_ID = 0


# ─────────────────────────────────────────────
# 2. JSON SCHEMAS
# ─────────────────────────────────────────────

STATE_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "state_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_trivial": {
                    "type": "boolean",
                    "description": "True if this state has no strategic significance"
                },
                "trivial_reason": {
                    "type": ["string", "null"],
                    "description": "If trivial, why (e.g., 'opening_greeting', 'small_talk', 'acknowledgment'). Null if not trivial."
                },
                "customer_intent": {
                    "type": "string",
                    "description": "What the customer is trying to accomplish"
                },
                "customer_sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "cautious", "frustrated", "hostile"]
                },
                "objections_raised": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "offers_made": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "conversation_stage": {
                    "type": "string",
                    "enum": ["opening", "discovery", "objection_handling", "negotiation", "closing", "escalation"]
                },
                "competitive_mentions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "key_context": {"type": "string"},
                "summary": {"type": "string"}
            },
            "required": [
                "is_trivial", "trivial_reason",
                "customer_intent", "customer_sentiment", "objections_raised",
                "offers_made", "conversation_stage", "competitive_mentions",
                "key_context", "summary"
            ],
            "additionalProperties": False
        }
    }
}

ACTION_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_trivial": {
                    "type": "boolean",
                    "description": "True if this action has no strategic significance"
                },
                "trivial_reason": {
                    "type": ["string", "null"],
                    "description": "If trivial, why (e.g., 'greeting', 'filler', 'acknowledgment'). Null if not trivial."
                },
                "action_type": {"type": "string"},
                "tactic": {"type": "string"},
                "tone": {
                    "type": "string",
                    "enum": ["empathetic", "assertive", "consultative", "apologetic", "enthusiastic", "neutral"]
                },
                "personalization_level": {
                    "type": "string",
                    "enum": ["generic", "moderate", "highly_personalized"]
                },
                "summary": {"type": "string"}
            },
            "required": [
                "is_trivial", "trivial_reason",
                "action_type", "tactic", "tone", "personalization_level", "summary"
            ],
            "additionalProperties": False
        }
    }
}

MATCH_DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "match_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "match": {"type": "boolean"},
                "matched_id": {"type": ["integer", "null"]},
                "reasoning": {"type": "string"},
                "suggested_category": {"type": "string"}
            },
            "required": ["match", "matched_id", "reasoning", "suggested_category"],
            "additionalProperties": False
        }
    }
}


# ─────────────────────────────────────────────
# 3. RETRIEVAL KEY CONSTRUCTION
# ─────────────────────────────────────────────

def build_state_retrieval_key(s: dict) -> str:
    parts = [
        f"intent: {s['customer_intent']}",
        f"sentiment: {s['customer_sentiment']}",
        f"stage: {s['conversation_stage']}",
    ]
    if s["objections_raised"]:
        parts.append(f"objections: {', '.join(s['objections_raised'])}")
    if s["competitive_mentions"]:
        parts.append(f"competitors: {', '.join(s['competitive_mentions'])}")
    if s["offers_made"]:
        parts.append(f"prior_offers: {', '.join(s['offers_made'])}")
    return " | ".join(parts)


def build_action_retrieval_key(a: dict) -> str:
    return f"type: {a['action_type']} | tactic: {a['tactic']} | tone: {a['tone']}"


# ─────────────────────────────────────────────
# 4. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class DictionaryEntry:
    id: int
    category: str
    description: str
    structured_summary: dict
    retrieval_key: str
    example_conversations: list = field(default_factory=list)
    example_utterances: list = field(default_factory=list)  # raw text examples
    count: int = 0

    def add_example(self, conversation_id: str, utterance: str):
        """Record a hit with the raw utterance text."""
        self.count += 1
        if conversation_id not in self.example_conversations:
            self.example_conversations.append(conversation_id)
        # Keep up to MAX_EXAMPLE_UTTERANCES diverse examples
        if len(self.example_utterances) < MAX_EXAMPLE_UTTERANCES:
            self.example_utterances.append({
                "conversation_id": conversation_id,
                "text": utterance[:500],  # truncate very long utterances
            })


class TaxonomyStore:
    def __init__(self, name: str, embed_dim: int = EMBED_DIM):
        self.name = name
        self.embed_dim = embed_dim
        self.index = faiss.IndexFlatIP(embed_dim)
        self.entries: dict[int, DictionaryEntry] = {}
        self.next_id = 1  # start at 1; 0 is reserved for trivial

        # Pre-create the trivial entry
        trivial = DictionaryEntry(
            id=0,
            category="trivial",
            description="Trivial/filler turn with no strategic significance (greetings, acknowledgments, pleasantries, small talk)",
            structured_summary={"is_trivial": True},
            retrieval_key="trivial",
        )
        self.entries[0] = trivial

    def add_entry(self, entry: DictionaryEntry, embedding: np.ndarray) -> int:
        vec = embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(vec)
        self.index.add(vec)
        entry.id = self.next_id
        entry.count = 1
        self.entries[self.next_id] = entry
        self.next_id += 1
        return entry.id

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K):
        if self.index.ntotal == 0:
            return []
        vec = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        # Map FAISS sequential index back to entry IDs
        # FAISS index position i corresponds to entry with id = i + 1
        # (because id=0 is trivial and NOT in the FAISS index)
        return [
            (self.entries[int(idx) + 1], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for eid, e in self.entries.items():
            row = {
                f"{self.name}_id": eid,
                f"{self.name}_category": e.category,
                f"{self.name}_description": e.description,
                f"{self.name}_retrieval_key": e.retrieval_key,
                f"{self.name}_count": e.count,
                f"{self.name}_num_conversations": len(e.example_conversations),
                # Store example utterances as JSON string for CSV
                f"{self.name}_example_utterances": json.dumps(
                    e.example_utterances, ensure_ascii=False
                ),
            }
            # Flatten structured summary fields
            for k, v in e.structured_summary.items():
                if isinstance(v, list):
                    row[f"{self.name}_{k}"] = "; ".join(str(x) for x in v) if v else ""
                else:
                    row[f"{self.name}_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}/{self.name}_index.faiss")
        meta = {k: asdict(v) for k, v in self.entries.items()}
        with open(f"{path}/{self.name}_meta.json", "w") as f:
            json.dump({"entries": meta, "next_id": self.next_id}, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/{self.name}_index.faiss")
        with open(f"{path}/{self.name}_meta.json") as f:
            data = json.load(f)
        self.next_id = data["next_id"]
        self.entries = {int(k): DictionaryEntry(**v) for k, v in data["entries"].items()}


# ─────────────────────────────────────────────
# 5. SYSTEM PROMPTS
# ─────────────────────────────────────────────

STATE_SUMMARY_SYSTEM = """\
You are analyzing sales conversations between an Agent and a Customer.
Given a conversation history (ending with the customer's most recent turn),
characterize the current conversational STATE.

IMPORTANT — Trivial state detection:
Set is_trivial=true if the conversation so far contains NO strategically meaningful
information. Examples of trivial states:
- Customer just said hello/greeted the agent
- Customer is responding to a greeting with small talk
- Customer acknowledged something with "OK", "sure", "thanks" without adding substance
- The conversation has not yet revealed any intent, objection, or decision point

If is_trivial=true, still fill in the other fields with best guesses, but they
won't be used for taxonomy matching.

If is_trivial=false, focus on what matters for deciding the agent's next move:
the customer's goal, emotional state, objections, what has been tried, and
where we are in the conversation flow."""

ACTION_SUMMARY_SYSTEM = """\
You are analyzing sales conversations between an Agent and a Customer.
Given the conversation history and the agent's response,
characterize the agent's strategic ACTION.

IMPORTANT — Trivial action detection:
Set is_trivial=true if the agent's response has NO strategic significance.
Examples of trivial actions:
- Standard greeting ("Thank you for calling, how can I help?")
- Pure acknowledgment with no strategic content ("I see", "Got it", "Let me pull up your account")
- Procedural/administrative responses ("Can you verify your name and address?")
- Generic rapport building with no persuasion or information gathering intent
- Hold/transfer announcements ("Let me place you on a brief hold")

Trivial does NOT include:
- Discovery questions (even simple ones like "What brings you in today?" — these gather info)
- Empathy statements paired with a pivot ("I understand, let me show you...")
- Any response that contains an offer, reframe, or objection handling

If is_trivial=false, focus on the type of strategy, specific tactic, and tone."""

STATE_MATCH_SYSTEM = """\
You are building a taxonomy of conversation STATEs for a sales call RL system.
A new state has been observed. Decide whether it matches any existing entry.
Rules:
- Match = same conversational situation where an agent would use the same strategy.
- Minor surface differences (dollar amounts, card names, specific dates) do NOT prevent a match.
- Different customer intent, sentiment, or conversation stage = DIFFERENT state.
- If no candidates are provided, set match=false."""

ACTION_MATCH_SYSTEM = """\
You are building a taxonomy of agent ACTIONs for a sales call RL system.
A new action has been observed. Decide whether it matches any existing entry.
Rules:
- Match = same strategic move by the agent.
- Minor wording/product differences do NOT prevent a match.
- Same action_type but different tactic = DIFFERENT action.
- If no candidates are provided, set match=false."""


# ─────────────────────────────────────────────
# 6. CORE FUNCTIONS
# ─────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype="float32")


def call_gpt_structured(system: str, user_content: str, response_format: dict) -> dict:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                response_format=response_format,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                print(f"  [RETRY] GPT call failed: {e}, retrying in 5s...")
                time.sleep(5)
            else:
                raise


def format_candidates(results: list[tuple[DictionaryEntry, float]]) -> str:
    if not results:
        return "(No existing entries yet.)"
    lines = []
    for entry, score in results:
        examples_str = ""
        if entry.example_utterances:
            ex_texts = [ex["text"][:150] for ex in entry.example_utterances[:3]]
            examples_str = "\n  Example utterances:\n" + "\n".join(f'    - "{t}"' for t in ex_texts)
        lines.append(
            f"[ID={entry.id}, similarity={score:.3f}]\n"
            f"  Category: {entry.category}\n"
            f"  Key: {entry.retrieval_key}\n"
            f"  Description: {entry.description}"
            f"{examples_str}"
        )
    return "\n\n".join(lines)


def classify_state(
    conversation: str,
    state_text: str,  # the customer's raw utterance for this turn
    store: TaxonomyStore,
    conv_id: str,
) -> tuple[int, dict]:
    """Returns (state_id, structured_summary)."""
    summary = call_gpt_structured(
        STATE_SUMMARY_SYSTEM,
        f"<conversation>\n{conversation}\n</conversation>",
        STATE_SUMMARY_SCHEMA,
    )

    # Check if trivial
    if summary.get("is_trivial", False):
        store.entries[TRIVIAL_STATE_ID].add_example(conv_id, state_text)
        print(f"  [STATE] Trivial (reason: {summary.get('trivial_reason', 'N/A')})")
        return TRIVIAL_STATE_ID, summary

    key = build_state_retrieval_key(summary)
    emb = get_embedding(key)
    candidates = store.search(emb, TOP_K)

    if not candidates or candidates[0][1] < SIMILARITY_THRESHOLD:
        entry = DictionaryEntry(
            id=-1,
            category=summary["customer_intent"].replace(" ", "_").lower()[:60],
            description=summary["summary"],
            structured_summary=summary,
            retrieval_key=key,
        )
        entry.add_example(conv_id, state_text)
        sid = store.add_entry(entry, emb)
        print(f"  [STATE] New #{sid}: {entry.category}")
        return sid, summary

    match = call_gpt_structured(
        STATE_MATCH_SYSTEM,
        f"New state:\n{json.dumps(summary, indent=2)}\nKey: {key}\n\n"
        f"Candidates:\n{format_candidates(candidates)}",
        MATCH_DECISION_SCHEMA,
    )

    if match["match"] and match["matched_id"] is not None:
        mid = match["matched_id"]
        store.entries[mid].add_example(conv_id, state_text)
        print(f"  [STATE] Matched → #{mid}: {store.entries[mid].category}")
        return mid, summary

    entry = DictionaryEntry(
        id=-1,
        category=match.get("suggested_category", "unknown"),
        description=summary["summary"],
        structured_summary=summary,
        retrieval_key=key,
    )
    entry.add_example(conv_id, state_text)
    sid = store.add_entry(entry, emb)
    print(f"  [STATE] New #{sid}: {entry.category}")
    return sid, summary


def classify_action(
    conversation: str,
    agent_text: str,
    store: TaxonomyStore,
    conv_id: str,
) -> tuple[int, dict]:
    """Returns (action_id, structured_summary)."""
    summary = call_gpt_structured(
        ACTION_SUMMARY_SYSTEM,
        f"<conversation_history>\n{conversation}\n</conversation_history>\n"
        f"<agent_response>\n{agent_text}\n</agent_response>",
        ACTION_SUMMARY_SCHEMA,
    )

    # Check if trivial
    if summary.get("is_trivial", False):
        store.entries[TRIVIAL_ACTION_ID].add_example(conv_id, agent_text)
        print(f"  [ACTION] Trivial (reason: {summary.get('trivial_reason', 'N/A')})")
        return TRIVIAL_ACTION_ID, summary

    key = build_action_retrieval_key(summary)
    emb = get_embedding(key)
    candidates = store.search(emb, TOP_K)

    if not candidates or candidates[0][1] < SIMILARITY_THRESHOLD:
        entry = DictionaryEntry(
            id=-1,
            category=f"{summary['action_type']}_{summary['tactic']}",
            description=summary["summary"],
            structured_summary=summary,
            retrieval_key=key,
        )
        entry.add_example(conv_id, agent_text)
        aid = store.add_entry(entry, emb)
        print(f"  [ACTION] New #{aid}: {entry.category}")
        return aid, summary

    match = call_gpt_structured(
        ACTION_MATCH_SYSTEM,
        f"New action:\n{json.dumps(summary, indent=2)}\nKey: {key}\n\n"
        f"Candidates:\n{format_candidates(candidates)}",
        MATCH_DECISION_SCHEMA,
    )

    if match["match"] and match["matched_id"] is not None:
        mid = match["matched_id"]
        store.entries[mid].add_example(conv_id, agent_text)
        print(f"  [ACTION] Matched → #{mid}: {store.entries[mid].category}")
        return mid, summary

    entry = DictionaryEntry(
        id=-1,
        category=match.get("suggested_category", "unknown"),
        description=summary["summary"],
        structured_summary=summary,
        retrieval_key=key,
    )
    entry.add_example(conv_id, agent_text)
    aid = store.add_entry(entry, emb)
    print(f"  [ACTION] New #{aid}: {entry.category}")
    return aid, summary


# ─────────────────────────────────────────────
# 7. TRANSCRIPT PROCESSING
# ─────────────────────────────────────────────

def parse_transcript(text: str) -> list[dict]:
    turns = []
    current_role = None
    current_text = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("agent:"):
            if current_role:
                turns.append({"role": current_role, "text": " ".join(current_text)})
            current_role = "agent"
            current_text = [line[len("agent:"):].strip()]
        elif line.lower().startswith("customer:"):
            if current_role:
                turns.append({"role": current_role, "text": " ".join(current_text)})
            current_role = "customer"
            current_text = [line[len("customer:"):].strip()]
        else:
            current_text.append(line)
    if current_role:
        turns.append({"role": current_role, "text": " ".join(current_text)})
    return turns


def process_single_transcript(
    conv_id: str,
    transcript: str,
    state_store: TaxonomyStore,
    action_store: TaxonomyStore,
) -> list[dict]:
    """Process one transcript, return list of turn-level rows."""
    turns = parse_transcript(transcript)
    rows = []
    conversation_so_far = ""

    for i, turn in enumerate(turns):
        if turn["role"] == "customer":
            conversation_so_far += f"Customer: {turn['text']}\n"
            if i + 1 < len(turns) and turns[i + 1]["role"] == "agent":
                agent_turn = turns[i + 1]

                print(f"\n  --- Turn {i} ---")

                state_id, state_summary = classify_state(
                    conversation_so_far,
                    turn["text"],  # customer's raw utterance
                    state_store,
                    conv_id,
                )
                action_id, action_summary = classify_action(
                    conversation_so_far,
                    agent_turn["text"],
                    action_store,
                    conv_id,
                )

                state_entry = state_store.entries[state_id]
                action_entry = action_store.entries[action_id]

                rows.append({
                    # ── Identifiers ──
                    "conversation_id": conv_id,
                    "turn_index": i,

                    # ── Raw text (for lookup) ──
                    "state_text": conversation_so_far.strip(),  # full history up to customer turn
                    "customer_utterance": turn["text"],          # just the customer's turn
                    "action_text": agent_turn["text"],           # agent's raw response

                    # ── State taxonomy ──
                    "state_id": state_id,
                    "state_category": state_entry.category,
                    "state_is_trivial": state_summary.get("is_trivial", False),

                    # State structured fields (for groupby/filtering)
                    "customer_intent": state_summary["customer_intent"],
                    "customer_sentiment": state_summary["customer_sentiment"],
                    "conversation_stage": state_summary["conversation_stage"],
                    "objections_raised": "; ".join(state_summary["objections_raised"]),
                    "competitive_mentions": "; ".join(state_summary["competitive_mentions"]),
                    "offers_made": "; ".join(state_summary["offers_made"]),

                    # ── Action taxonomy ──
                    "action_id": action_id,
                    "action_category": action_entry.category,
                    "action_is_trivial": action_summary.get("is_trivial", False),

                    # Action structured fields
                    "action_type": action_summary["action_type"],
                    "action_tactic": action_summary["tactic"],
                    "action_tone": action_summary["tone"],
                    "action_personalization": action_summary["personalization_level"],
                })

        elif turn["role"] == "agent":
            conversation_so_far += f"Agent: {turn['text']}\n"

    return rows


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_csv: str, output_dir: str, resume: bool = False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    store_path = output_path / "taxonomy_store"
    store_path.mkdir(exist_ok=True)

    state_store = TaxonomyStore("state")
    action_store = TaxonomyStore("action")
    if resume and (store_path / "state_index.faiss").exists():
        print("Resuming from saved taxonomy...")
        state_store.load(str(store_path))
        action_store.load(str(store_path))
        print(f"  Loaded {len(state_store.entries)} states, {len(action_store.entries)} actions")

    df_input = pd.read_csv(input_csv)
    assert "conversation_id" in df_input.columns, "Need 'conversation_id' column"
    assert "transcript" in df_input.columns, "Need 'transcript' column"
    print(f"Loaded {len(df_input)} transcripts from {input_csv}")

    traj_file = output_path / "trajectories.csv"
    processed_ids = set()
    if resume and traj_file.exists():
        existing = pd.read_csv(traj_file)
        processed_ids = set(existing["conversation_id"].astype(str).unique())
        print(f"  Already processed: {len(processed_ids)} conversations")

    all_rows = []
    total = len(df_input)
    for idx, row in df_input.iterrows():
        conv_id = str(row["conversation_id"])
        if conv_id in processed_ids:
            continue

        print(f"\n{'='*50}")
        print(f"[{idx+1}/{total}] Processing {conv_id}")
        try:
            turn_rows = process_single_transcript(
                conv_id, row["transcript"], state_store, action_store
            )
            all_rows.extend(turn_rows)

            if (idx + 1) % 10 == 0:
                _save_outputs(all_rows, state_store, action_store, output_path, store_path, append=resume)
                print(f"\n  [Checkpoint] Saved at {idx+1} transcripts")

        except Exception as e:
            print(f"  [ERROR] Failed on {conv_id}: {e}")
            continue

    _save_outputs(all_rows, state_store, action_store, output_path, store_path, append=resume)
    _print_summary(state_store, action_store, all_rows)


def _save_outputs(rows, state_store, action_store, output_path, store_path, append=False):
    df_traj = pd.DataFrame(rows)
    traj_file = output_path / "trajectories.csv"
    if append and traj_file.exists():
        existing = pd.read_csv(traj_file)
        df_traj = pd.concat([existing, df_traj], ignore_index=True)
        df_traj = df_traj.drop_duplicates(subset=["conversation_id", "turn_index"])
    df_traj.to_csv(traj_file, index=False)

    state_store.to_dataframe().to_csv(output_path / "state_dictionary.csv", index=False)
    action_store.to_dataframe().to_csv(output_path / "action_dictionary.csv", index=False)

    state_store.save(str(store_path))
    action_store.save(str(store_path))


def _print_summary(state_store, action_store, rows):
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Turns processed:       {len(rows)}")
    print(f"Unique states:         {len(state_store.entries)} (incl. trivial)")
    print(f"Unique actions:        {len(action_store.entries)} (incl. trivial)")

    if rows:
        df = pd.DataFrame(rows)
        n_trivial_s = df["state_is_trivial"].sum()
        n_trivial_a = df["action_is_trivial"].sum()
        print(f"Conversations:         {df['conversation_id'].nunique()}")
        print(f"Trivial state turns:   {n_trivial_s} ({100*n_trivial_s/len(df):.1f}%)")
        print(f"Trivial action turns:  {n_trivial_a} ({100*n_trivial_a/len(df):.1f}%)")

        # Non-trivial distributions
        df_nt = df[~df["state_is_trivial"]]
        if len(df_nt) > 0:
            print(f"\nTop 10 states (non-trivial):")
            print(df_nt["state_category"].value_counts().head(10).to_string())
            print(f"\nTop 10 actions (non-trivial):")
            df_nta = df[~df["action_is_trivial"]]
            print(df_nta["action_category"].value_counts().head(10).to_string())
            print(f"\nSentiment distribution:")
            print(df_nt["customer_sentiment"].value_counts().to_string())

    print("\nOutput files:")
    print("  trajectories.csv       - one row per turn, includes raw text")
    print("  state_dictionary.csv   - state taxonomy with example utterances")
    print("  action_dictionary.csv  - action taxonomy with example utterances")
    print("  taxonomy_store/        - FAISS indices for incremental updates")


# ─────────────────────────────────────────────
# 9. ANALYSIS HELPERS
# ─────────────────────────────────────────────

def load_results(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(output_dir)
    traj = pd.read_csv(p / "trajectories.csv")
    states = pd.read_csv(p / "state_dictionary.csv")
    actions = pd.read_csv(p / "action_dictionary.csv")
    return traj, states, actions


def lookup_action(action_id: int, output_dir: str) -> None:
    """Quick lookup: what does action #N look like?"""
    traj, _, actions = load_results(output_dir)
    entry = actions[actions["action_id"] == action_id].iloc[0]
    print(f"Action #{action_id}: {entry['action_category']}")
    print(f"  Description: {entry['action_description']}")
    print(f"  Key: {entry['action_retrieval_key']}")
    print(f"  Count: {entry['action_count']}")

    # Show example utterances from the dictionary
    examples = json.loads(entry["action_example_utterances"])
    print(f"\n  Example utterances ({len(examples)}):")
    for ex in examples:
        print(f"    [{ex['conversation_id']}]: \"{ex['text'][:200]}\"")

    # Also show from trajectories for full context
    matching = traj[traj["action_id"] == action_id].head(5)
    print(f"\n  Sample turns from trajectories:")
    for _, row in matching.iterrows():
        print(f"    [{row['conversation_id']}, turn {row['turn_index']}]")
        print(f"      Customer said: \"{row['customer_utterance'][:150]}\"")
        print(f"      Agent said:    \"{row['action_text'][:150]}\"")
        print()


def lookup_state(state_id: int, output_dir: str) -> None:
    """Quick lookup: what does state #N look like?"""
    traj, states, _ = load_results(output_dir)
    entry = states[states["state_id"] == state_id].iloc[0]
    print(f"State #{state_id}: {entry['state_category']}")
    print(f"  Description: {entry['state_description']}")
    print(f"  Key: {entry['state_retrieval_key']}")
    print(f"  Count: {entry['state_count']}")

    examples = json.loads(entry["state_example_utterances"])
    print(f"\n  Example utterances ({len(examples)}):")
    for ex in examples:
        print(f"    [{ex['conversation_id']}]: \"{ex['text'][:200]}\"")

    matching = traj[traj["state_id"] == state_id].head(5)
    print(f"\n  Sample turns from trajectories:")
    for _, row in matching.iterrows():
        print(f"    [{row['conversation_id']}, turn {row['turn_index']}]")
        print(f"      Customer: \"{row['customer_utterance'][:150]}\"")
        print(f"      → Agent chose action #{row['action_id']} ({row['action_category']})")
        print()


def build_transition_matrix(traj: pd.DataFrame, exclude_trivial: bool = True) -> pd.DataFrame:
    df = traj.copy()
    if exclude_trivial:
        df = df[~df["state_is_trivial"] & ~df["action_is_trivial"]]

    records = []
    for conv_id, group in df.groupby("conversation_id"):
        group = group.sort_values("turn_index")
        for i in range(len(group) - 1):
            cur = group.iloc[i]
            nxt = group.iloc[i + 1]
            records.append({
                "state_id": cur["state_id"],
                "state_category": cur["state_category"],
                "action_id": cur["action_id"],
                "action_category": cur["action_category"],
                "next_state_id": nxt["state_id"],
                "next_state_category": nxt["state_category"],
            })

    if not records:
        return pd.DataFrame()

    df_t = pd.DataFrame(records)
    trans = (
        df_t.groupby(["state_id", "state_category", "action_id", "action_category",
                       "next_state_id", "next_state_category"])
        .size().reset_index(name="count")
    )
    totals = trans.groupby(["state_id", "action_id"])["count"].transform("sum")
    trans["probability"] = trans["count"] / totals
    return trans.sort_values(["state_id", "action_id", "count"], ascending=[True, True, False])


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build state/action taxonomy from transcripts")
    parser.add_argument("--input", required=True, help="CSV with conversation_id, transcript")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.resume)
