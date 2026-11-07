"""
main.py
=======
SMART CHATBOT — Pure White & Black Minimalist Desktop Interface.
100% Offline AI Assistant powered by Local Vector NLP & Standard Library.
"""

import ast
from datetime import datetime
from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple
import urllib.error
import urllib.request

# Ensure local chatbot package is importable regardless of working directory
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# Attempt to import the offline logic brain from chatbot.py
try:
    import chatbot as _offline_module
    _OFFLINE_BRAIN_CLS = _offline_module.Chatbot
    _OFFLINE_TRY_MATH = _offline_module.try_math
except Exception:
    _OFFLINE_BRAIN_CLS = None
    _OFFLINE_TRY_MATH = None

# Fallback-tolerant vectorizer and numpy imports
try:
    import numpy as np
except ImportError:
    np = None  # Handled transparently by pure-python vectorizer

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _SklearnTfidf
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_sim
    _USE_SKLEARN = True
except ImportError:
    _USE_SKLEARN = False


# ==============================================================================
# ZERO-DEPENDENCY VECTOR ENGINE (Standard Library / NumPy Compatible)
# ==============================================================================
class PurePythonTfidfVectorizer:
    """Lightweight, 100% standard library TF-IDF vectorizer (with 1-3 ngrams)."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)) -> None:
        self.ngram_range = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []

    def _extract_ngrams(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        tokens = []
        for n in range(self.ngram_range[0], min(len(words) + 1, self.ngram_range[1] + 1)):
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i : i + n]))
        return tokens

    def fit_transform(self, corpus: List[str]):
        doc_tokens = [self._extract_ngrams(doc) for doc in corpus]
        vocab: Dict[str, int] = {}
        for tokens in doc_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab

        n_docs = len(corpus)
        n_features = len(vocab)

        df = [0] * n_features
        for tokens in doc_tokens:
            for t in set(tokens):
                df[self.vocab[t]] += 1

        self.idf = [
            math.log((1 + n_docs) / (1 + df[i])) + 1.0 for i in range(n_features)
        ]

        matrix = []
        for tokens in doc_tokens:
            row = [0.0] * n_features
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, cnt in counts.items():
                idx = self.vocab[t]
                row[idx] = cnt * self.idf[idx]
            norm = math.sqrt(sum(val * val for val in row))
            if norm > 0:
                row = [val / norm for val in row]
            matrix.append(row)

        if np is not None:
            return np.array(matrix)
        return matrix

    def transform(self, raw_documents: List[str]):
        n_features = len(self.vocab)
        matrix = []
        for doc in raw_documents:
            tokens = self._extract_ngrams(doc)
            row = [0.0] * n_features
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, cnt in counts.items():
                if t in self.vocab:
                    idx = self.vocab[t]
                    row[idx] = cnt * self.idf[idx]
            norm = math.sqrt(sum(val * val for val in row))
            if norm > 0:
                row = [val / norm for val in row]
            matrix.append(row)

        if np is not None:
            return np.array(matrix)
        return matrix


def calculate_cosine_similarity(query_vec, doc_matrix):
    """Compute cosine similarity scores safely with or without sklearn/numpy."""
    if _USE_SKLEARN:
        return _sklearn_cosine_sim(query_vec, doc_matrix)

    if np is not None and isinstance(query_vec, np.ndarray) and isinstance(doc_matrix, np.ndarray):
        dot = np.dot(doc_matrix, query_vec.T).T
        return dot

    # Pure Python list fallback
    q = query_vec[0] if hasattr(query_vec, "__getitem__") else list(query_vec)
    scores = []
    for doc in doc_matrix:
        dot_product = sum(a * b for a, b in zip(q, doc))
        scores.append(dot_product)
    return [scores]


# ==============================================================================
# 1. LOCAL NLP ENGINE (Zero API Keys)
# ==============================================================================
class LocalChatbot:
    """Core Offline NLP Chatbot Engine with Memory, Vector Matching and Math."""

    def __init__(
        self,
        knowledge_path: Optional[str] = None,
        use_local_llm: bool = False,
        llm_model: str = "llama3.2"
    ):
        self.memory: Dict[str, str] = {}
        self.history: List[Dict[str, str]] = []
        self.use_local_llm = use_local_llm
        self.llm_model = llm_model

        # Integrated offline brain from chatbot.py if available
        self.offline_brain = _OFFLINE_BRAIN_CLS() if _OFFLINE_BRAIN_CLS is not None else None

        self.intents = [
            {
                "tag": "greeting",
                "patterns": ["hello", "hi", "hey", "good morning", "greetings", "whats up"],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hey there! Ready to assist with whatever you need."
                ]
            },
            {
                "tag": "status",
                "patterns": ["how are you", "how is it going", "are you working"],
                "responses": ["All local systems operating smoothly at 100% efficiency!"]
            },
            {
                "tag": "capabilities",
                "patterns": ["what can you do", "help me", "features", "commands", "how do you work"],
                "responses": [
                    "I operate using local vector NLP, memory retention, arithmetic parsing, "
                    "and live intent training—all offline without cloud APIs."
                ]
            },
            {
                "tag": "time",
                "patterns": ["what time is it", "current time", "tell me the date", "today's date"],
                "responses": ["__DYNAMIC_TIME__"]
            },
            {
                "tag": "joke",
                "patterns": ["tell me a joke", "joke", "jokes", "make me laugh", "tell me something funny"],
                "responses": ["__DYNAMIC_JOKE__"]
            },
            {
                "tag": "fact",
                "patterns": ["tell me a fact", "tell me a fun fact", "fun fact", "interesting fact", "trivia"],
                "responses": ["__DYNAMIC_FACT__"]
            },
            {
                "tag": "quote",
                "patterns": ["motivate me", "inspire me", "quote", "motivational quote", "give me inspiration"],
                "responses": ["__DYNAMIC_QUOTE__"]
            },
            {
                "tag": "goodbye",
                "patterns": ["bye", "exit", "quit", "see you later", "goodbye"],
                "responses": ["Have a great day ahead! Feel free to return anytime."]
            }
        ]

        if knowledge_path and Path(knowledge_path).exists():
            self.load_knowledge(knowledge_path)

        self._build_vectorizer()

    def _build_vectorizer(self) -> None:
        self.corpus: List[str] = []
        self.pattern_to_tag: List[str] = []

        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.corpus.append(pattern.lower())
                self.pattern_to_tag.append(intent["tag"])

        if _USE_SKLEARN:
            self.vectorizer = _SklearnTfidf(ngram_range=(1, 3))
        else:
            self.vectorizer = PurePythonTfidfVectorizer(ngram_range=(1, 3))

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def learn(self, tag: str, pattern: str, response: str) -> None:
        for intent in self.intents:
            if intent["tag"] == tag:
                intent["patterns"].append(pattern)
                if response not in intent["responses"]:
                    intent["responses"].append(response)
                self._build_vectorizer()
                return

        self.intents.append({
            "tag": tag,
            "patterns": [pattern],
            "responses": [response]
        })
        self._build_vectorizer()

    def _check_memory_triggers(self, user_input: str) -> Optional[str]:
        name_match = re.search(
            r"\b(?:my name is|call me|i am)\s+([a-zA-Z0-9_-]+)",
            user_input,
            re.IGNORECASE
        )
        if name_match:
            name = name_match.group(1).capitalize()
            self.memory["user_name"] = name
            if self.offline_brain:
                self.offline_brain.user_name = name
            return f"Understood, {name}! I have committed your name to local session memory."

        if re.search(r"\b(?:what is my name|who am i)\b", user_input, re.IGNORECASE):
            if "user_name" in self.memory:
                return f"Your name is recorded as {self.memory['user_name']}."
            return "I don't have your name stored yet. Try saying 'My name is [Name]'."

        return None

    def _evaluate_math(self, user_input: str) -> Optional[str]:
        # First check offline try_math for percentage, sqrt, and word-based expressions
        if _OFFLINE_TRY_MATH:
            try:
                res = _OFFLINE_TRY_MATH(user_input)
                if res:
                    return f"Computed result: {res}"
            except Exception:
                pass

        cleaned = re.sub(r"[^0-9+\-*/().\s]", "", user_input).strip()
        if cleaned and any(op in cleaned for op in ["+", "-", "*", "/"]) and re.search(r"\d", cleaned):
            try:
                result = eval(cleaned, {"__builtins__": None}, {})
                return f"Computed result: {result}"
            except Exception:
                pass
        return None

    def _query_local_ollama(self, user_input: str) -> Optional[str]:
        if not self.use_local_llm:
            return None

        url = "http://localhost:11434/api/chat"
        messages = self.history[-6:] + [{"role": "user", "content": user_input}]
        payload = json.dumps({
            "model": self.llm_model,
            "messages": messages,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("message", {}).get("content")
        except Exception:
            return None

    def _match_intent(self, user_input: str) -> Tuple[Optional[str], float]:
        query = user_input.lower().strip()
        query_vec = self.vectorizer.transform([query])
        scores_matrix = calculate_cosine_similarity(query_vec, self.tfidf_matrix)

        if hasattr(scores_matrix, "flatten"):
            cosine_scores = scores_matrix.flatten()
        else:
            cosine_scores = scores_matrix[0]

        best_score = 0.0
        best_tag = None

        for idx, pattern in enumerate(self.corpus):
            fuzzy_score = SequenceMatcher(None, query, pattern).ratio()
            cos_val = float(cosine_scores[idx]) if idx < len(cosine_scores) else 0.0
            combined_score = (cos_val * 0.65) + (fuzzy_score * 0.35)
            if combined_score > best_score:
                best_score = combined_score
                best_tag = self.pattern_to_tag[idx]

        return best_tag, best_score

    def get_response(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please provide an input query."

        # 1. Memory inspection
        memory_resp = self._check_memory_triggers(user_input)
        if memory_resp:
            self._log_turn(user_input, memory_resp)
            return memory_resp

        # 2. Arithmetic & calculation evaluation
        math_resp = self._evaluate_math(user_input)
        if math_resp:
            self._log_turn(user_input, math_resp)
            return math_resp

        # 3. Vector intent matching
        tag, score = self._match_intent(user_input)
        if tag and score >= 0.42:
            for intent in self.intents:
                if intent["tag"] == tag:
                    if np is not None:
                        resp = np.random.choice(intent["responses"])
                    else:
                        resp = random.choice(intent["responses"])

                    if resp == "__DYNAMIC_TIME__":
                        resp = f"Current system time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}."
                    elif resp == "__DYNAMIC_JOKE__":
                        if self.offline_brain and _offline_module:
                            resp = self.offline_brain.draw("joke", _offline_module.JOKES)
                        else:
                            resp = "Why do programmers prefer dark mode? Because light attracts bugs!"
                    elif resp == "__DYNAMIC_FACT__":
                        if self.offline_brain and _offline_module:
                            resp = "Fun fact: " + self.offline_brain.draw("fact", _offline_module.FACTS)
                        else:
                            resp = "Fun fact: Octopuses have three hearts and blue blood."
                    elif resp == "__DYNAMIC_QUOTE__":
                        if self.offline_brain and _offline_module:
                            resp = self.offline_brain.draw("quote", _offline_module.QUOTES)
                        else:
                            resp = "Small steps every day add up to big results."
                    if "user_name" in self.memory and tag == "greeting":
                        resp = f"Welcome back, {self.memory['user_name']}! " + resp
                    self._log_turn(user_input, resp)
                    return resp

        # 4. Local Ollama LLM pipeline (if enabled)
        llm_resp = self._query_local_ollama(user_input)
        if llm_resp:
            self._log_turn(user_input, llm_resp)
            return llm_resp

        # 5. Integrated offline logic brain (jokes, quotes, facts, small talk, games)
        if self.offline_brain:
            try:
                offline_reply = self.offline_brain.reply(user_input)
                # If a rule was matched (not an unhandled fallback)
                if self.offline_brain.last_intent is not None:
                    self._log_turn(user_input, offline_reply)
                    return offline_reply
            except Exception:
                pass

        # 6. Fallback guidance
        fallback = (
            "No direct local intent matched this query. "
            "Teach me this pattern in the 'Intent Studio' panel on the right."
        )
        self._log_turn(user_input, fallback)
        return fallback

    def _log_turn(self, user_msg: str, bot_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": bot_msg})

    def save_knowledge(self, path: str = "knowledge.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.intents, f, indent=2)

    def load_knowledge(self, path: str = "knowledge.json") -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)
        self._build_vectorizer()


# ==============================================================================
# 2. PURE WHITE & BLACK MINIMALIST INTERFACE
# ==============================================================================
class SmartChatbotUI:
    """Professional White & Black Minimalist Desktop Chatbot Interface."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SMART CHATBOT — Offline Neural Interface")
        self.root.geometry("1280x780")
        self.root.minsize(1080, 650)

        # Strictly pure white & black color palette as required
        self.colors = {
            "app_bg": "#FFFFFF",           # Main application background
            "chat_bg": "#FFFFFF",          # Chat workspace background
            "sidebar_bg": "#F7F7F7",       # Sidebar background
            "card_bg": "#FFFFFF",          # Cards and panels
            "btn_primary_bg": "#000000",   # Primary buttons
            "btn_primary_fg": "#FFFFFF",   # Primary button text
            "btn_secondary_bg": "#F2F2F2", # Secondary buttons
            "btn_secondary_fg": "#000000", # Secondary button text
            "text_main": "#000000",        # Main text
            "text_secondary": "#555555",   # Secondary text
            "border": "#E5E5E5",           # Borders
            "user_bubble_bg": "#F2F2F2",   # User message background
            "user_bubble_fg": "#000000",   # User message text
            "bot_bubble_bg": "#FFFFFF",    # Bot message background
            "bot_bubble_fg": "#000000",    # Bot message text
            "input_bg": "#FFFFFF",         # Input field background
            "input_fg": "#000000",         # Input field text
            "placeholder": "#888888",      # Placeholder text
            "hover_bg": "#EAEAEA",         # Hover background
            "focus_border": "#000000",     # Focus border
            # Interactive hover states
            "btn_primary_hover": "#222222",
            "btn_primary_active": "#333333",
            "btn_secondary_hover": "#EAEAEA",
            "btn_secondary_active": "#DCDCDC",
            # Backwards-compatible aliases
            "canvas_bg": "#FFFFFF",
            "card_border": "#E5E5E5",
            "input_surface": "#FFFFFF",
            "accent_cyan": "#000000",
            "accent_cyan_glow": "#222222",
            "accent_violet": "#000000",
            "accent_green": "#000000",
            "text_bright": "#000000",
            "text_dim": "#555555",
            "bubble_user": "#F2F2F2",
            "bubble_bot": "#FFFFFF",
        }

        self.root.configure(bg=self.colors["app_bg"])
        self.bot = LocalChatbot(knowledge_path="knowledge.json", use_local_llm=False)

        self.placeholder_text = "Type your query here... (Press Enter to send)"
        self.typing_job = None
        self.typing_frame = None
        self._active_nav = "chat"
        self._bubble_labels = []

        self._build_interface()
        self._refresh_inspectors()

        # Initial Welcome Message
        self._append_message(
            "Bot",
            "Welcome to SMART CHATBOT. Running 100% offline with zero cloud dependencies. "
            "You can query system time, evaluate arithmetic expressions, retain session memory, "
            "or train custom vector intents dynamically."
        )

    # --------------------------------------------------------------------------
    # Layout Builder
    # --------------------------------------------------------------------------
    def _build_interface(self) -> None:
        self.root.grid_columnconfigure(0, weight=0, minsize=280)  # Left Sidebar
        self.root.grid_columnconfigure(1, weight=1)               # Center Workspace
        self.root.grid_columnconfigure(2, weight=0, minsize=330)  # Right Intent Studio
        self.root.grid_rowconfigure(0, weight=1)

        self._build_left_deck()
        self._build_center_chat()
        self._build_right_studio()

    def _create_card(self, parent, **kwargs) -> tk.Frame:
        """Create clean minimalist cards with subtle borders."""
        return tk.Frame(
            parent,
            bg=self.colors["card_bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            **kwargs
        )

    # --------------------------------------------------------------------------
    # 1. Left Sidebar
    # --------------------------------------------------------------------------
    def _build_left_deck(self) -> None:
        sidebar = tk.Frame(
            self.root,
            bg=self.colors["sidebar_bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=18,
            pady=20
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        # Header branding with minimal black logo
        brand_frame = tk.Frame(sidebar, bg=self.colors["sidebar_bg"])
        brand_frame.pack(fill="x", pady=(0, 14))

        # Minimal black logo badge
        logo_canvas = tk.Canvas(
            brand_frame,
            width=24,
            height=24,
            bg=self.colors["sidebar_bg"],
            highlightthickness=0
        )
        logo_canvas.pack(side="left", padx=(0, 10))
        logo_canvas.create_rectangle(1, 1, 23, 23, fill="#000000", outline="#000000")
        logo_canvas.create_oval(8, 8, 16, 16, fill="#FFFFFF", outline="#FFFFFF")

        title_col = tk.Frame(brand_frame, bg=self.colors["sidebar_bg"])
        title_col.pack(side="left", fill="x")

        tk.Label(
            title_col,
            text="SMART CHATBOT",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_main"]
        ).pack(anchor="w")

        status_sub = tk.Frame(title_col, bg=self.colors["sidebar_bg"])
        status_sub.pack(anchor="w", pady=(2, 0))

        tk.Label(
            status_sub,
            text="●",
            font=("Segoe UI", 8),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_main"]
        ).pack(side="left")

        tk.Label(
            status_sub,
            text=" 100% Offline Engine Active",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_secondary"]
        ).pack(side="left")

        # Subtle separator
        tk.Frame(sidebar, height=1, bg=self.colors["border"]).pack(fill="x", pady=12)

        # Navigation Section
        tk.Label(
            sidebar,
            text="NAVIGATION",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", pady=(0, 6))

        self.nav_items_frame = tk.Frame(sidebar, bg=self.colors["sidebar_bg"])
        self.nav_items_frame.pack(fill="x", pady=(0, 8))

        self.nav_buttons: Dict[str, Tuple[tk.Frame, tk.Frame, tk.Label]] = {}
        nav_defs = [
            ("chat", "💬", "Chat Conversation"),
            ("studio", "⚙", "Intent Studio"),
            ("status", "ℹ", "System Runtime")
        ]

        for key, icon, label_text in nav_defs:
            row = tk.Frame(self.nav_items_frame, bg=self.colors["sidebar_bg"], cursor="hand2")
            row.pack(fill="x", pady=2)

            indicator = tk.Frame(
                row,
                width=3,
                bg="#000000" if key == self._active_nav else self.colors["sidebar_bg"]
            )
            indicator.pack(side="left", fill="y")

            lbl = tk.Label(
                row,
                text=f"  {icon}  {label_text}",
                font=("Segoe UI", 9, "bold" if key == self._active_nav else "normal"),
                bg=self.colors["sidebar_bg"],
                fg=self.colors["text_main"],
                anchor="w",
                padx=8,
                pady=6
            )
            lbl.pack(side="left", fill="x", expand=True)

            self.nav_buttons[key] = (row, indicator, lbl)

            # Bind hover & click
            for w in (row, lbl):
                w.bind("<Enter>", lambda _, r=row, l=lbl: self._on_nav_hover(r, l, True))
                w.bind("<Leave>", lambda _, r=row, l=lbl, k=key: self._on_nav_hover(r, l, False, k))
                w.bind("<Button-1>", lambda _, k=key: self._set_active_nav(k))

        # Subtle separator
        tk.Frame(sidebar, height=1, bg=self.colors["border"]).pack(fill="x", pady=10)

        # Core Runtime Toggle
        tk.Label(
            sidebar,
            text="CORE RUNTIME",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", pady=(0, 6))

        self.ollama_var = tk.BooleanVar(value=self.bot.use_local_llm)
        ollama_toggle = tk.Checkbutton(
            sidebar,
            text="Local Ollama Pipeline",
            variable=self.ollama_var,
            command=self._toggle_ollama,
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_main"],
            selectcolor=self.colors["card_bg"],
            activebackground=self.colors["sidebar_bg"],
            activeforeground=self.colors["text_main"],
            font=("Segoe UI", 9),
            cursor="hand2"
        )
        ollama_toggle.pack(anchor="w")

        # Subtle separator
        tk.Frame(sidebar, height=1, bg=self.colors["border"]).pack(fill="x", pady=10)

        # Quick Actions Section
        tk.Label(
            sidebar,
            text="ONE-CLICK PROMPTS",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", pady=(0, 8))

        prompts = [
            ("⚡ Introduce System", "What can you do?"),
            ("🕒 Fetch Current Time", "What time is it right now?"),
            ("📊 Math Calculation", "Calculate (150 * 4) + (250 / 5)"),
            ("💾 Set User Persona", "My name is Commander"),
            ("🔎 Check User Persona", "What is my name?"),
            ("👋 Diagnostic Ping", "How are you doing?"),
            ("😄 Tell a Joke", "Tell me a joke"),
            ("💡 Fun Fact", "Tell me a fun fact")
        ]

        prompts_container = tk.Frame(sidebar, bg=self.colors["sidebar_bg"])
        prompts_container.pack(fill="both", expand=True)

        for title, prompt_val in prompts:
            btn = tk.Button(
                prompts_container,
                text=title,
                anchor="w",
                padx=12,
                pady=6,
                bg=self.colors["btn_secondary_bg"],
                fg=self.colors["btn_secondary_fg"],
                activebackground=self.colors["btn_secondary_active"],
                activeforeground=self.colors["btn_secondary_fg"],
                relief="flat",
                bd=0,
                font=("Segoe UI", 9),
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=self.colors["border"],
                command=lambda p=prompt_val: self._send_quick_prompt(p)
            )
            btn.pack(fill="x", pady=2)
            self._add_secondary_hover(btn)

        # Bottom Utilities
        bot_actions = tk.Frame(sidebar, bg=self.colors["sidebar_bg"])
        bot_actions.pack(side="bottom", fill="x", pady=(12, 0))

        save_btn = tk.Button(
            bot_actions,
            text="Save Knowledge Matrix",
            pady=8,
            bg=self.colors["btn_primary_bg"],
            fg=self.colors["btn_primary_fg"],
            activebackground=self.colors["btn_primary_active"],
            activeforeground=self.colors["btn_primary_fg"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._save_knowledge
        )
        save_btn.pack(fill="x", pady=(0, 6))
        self._add_primary_hover(save_btn)

        clear_btn = tk.Button(
            bot_actions,
            text="Clear Chat Feed",
            pady=7,
            bg=self.colors["btn_secondary_bg"],
            fg=self.colors["btn_secondary_fg"],
            activebackground=self.colors["btn_secondary_active"],
            activeforeground=self.colors["btn_secondary_fg"],
            font=("Segoe UI", 9),
            relief="flat",
            bd=0,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            command=self._clear_chat
        )
        clear_btn.pack(fill="x")
        self._add_secondary_hover(clear_btn)

    def _on_nav_hover(self, row: tk.Frame, lbl: tk.Label, is_hover: bool, key: str = "") -> None:
        if is_hover:
            row.configure(bg=self.colors["hover_bg"])
            lbl.configure(bg=self.colors["hover_bg"])
        else:
            bg_color = self.colors["sidebar_bg"]
            row.configure(bg=bg_color)
            lbl.configure(bg=bg_color)

    def _set_active_nav(self, nav_name: str) -> None:
        self._active_nav = nav_name
        for key, (_row, indicator, lbl) in self.nav_buttons.items():
            if key == nav_name:
                indicator.configure(bg="#000000")
                lbl.configure(font=("Segoe UI", 9, "bold"))
            else:
                indicator.configure(bg=self.colors["sidebar_bg"])
                lbl.configure(font=("Segoe UI", 9, "normal"))

        if nav_name == "studio":
            self.pattern_entry.focus()
        elif nav_name == "chat":
            self.user_entry.focus()

    # --------------------------------------------------------------------------
    # 2. Center Chat Workspace
    # --------------------------------------------------------------------------
    def _build_center_chat(self) -> None:
        container = tk.Frame(self.root, bg=self.colors["chat_bg"])
        container.grid(row=0, column=1, sticky="nsew", padx=6, pady=16)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Header Bar
        header = self._create_card(container, padx=20, pady=12)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            header,
            text="WORKSPACE CANVAS",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_main"]
        ).pack(side="left")

        self.stats_pill = tk.Label(
            header,
            text="READY",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["btn_secondary_bg"],
            fg=self.colors["text_main"],
            padx=10,
            pady=4,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        self.stats_pill.pack(side="right")

        # Chat Bubble Viewport
        chat_card = self._create_card(container, padx=12, pady=12)
        chat_card.grid(row=1, column=0, sticky="nsew")
        chat_card.grid_rowconfigure(0, weight=1)
        chat_card.grid_columnconfigure(0, weight=1)

        self.chat_display = tk.Text(
            chat_card,
            bg=self.colors["card_bg"],
            fg=self.colors["text_main"],
            font=("Segoe UI", 10),
            wrap="word",
            relief="flat",
            state="disabled",
            padx=10,
            pady=10,
            highlightthickness=0
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        scroll = tk.Scrollbar(
            chat_card,
            command=self.chat_display.yview,
            bg=self.colors["btn_secondary_bg"],
            width=10,
            relief="flat"
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.chat_display.configure(yscrollcommand=scroll.set)

        # Tags for text content compatibility
        self.chat_display.tag_configure(
            "user_tag",
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 8, "bold")
        )
        self.chat_display.tag_configure(
            "bot_tag",
            foreground=self.colors["text_main"],
            font=("Segoe UI", 8, "bold")
        )
        self.chat_display.tag_configure(
            "user_body",
            foreground=self.colors["user_bubble_fg"],
            background=self.colors["user_bubble_bg"],
            font=("Segoe UI", 10)
        )
        self.chat_display.tag_configure(
            "bot_body",
            foreground=self.colors["bot_bubble_fg"],
            background=self.colors["bot_bubble_bg"],
            font=("Segoe UI", 10)
        )

        # Responsive bubble wraplength listener
        self.chat_display.bind("<Configure>", self._on_chat_resize)

        # Elevated Input Dock
        self.input_dock = self._create_card(container, padx=14, pady=10)
        self.input_dock.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.input_dock.grid_columnconfigure(0, weight=1)

        self.user_entry = tk.Entry(
            self.input_dock,
            bg=self.colors["input_bg"],
            fg=self.colors["placeholder"],
            insertbackground=self.colors["focus_border"],
            font=("Segoe UI", 10),
            relief="flat",
            highlightthickness=0
        )
        self.user_entry.insert(0, self.placeholder_text)
        self.user_entry.grid(row=0, column=0, sticky="ew", ipady=8, padx=(4, 12))

        # Focus & placeholder bindings
        self.user_entry.bind("<FocusIn>", self._on_entry_focus)
        self.user_entry.bind("<FocusOut>", self._on_entry_blur)
        self.user_entry.bind("<Return>", lambda _: self._handle_send())

        self.send_btn = tk.Button(
            self.input_dock,
            text="SEND  ➔",
            padx=20,
            pady=7,
            bg=self.colors["btn_primary_bg"],
            fg=self.colors["btn_primary_fg"],
            activebackground=self.colors["btn_primary_active"],
            activeforeground=self.colors["btn_primary_fg"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._handle_send
        )
        self.send_btn.grid(row=0, column=1)
        self._add_primary_hover(self.send_btn)

    def _on_chat_resize(self, event) -> None:
        """Dynamically adjust message wraplength when workspace size changes."""
        calc_wrap = max(280, event.width - 120)
        for lbl in self._bubble_labels:
            if lbl.winfo_exists():
                lbl.configure(wraplength=calc_wrap)

    def _on_entry_focus(self, _) -> None:
        self.input_dock.configure(highlightbackground=self.colors["focus_border"], highlightthickness=1)
        if self.user_entry.get() == self.placeholder_text:
            self.user_entry.delete(0, "end")
            self.user_entry.configure(fg=self.colors["input_fg"])

    def _on_entry_blur(self, _) -> None:
        self.input_dock.configure(highlightbackground=self.colors["border"], highlightthickness=1)
        if not self.user_entry.get().strip():
            self.user_entry.delete(0, "end")
            self.user_entry.insert(0, self.placeholder_text)
            self.user_entry.configure(fg=self.colors["placeholder"])

    # --------------------------------------------------------------------------
    # 3. Right Intent Studio
    # --------------------------------------------------------------------------
    def _build_right_studio(self) -> None:
        card = self._create_card(self.root, padx=20, pady=20)
        card.grid(row=0, column=2, sticky="nsew", padx=(6, 16), pady=16)

        # Studio Header
        tk.Label(
            card,
            text="INTENT STUDIO",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_main"]
        ).pack(anchor="w")

        tk.Label(
            card,
            text="Expand vector dictionary locally:",
            font=("Segoe UI", 8),
            bg=self.colors["card_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", pady=(2, 14))

        # Inputs
        tk.Label(
            card,
            text="USER TRIGGER PHRASE",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w")

        self.pattern_entry_container = tk.Frame(
            card,
            bg=self.colors["input_bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        self.pattern_entry_container.pack(fill="x", pady=(4, 12))

        self.pattern_entry = tk.Entry(
            self.pattern_entry_container,
            bg=self.colors["input_bg"],
            fg=self.colors["input_fg"],
            insertbackground=self.colors["focus_border"],
            relief="flat",
            font=("Segoe UI", 9)
        )
        self.pattern_entry.pack(fill="x", ipady=6, padx=8)
        self._add_focus_border(self.pattern_entry, self.pattern_entry_container)

        tk.Label(
            card,
            text="RESPONSE PAYLOAD",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w")

        self.response_entry_container = tk.Frame(
            card,
            bg=self.colors["input_bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        self.response_entry_container.pack(fill="x", pady=(4, 14))

        self.response_entry = tk.Entry(
            self.response_entry_container,
            bg=self.colors["input_bg"],
            fg=self.colors["input_fg"],
            insertbackground=self.colors["focus_border"],
            relief="flat",
            font=("Segoe UI", 9)
        )
        self.response_entry.pack(fill="x", ipady=6, padx=8)
        self._add_focus_border(self.response_entry, self.response_entry_container)

        train_btn = tk.Button(
            card,
            text="+ Inject Pattern",
            pady=8,
            bg=self.colors["btn_primary_bg"],
            fg=self.colors["btn_primary_fg"],
            activebackground=self.colors["btn_primary_active"],
            activeforeground=self.colors["btn_primary_fg"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._train_bot
        )
        train_btn.pack(fill="x")
        self._add_primary_hover(train_btn)

        # Subtle separator
        tk.Frame(card, height=1, bg=self.colors["border"]).pack(fill="x", pady=16)

        # Live State Monitors
        tk.Label(
            card,
            text="SESSION ENTITY MEMORY",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_main"]
        ).pack(anchor="w", pady=(0, 6))

        self.memory_box = tk.Listbox(
            card,
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_main"],
            selectbackground=self.colors["hover_bg"],
            selectforeground=self.colors["text_main"],
            font=("Consolas", 9),
            height=4,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        self.memory_box.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text="ACTIVE VECTOR REGISTRY",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_main"]
        ).pack(anchor="w", pady=(0, 6))

        self.intents_box = tk.Listbox(
            card,
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_secondary"],
            selectbackground=self.colors["hover_bg"],
            selectforeground=self.colors["text_main"],
            font=("Consolas", 9),
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        self.intents_box.pack(fill="both", expand=True)

    def _add_focus_border(self, entry: tk.Entry, container: tk.Frame) -> None:
        entry.bind("<FocusIn>", lambda _: container.configure(highlightbackground=self.colors["focus_border"]))
        entry.bind("<FocusOut>", lambda _: container.configure(highlightbackground=self.colors["border"]))

    # --------------------------------------------------------------------------
    # Interactive Feedback & Hover Effects
    # --------------------------------------------------------------------------
    def _add_hover_effect(self, widget: tk.Widget, default_bg: str, hover_bg: str) -> None:
        """Generic hover configuration."""
        widget.bind("<Enter>", lambda _: widget.configure(bg=hover_bg))
        widget.bind("<Leave>", lambda _: widget.configure(bg=default_bg))

    def _add_primary_hover(self, btn: tk.Button) -> None:
        """Hover styling for solid black primary buttons."""
        btn.bind("<Enter>", lambda _: btn.configure(bg=self.colors["btn_primary_hover"]))
        btn.bind("<Leave>", lambda _: btn.configure(bg=self.colors["btn_primary_bg"]))
        btn.bind("<Button-1>", lambda _: btn.configure(bg=self.colors["btn_primary_active"]))
        btn.bind("<ButtonRelease-1>", lambda _: btn.configure(bg=self.colors["btn_primary_hover"]))

    def _add_secondary_hover(self, btn: tk.Button) -> None:
        """Hover styling for light grey secondary buttons."""
        btn.bind("<Enter>", lambda _: btn.configure(bg=self.colors["btn_secondary_hover"]))
        btn.bind("<Leave>", lambda _: btn.configure(bg=self.colors["btn_secondary_bg"]))
        btn.bind("<Button-1>", lambda _: btn.configure(bg=self.colors["btn_secondary_active"]))
        btn.bind("<ButtonRelease-1>", lambda _: btn.configure(bg=self.colors["btn_secondary_hover"]))

    # --------------------------------------------------------------------------
    # Messaging & Bubble Rendering
    # --------------------------------------------------------------------------
    def _append_message(self, sender: str, message: str) -> None:
        self.chat_display.configure(state="normal")
        timestamp = datetime.now().strftime("%I:%M %p")

        # Container row spanning the chat workspace
        row = tk.Frame(self.chat_display, bg=self.colors["chat_bg"], padx=4, pady=3)

        chat_width = self.chat_display.winfo_width()
        wrap_limit = max(280, (chat_width - 120) if chat_width > 120 else 460)

        if sender == "You":
            # Right-aligned bubble container
            align_box = tk.Frame(row, bg=self.colors["chat_bg"])
            align_box.pack(side="right", anchor="e")

            # User Bubble: Background #F2F2F2, border #E5E5E5
            bubble = tk.Frame(
                align_box,
                bg=self.colors["user_bubble_bg"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                padx=14,
                pady=9
            )
            bubble.pack(anchor="e")

            meta_row = tk.Frame(bubble, bg=self.colors["user_bubble_bg"])
            meta_row.pack(anchor="e", fill="x", pady=(0, 2))

            tk.Label(
                meta_row,
                text="YOU",
                font=("Segoe UI", 8, "bold"),
                bg=self.colors["user_bubble_bg"],
                fg=self.colors["text_secondary"]
            ).pack(side="left")

            tk.Label(
                meta_row,
                text=f" • {timestamp}",
                font=("Segoe UI", 8),
                bg=self.colors["user_bubble_bg"],
                fg=self.colors["placeholder"]
            ).pack(side="left")

            body_lbl = tk.Label(
                bubble,
                text=message,
                font=("Segoe UI", 10),
                bg=self.colors["user_bubble_bg"],
                fg=self.colors["user_bubble_fg"],
                justify="left",
                wraplength=wrap_limit
            )
            body_lbl.pack(anchor="w")
            self._bubble_labels.append(body_lbl)

            # Insert tag record for accessibility / programmatic reading
            self.chat_display.insert("end", f"\nYOU  •  {timestamp}\n", "user_tag")
            self.chat_display.insert("end", f"  {message}  \n", "user_body")

        else:
            # Left-aligned bubble container
            align_box = tk.Frame(row, bg=self.colors["chat_bg"])
            align_box.pack(side="left", anchor="w")

            # Bot Bubble: Background #FFFFFF, border #E5E5E5
            bubble = tk.Frame(
                align_box,
                bg=self.colors["bot_bubble_bg"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                padx=14,
                pady=9
            )
            bubble.pack(anchor="w")

            meta_row = tk.Frame(bubble, bg=self.colors["bot_bubble_bg"])
            meta_row.pack(anchor="w", fill="x", pady=(0, 2))

            tk.Label(
                meta_row,
                text="SMART CHATBOT",
                font=("Segoe UI", 8, "bold"),
                bg=self.colors["bot_bubble_bg"],
                fg=self.colors["text_main"]
            ).pack(side="left")

            tk.Label(
                meta_row,
                text=f" • {timestamp}",
                font=("Segoe UI", 8),
                bg=self.colors["bot_bubble_bg"],
                fg=self.colors["placeholder"]
            ).pack(side="left")

            body_lbl = tk.Label(
                bubble,
                text=message,
                font=("Segoe UI", 10),
                bg=self.colors["bot_bubble_bg"],
                fg=self.colors["bot_bubble_fg"],
                justify="left",
                wraplength=wrap_limit
            )
            body_lbl.pack(anchor="w")
            self._bubble_labels.append(body_lbl)

            # Insert tag record for accessibility / programmatic reading
            self.chat_display.insert("end", f"\nSMART CHATBOT  •  {timestamp}\n", "bot_tag")
            self.chat_display.insert("end", f"  {message}  \n", "bot_body")

        self.chat_display.window_create("end", window=row)
        self.chat_display.insert("end", "\n")
        self.chat_display.configure(state="disabled")
        self._smooth_scroll_to_end()

    # --------------------------------------------------------------------------
    # Animations & Loading Indicators
    # --------------------------------------------------------------------------
    def _show_typing_indicator(self) -> None:
        """Display smooth animated typing indicator bubble."""
        self._remove_typing_indicator()

        self.chat_display.configure(state="normal")
        self.typing_frame = tk.Frame(self.chat_display, bg=self.colors["chat_bg"], padx=4, pady=3)

        bubble = tk.Frame(
            self.typing_frame,
            bg=self.colors["bot_bubble_bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=14,
            pady=8
        )
        bubble.pack(side="left", anchor="w")

        tk.Label(
            bubble,
            text="SMART CHATBOT",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["bot_bubble_bg"],
            fg=self.colors["text_main"]
        ).pack(anchor="w", pady=(0, 2))

        self.dots_label = tk.Label(
            bubble,
            text="Thinking  ●  ○  ○",
            font=("Segoe UI", 9),
            bg=self.colors["bot_bubble_bg"],
            fg=self.colors["text_secondary"]
        )
        self.dots_label.pack(anchor="w")

        self.chat_display.window_create("end", window=self.typing_frame)
        self.chat_display.insert("end", "\n")
        self.chat_display.configure(state="disabled")
        self._smooth_scroll_to_end()

        # Animate dots
        self._dot_step = 0
        self._animate_typing_dots()

    def _animate_typing_dots(self) -> None:
        if not self.typing_frame or not self.typing_frame.winfo_exists():
            return
        frames = [
            "Thinking  ●  ○  ○",
            "Thinking  ○  ●  ○",
            "Thinking  ○  ○  ●",
            "Thinking  ●  ●  ●"
        ]
        self._dot_step = (self._dot_step + 1) % len(frames)
        if hasattr(self, "dots_label") and self.dots_label.winfo_exists():
            self.dots_label.configure(text=frames[self._dot_step])
            self.typing_job = self.root.after(110, self._animate_typing_dots)

    def _remove_typing_indicator(self) -> None:
        if self.typing_job:
            try:
                self.root.after_cancel(self.typing_job)
            except Exception:
                pass
            self.typing_job = None

        if self.typing_frame and self.typing_frame.winfo_exists():
            self.chat_display.configure(state="normal")
            self.typing_frame.destroy()
            self.typing_frame = None
            self.chat_display.configure(state="disabled")

    def _smooth_scroll_to_end(self, step: int = 0) -> None:
        """Ensure reliable auto-scrolling to the latest message."""
        self.chat_display.see("end")
        if step < 2:
            self.root.after(40, lambda: self._smooth_scroll_to_end(step + 1))

    # --------------------------------------------------------------------------
    # Handlers & Actions
    # --------------------------------------------------------------------------
    def _handle_send(self) -> None:
        text = self.user_entry.get().strip()
        if not text or text == self.placeholder_text:
            return

        self.user_entry.delete(0, "end")
        self._append_message("You", text)

        # Show typing indicator, calculate reply, and schedule smooth presentation
        self._show_typing_indicator()

        def deliver_reply():
            self._remove_typing_indicator()
            reply = self.bot.get_response(text)
            self._append_message("Bot", reply)
            self._refresh_inspectors()

        # Slight delay to create a polished, natural interaction cadence
        self.root.after(280, deliver_reply)

    def _send_quick_prompt(self, text: str) -> None:
        if self.user_entry.get() == self.placeholder_text:
            self.user_entry.delete(0, "end")
        self.user_entry.delete(0, "end")
        self.user_entry.insert(0, text)
        self.user_entry.configure(fg=self.colors["input_fg"])
        self._handle_send()

    def _train_bot(self) -> None:
        pat = self.pattern_entry.get().strip()
        res = self.response_entry.get().strip()
        if not pat or not res:
            messagebox.showwarning("Warning", "Both pattern and response fields are required.")
            return

        tag = f"custom_{len(self.bot.intents) + 1}"
        self.bot.learn(tag=tag, pattern=pat, response=res)
        self.pattern_entry.delete(0, "end")
        self.response_entry.delete(0, "end")

        self._append_message("Bot", f"Pattern registered successfully into vector cluster [{tag}].")
        self._refresh_inspectors()

    def _refresh_inspectors(self) -> None:
        turns = len(self.bot.history) // 2
        self.stats_pill.configure(text=f"{len(self.bot.intents)} INTENTS | {turns} TURNS")

        self.memory_box.delete(0, "end")
        if not self.bot.memory:
            self.memory_box.insert("end", " [Empty session memory]")
        else:
            for k, v in self.bot.memory.items():
                self.memory_box.insert("end", f" {k} -> {v}")

        self.intents_box.delete(0, "end")
        for intent in self.bot.intents:
            count = len(intent.get("patterns", []))
            self.intents_box.insert("end", f" • {intent['tag']} ({count} patterns)")

    def _toggle_ollama(self) -> None:
        self.bot.use_local_llm = self.ollama_var.get()

    def _save_knowledge(self) -> None:
        self.bot.save_knowledge("knowledge.json")
        messagebox.showinfo("Saved", "Knowledge matrix written to knowledge.json")

    def _clear_chat(self) -> None:
        self._remove_typing_indicator()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self._bubble_labels.clear()


# Backward-compatible alias for existing imports / test suites
ModernLandscapeUI = SmartChatbotUI


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartChatbotUI(root)
    root.mainloop()
