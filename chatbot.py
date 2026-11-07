"""
chatbot.py
==========
The "brain" of the Offline Chatbot (v2.0).

100% offline: standard library only - no AI models, no cloud, no API keys.

This file contains ONLY chatbot logic (no Tkinter / GUI code), so it can be
used from the graphical app (main.py) or directly in the terminal:

    python chatbot.py

What's new in v2.0
------------------
* Scored intent matching (the most specific phrase wins, not just the first).
* Typo tolerance:  "helo", "thnks", "wats up", "heyyyy" all still work.
* Memory: remembers your name (and, optionally, between sessions).
* Context: "another one" after a joke gives you another joke.
* Built-in skills: time, date, calculator, jokes, facts, quotes,
  coin flip, dice, random numbers.
* Greeting + question in one message ("hi, how are you?") is handled.
* A caring reply if someone mentions self-harm.
"""

import ast
import datetime as _dt
import difflib
import json
import math
import operator
import os
import random
import re

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BOT_NAME = "ChatBot"
VERSION = "2.0"

FAREWELL_REPLIES = [
    "Goodbye! Have a great day!",
    "See you later! Take care!",
    "Bye for now! Come back anytime.",
]
FAREWELL_REPLY = FAREWELL_REPLIES[0]   # kept for backwards compatibility
DEFAULT_REPLIES = [
    "Sorry, I didn't understand that. Can you try again? (Say 'help' to see what I can do.)",
    "Hmm, I'm not sure about that one. Try 'help' to see what I know!",
    "I didn't catch that. I can chat, tell jokes, do maths and more - say 'help'.",
]
DEFAULT_REPLY = DEFAULT_REPLIES[0]     # kept for backwards compatibility

# Words that close the chatbot.
EXIT_COMMANDS = ("exit", "quit", "close", "exit chat", "quit chat")


def welcome_text(name: str = "") -> str:
    """Welcome message shown when the chat starts."""
    if name:
        return (
            f"Welcome back, {name}! I'm {BOT_NAME}.\n"
            "Say 'help' to see what I can do, or type 'exit' to quit."
        )
    return (
        f"Hello! I'm {BOT_NAME}, your offline virtual assistant.\n"
        "Try 'help' to see what I can do, or type 'exit' to quit."
    )


WELCOME_MESSAGE = welcome_text()

# --------------------------------------------------------------------------
# Content banks
# --------------------------------------------------------------------------
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the developer go broke? He used up all his cache.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    "Why was the computer cold? It left its Windows open.",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    "Why do Java developers wear glasses? Because they can't C#.",
    "What's a computer's favourite snack? Microchips!",
    "How many programmers does it take to change a light bulb? None - that's a hardware problem.",
    "Debugging: being the detective in a crime movie where you are also the murderer.",
]

FACTS = [
    "Honey never spoils - edible honey has been found in ancient Egyptian tombs.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than a year on Venus.",
    "Bananas are berries, but strawberries are not.",
    "Sharks existed before trees did.",
    "Wombat droppings are cube-shaped.",
    "The first computer 'bug' was a real moth found in a Harvard computer in 1947.",
    "Python is named after Monty Python's Flying Circus, not the snake.",
    "Sunlight takes about 8 minutes to reach Earth.",
    "The Eiffel Tower grows a few centimetres taller in summer because metal expands in heat.",
]

QUOTES = [
    "Small steps every day add up to big results.",
    "You don't have to be perfect to make progress.",
    "Mistakes are proof that you are trying.",
    "Start where you are. Use what you have. Do what you can.",
    "Done is better than perfect.",
    "Every expert was once a beginner.",
    "Tough days build strong people. Keep going!",
    "The best time to start was yesterday. The next best time is now.",
]

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Clean user input so it is easy to compare.

    Example: "  How ARE you?? " -> "how are you"
    """
    text = text.lower().strip()
    text = text.replace("'", "").replace("\u2019", "")  # what's -> whats
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # remove punctuation
    text = re.sub(r"\s+", " ", text)                    # collapse spaces
    return text.strip()


def contains_phrase(text: str, phrases: list) -> bool:
    """Return True if any phrase appears in `text` as whole word(s).

    Padding with spaces avoids false matches, e.g. the word "hi" must not
    match inside "this" or "which".
    """
    padded_text = f" {text} "
    for phrase in phrases:
        if f" {normalize(phrase)} " in padded_text:
            return True
    return False


def best_phrase_length(text: str, phrases: list) -> int:
    """Number of words in the longest matching phrase (0 = no match)."""
    padded_text = f" {text} "
    best = 0
    for phrase in phrases:
        clean = normalize(phrase)
        if clean and f" {clean} " in padded_text:
            best = max(best, len(clean.split()))
    return best


# --------------------------------------------------------------------------
# Safe calculator (no eval(): only numbers and + - * / % ** are allowed)
# --------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def safe_eval(expression: str) -> float:
    """Evaluate a plain arithmetic expression safely."""
    return _eval_node(ast.parse(expression.strip(), mode="eval"))


def format_number(value) -> str:
    """12.0 -> '12', 3.14159265 -> '3.141593'."""
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise ValueError("not a finite number")
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}" if abs(value) >= 1e15 else f"{round(value, 6):g}"
    return str(value)


_WORD_OPERATORS = [
    (r"\bmultiplied by\b", " * "), (r"\bdivided by\b", " / "),
    (r"\bto the power of\b", " ** "), (r"\bpower\b", " ** "),
    (r"\btimes\b", " * "), (r"\binto\b", " * "), (r"\bplus\b", " + "),
    (r"\bminus\b", " - "), (r"\bover\b", " / "), (r"\bmod\b", " % "),
]


def try_math(raw: str):
    """Return a calculator answer string, or None if `raw` is not maths."""
    text = raw.lower().replace("\u00d7", "*").replace("\u00f7", "/")

    # "20% of 50"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"{format_number(a)}% of {format_number(b)} = {format_number(a * b / 100)}"

    # "square root of 144" / "sqrt 144"
    m = re.search(r"(?:square root|sqrt)(?:\s+of)?\s*(\d+(?:\.\d+)?)", text)
    if m:
        n = float(m.group(1))
        return f"The square root of {format_number(n)} is {format_number(math.sqrt(n))}"

    for pattern, symbol in _WORD_OPERATORS:
        text = re.sub(pattern, symbol, text)
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", text)   # 3 x 4
    text = text.replace("^", "**")

    candidates = re.findall(r"[\d.()+\-*/%\s]{3,}", text)
    for candidate in sorted(candidates, key=len, reverse=True):
        expr = candidate.strip()
        # Needs "number operator number" so dates/phone numbers rarely match.
        if not re.search(r"[\d)]\s*(?:\*\*|[+\-*/%])\s*[\d(.\-]", expr):
            continue
        if re.fullmatch(r"\d{3,}(?:-\d+)+", expr):      # phone numbers, dates
            continue
        try:
            result = safe_eval(expr)
            if isinstance(result, int) and result.bit_length() > 300:
                return "That number is too big for me to calculate!"
            shown = re.sub(r"\s+", " ", expr).replace("**", "^")
            return f"{shown} = {format_number(result)}"
        except ZeroDivisionError:
            return "I can't divide by zero - not even a chatbot can do that!"
        except OverflowError:
            return "That number is too big for me to calculate!"
        except ValueError as error:
            if "too large" in str(error):
                return "That number is too big for me to calculate!"
            continue
        except (SyntaxError, RecursionError):
            continue
    return None


# --------------------------------------------------------------------------
# Skill handlers  (each returns a reply string)
# --------------------------------------------------------------------------
def _h_time(bot, raw, cleaned):
    return f"It's {_dt.datetime.now().strftime('%I:%M %p').lstrip('0')} right now."


def _h_date(bot, raw, cleaned):
    now = _dt.datetime.now()
    return f"Today is {now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}."


def _h_day(bot, raw, cleaned):
    return f"Today is {_dt.datetime.now().strftime('%A')}."


def _h_joke(bot, raw, cleaned):
    return bot.draw("joke", JOKES)


def _h_fact(bot, raw, cleaned):
    return "Fun fact: " + bot.draw("fact", FACTS)


def _h_quote(bot, raw, cleaned):
    return bot.draw("quote", QUOTES)


def _h_coin(bot, raw, cleaned):
    return f"I flipped a coin: it's {bot.rng.choice(['Heads', 'Tails'])}!"


def _h_dice(bot, raw, cleaned):
    sides = 6
    m = re.search(r"\b(\d{1,4})\s*sided\b|\bd(\d{1,4})\b", raw.lower())
    if m:
        sides = max(2, int(m.group(1) or m.group(2)))
    return f"I rolled a {sides}-sided die: {bot.rng.randint(1, sides)}"


def _h_random_number(bot, raw, cleaned):
    low, high = 1, 100
    m = re.search(r"(-?\d+)\s*(?:and|to|-)\s*(-?\d+)", raw.lower())
    if m:
        low, high = sorted((int(m.group(1)), int(m.group(2))))
    return f"Your random number between {low} and {high} is {bot.rng.randint(low, high)}."


def _h_greeting(bot, raw, cleaned):
    name = f" {bot.user_name}" if bot.user_name else ""
    return bot.rng.choice([
        f"Hi{name}! How can I help you?",
        f"Hello{name}! What can I do for you today?",
        f"Hey{name}! Good to see you.",
    ])


def _h_ask_user_name(bot, raw, cleaned):
    if bot.user_name:
        return f"Your name is {bot.user_name}. (I never forget a name!)"
    return "I don't know your name yet. Tell me by saying 'my name is ...'."


def _h_name_bot(bot, raw, cleaned):
    return f"I'm {BOT_NAME}, your offline virtual assistant."


def _h_help(bot, raw, cleaned):
    return (
        "Here's what I can do:\n"
        "- Chat: say hello, ask how I am, tell me your name\n"
        "- Time & date: 'what time is it?', 'what's the date?'\n"
        "- Maths: '12 * (3 + 4)', '20% of 150', 'square root of 81'\n"
        "- Fun: 'tell me a joke', 'fun fact', 'motivate me'\n"
        "- Games: 'flip a coin', 'roll a dice', 'random number 1 to 50'\n"
        "- Say 'repeat' to hear my last answer, or 'exit' to quit."
    )


# --------------------------------------------------------------------------
# Rules
#   (rule_name, [trigger phrases], reply)
#   `reply` is a list of replies (one is picked at random) or a handler
#   function. The longest matching phrase wins; on a tie the rule listed
#   first wins.
# --------------------------------------------------------------------------
RULES = [
    ("farewell",
     ["bye", "goodbye", "good bye", "see you", "see ya", "farewell",
      "talk to you later", "ttyl", "good night"],
     FAREWELL_REPLIES),
    ("how_are_you",
     ["how are you", "how are u", "how r you", "how r u", "hows it going",
      "how do you do", "whats up", "how you doing", "how have you been"],
     ["I'm doing great, thanks for asking! How are you?",
      "I'm fine, thanks! How are you?",
      "All systems running smoothly! How about you?"]),
    ("name_bot",
     ["what is your name", "whats your name", "your name", "who are you",
      "what are you called", "introduce yourself"],
     _h_name_bot),
    ("ask_user_name",
     ["what is my name", "whats my name", "do you know my name",
      "do you remember my name", "who am i"],
     _h_ask_user_name),
    ("abilities",
     ["what can you do", "what do you do", "help", "your abilities",
      "what are your features", "commands", "menu", "options"],
     _h_help),
    ("time", ["what time is it", "current time", "time now", "whats the time",
              "tell me the time", "time", "clock"], _h_time),
    ("date", ["what is the date", "whats the date", "todays date",
              "current date", "date today", "date"], _h_date),
    ("day", ["what day is it", "what day is today", "which day", "day today"],
     _h_day),
    ("joke", ["joke", "jokes", "make me laugh", "funny", "tell me something funny"],
     _h_joke),
    ("fact", ["fact", "facts", "fun fact", "tell me something interesting",
              "did you know", "trivia"], _h_fact),
    ("quote", ["motivate me", "motivation", "inspire me", "inspiration",
               "quote", "cheer me up", "encourage me"], _h_quote),
    ("coin", ["flip a coin", "toss a coin", "coin flip", "coin toss",
              "heads or tails"], _h_coin),
    ("dice", ["roll a dice", "roll a die", "roll dice", "roll the dice",
              "throw a dice", "dice", "die"], _h_dice),
    ("random_number", ["random number", "pick a number", "give me a number"],
     _h_random_number),
    ("thanks",
     ["thanks", "thank you", "thx", "thank u", "much appreciated", "cheers"],
     ["You're welcome! Happy to help.",
      "No problem at all!",
      "Anytime! Anything else I can do?"]),
    ("user_feeling_bad",
     ["i am sad", "im sad", "i feel sad", "feeling sad", "i am tired", "im tired",
      "feeling tired", "i feel bad", "feeling bad", "im feeling down",
      "i feel down", "i am upset", "im upset", "im stressed", "i am stressed",
      "feeling low", "i feel lonely", "im lonely", "im angry", "i am angry",
      "not good", "not well", "not great", "not okay", "im not ok",
      "im not okay", "bad day", "rough day"],
     ["I'm sorry to hear that. Want to talk about it, or shall I tell you a joke or share something uplifting?",
      "That sounds tough. I'm here to chat - or I can try to cheer you up with a joke!",
      "Sorry you're feeling that way. Say 'motivate me' or 'tell me a joke' if you'd like a lift."]),
    ("bored",
     ["i am bored", "im bored", "so bored", "bored", "nothing to do"],
     ["Let's fix that! Try 'tell me a joke', 'fun fact', or 'flip a coin'.",
      "Boredom, meet your match: say 'fun fact' or 'roll a dice' and let's play!"]),
    ("user_feeling_good",
     ["i am fine", "im fine", "i am good", "im good", "i am great", "im great",
      "doing well", "not bad", "i feel good", "i feel great", "im happy",
      "i am happy", "feeling good", "feeling great", "im doing good",
      "im doing great", "im doing well", "im okay", "im ok", "i am okay"],
     ["Glad to hear that! What would you like to talk about?",
      "That's wonderful! How can I help today?",
      "Great to hear! Want a joke or a fun fact?"]),
    ("compliment",
     ["you are smart", "youre smart", "you are great", "youre great",
      "you are awesome", "youre awesome", "good bot", "nice bot", "well done",
      "good job", "i like you", "you are funny", "youre funny", "you are cool",
      "you are amazing", "youre amazing", "love you", "i love you"],
     ["Aw, thank you! You're pretty great yourself.",
      "That made my circuits warm! Thanks!",
      "Thanks! I'm just a bunch of rules, but I try my best."]),
    ("insult",
     ["you are stupid", "youre stupid", "you are dumb", "youre dumb", "useless",
      "you suck", "i hate you", "bad bot", "you are bad", "shut up"],
     ["Sorry I let you down. Tell me what you need and I'll try again.",
      "Ouch! I'm only a simple bot, but I'm trying. How can I do better?"]),
    ("creator",
     ["who made you", "who created you", "who built you", "who is your creator",
      "who programmed you", "who developed you"],
     ["I was written in Python with simple rules - no AI models, no internet needed!"]),
    ("robot",
     ["are you a robot", "are you human", "are you real", "are you a bot",
      "are you ai", "are you an ai", "are you a person"],
     ["I'm a rule-based chatbot - software, not a person. I run fully offline on your computer!"]),
    ("offline",
     ["do you need internet", "are you online", "do you use internet",
      "do you use api", "do you need an api key", "do you use ai",
      "are you connected", "do you use cloud"],
     ["Nope! I'm fully offline: no cloud, no API keys, no internet - everything runs on your computer."]),
    ("weather",
     ["weather", "temperature", "forecast", "is it raining"],
     ["I can't check the weather - I'm fully offline with no internet. A weather app or window will do better!"]),
    ("laugh",
     ["lol", "haha", "hahaha", "lmao", "rofl", "hehe"],
     ["Glad I made you smile!", "Haha! :)"]),
    ("acknowledge",
     ["ok", "okay", "cool", "nice", "great", "alright", "got it", "i see",
      "awesome", "wow", "k"],
     ["Anything else you'd like to try?", "Cool! Let me know what's next.",
      "Great! Just ask if you need anything."]),
    ("greeting",
     ["hello", "hi", "hey", "hiya", "howdy", "yo", "sup", "good morning",
      "good afternoon", "good evening", "greetings"],
     _h_greeting),
]

# Intents that can be repeated by saying "another one", "more", "again" ...
REPEATABLE = {"joke": _h_joke, "fact": _h_fact, "quote": _h_quote,
              "coin": _h_coin, "dice": _h_dice, "random_number": _h_random_number}

FOLLOW_UP_YES = {"another one", "another", "one more", "more", "again", "next",
                 "next one", "yes", "yeah", "yep", "sure", "yes please",
                 "please", "ok another", "okay another"}
FOLLOW_UP_NO = {"no", "nope", "no thanks", "no thank you", "nah", "stop"}

# Quick-reply suggestions the GUI can show after each answer.
DEFAULT_SUGGESTIONS = ["Hello", "Tell me a joke", "What can you do?"]
SUGGESTIONS = {
    "greeting": ["How are you?", "Tell me a joke", "What can you do?"],
    "how_are_you": ["I'm good", "Tell me a joke", "What time is it?"],
    "joke": ["Another one", "Fun fact", "Flip a coin"],
    "fact": ["Another one", "Tell me a joke", "Motivate me"],
    "quote": ["Another one", "Tell me a joke", "Fun fact"],
    "coin": ["Flip again", "Roll a dice", "Tell me a joke"],
    "dice": ["Roll again", "Flip a coin", "Random number"],
    "random_number": ["Another one", "Roll a dice", "Flip a coin"],
    "time": ["What's the date?", "Tell me a joke", "Help"],
    "date": ["What time is it?", "Tell me a joke", "Help"],
    "abilities": ["12 * (3 + 4)", "Fun fact", "Flip a coin"],
    "math": ["20% of 150", "Square root of 81", "Help"],
    "user_feeling_bad": ["Motivate me", "Tell me a joke", "Fun fact"],
    "bored": ["Tell me a joke", "Fun fact", "Roll a dice"],
    "user_feeling_good": ["Tell me a joke", "Fun fact", "What time is it?"],
    "thanks": ["Tell me a joke", "Help", "Bye"],
}

POSITIVE_WORDS = {"good", "fine", "great", "okay", "ok", "well", "awesome", "happy",
                  "amazing", "excellent", "wonderful", "alright", "better"}
NEGATIVE_WORDS = {"bad", "sad", "tired", "terrible", "awful", "horrible", "sick",
                  "stressed", "angry", "upset", "lonely", "anxious", "worried",
                  "down", "meh"}

CRISIS_PHRASES = ["kill myself", "suicide", "want to die", "end my life",
                  "hurt myself", "self harm", "dont want to live"]
CRISIS_REPLY = (
    "I'm really sorry you're going through this. I'm only a simple chatbot and can't "
    "give you the support you deserve, but you matter. Please reach out to someone "
    "you trust, or contact your local emergency number or a crisis helpline right "
    "away. You don't have to face this alone."
)

NAME_STOPWORDS = {"not", "a", "an", "the", "fine", "good", "great", "ok", "okay",
                  "at", "back", "later", "maybe", "when", "if", "what", "who",
                  "bad", "sad", "tired", "so", "your", "my", "you", "me", "it"}

# Vocabulary of every word the bot knows - used to repair typos.
VOCAB = {
    word
    for _name, phrases, _reply in RULES
    for phrase in phrases
    for word in normalize(phrase).split()
}


def autocorrect(cleaned: str) -> str:
    """Repair typos and stretched words: 'heyyy' -> 'hey', 'thnks' -> 'thanks'."""
    fixed = []
    for token in cleaned.split():
        if token.isalpha():
            token = re.sub(r"(.)\1{2,}", r"\1", token)     # hellooo -> hello
            if token not in VOCAB:                         # hii -> hi, byee -> bye
                squeezed = re.sub(r"(.)\1+", r"\1", token)
                if squeezed in VOCAB:
                    token = squeezed
            if token not in VOCAB and len(token) >= 4:
                guess = difflib.get_close_matches(token, VOCAB, n=1, cutoff=0.84)
                if guess and guess[0][0] == token[0]:      # same first letter
                    token = guess[0]
        fixed.append(token)
    return " ".join(fixed)


# --------------------------------------------------------------------------
# The chatbot
# --------------------------------------------------------------------------
class Chatbot:
    """A small offline chatbot with memory and context.

    Parameters
    ----------
    memory_path : optional path of a JSON file. When given, the bot remembers
                  the user's name between sessions (stored only on this PC).
    seed        : optional number to make random choices repeatable (tests).
    """

    def __init__(self, memory_path: str = None, seed: int = None) -> None:
        self.memory_path = memory_path
        self.rng = random.Random(seed)
        self.user_name = ""
        self.last_intent = None
        self.last_reply = ""
        self._decks = {}
        self._load_memory()

    # ------------------------------ memory -------------------------------
    def _load_memory(self) -> None:
        if not self.memory_path or not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.user_name = str(json.load(f).get("name", ""))[:30]
        except (OSError, ValueError):
            self.user_name = ""

    def _save_memory(self) -> None:
        if not self.memory_path:
            return
        try:
            if self.user_name:
                with open(self.memory_path, "w", encoding="utf-8") as f:
                    json.dump({"name": self.user_name}, f)
            elif os.path.exists(self.memory_path):
                os.remove(self.memory_path)
        except OSError:
            pass   # memory is a bonus; never crash because of it

    # ------------------------------ helpers ------------------------------
    def draw(self, key: str, items: list) -> str:
        """Pick an item without repeating until all have been used."""
        deck = self._decks.get(key)
        if not deck:
            deck = list(items)
            self.rng.shuffle(deck)
            self._decks[key] = deck
        return deck.pop()

    @staticmethod
    def is_exit_command(message: str) -> bool:
        """Return True when the user wants to leave the chat."""
        return normalize(message) in EXIT_COMMANDS

    def welcome(self) -> str:
        return welcome_text(self.user_name)

    def reset_conversation(self) -> None:
        """Forget the conversation context (but not the user's name)."""
        self.last_intent = None
        self.last_reply = ""

    def suggestions(self) -> list:
        """Quick-reply ideas that fit the last topic."""
        return SUGGESTIONS.get(self.last_intent, DEFAULT_SUGGESTIONS)

    def _finish(self, intent: str, reply: str) -> str:
        self.last_intent = intent
        self.last_reply = reply
        return reply

    # ------------------------------ main API -----------------------------
    def reply(self, message: str) -> str:
        """Return the chatbot's reply for a user message."""
        raw = message.strip()
        cleaned = normalize(raw)

        # 1. Empty input.
        if not cleaned:
            return self._finish(self.last_intent,
                                "Please type something and I'll do my best to reply.")

        # 2. Exit.
        if cleaned in EXIT_COMMANDS:
            return self._finish("farewell", self.rng.choice(FAREWELL_REPLIES))

        # 3. Safety first.
        if contains_phrase(cleaned, CRISIS_PHRASES):
            return self._finish("crisis", CRISIS_REPLY)

        # 4. Forget me.
        if contains_phrase(cleaned, ["forget me", "forget my name", "reset memory",
                                     "clear memory", "erase my name"]):
            self.user_name = ""
            self._save_memory()
            return self._finish("forget", "Done - I've forgotten your name. We can start fresh!")

        # 5. "my name is ..." / "call me ..."
        m = re.search(
            r"\b(?:my name is|my name\'?s|call me|i am called|name\'?s)\s+([A-Za-z][A-Za-z\-]{0,24})",
            raw, re.IGNORECASE)
        if m and m.group(1).lower() not in NAME_STOPWORDS:
            self.user_name = m.group(1).capitalize()
            self._save_memory()
            return self._finish("set_name", self.rng.choice([
                f"Nice to meet you, {self.user_name}!",
                f"Great to meet you, {self.user_name}! I'll remember that.",
            ]))

        # 6. Calculator.
        maths = try_math(raw)
        if maths:
            return self._finish("math", maths)

        # 7. Repeat the last answer.
        if contains_phrase(cleaned, ["repeat", "say that again", "what did you say",
                                     "come again", "pardon"]) and self.last_reply:
            return self.last_reply

        # 8. Context: follow-ups like "another one" / "yes".
        if self.last_intent in REPEATABLE:
            if cleaned in FOLLOW_UP_YES or cleaned in ("flip again", "roll again"):
                return self._finish(self.last_intent,
                                    REPEATABLE[self.last_intent](self, raw, cleaned))
            if cleaned in FOLLOW_UP_NO:
                return self._finish("acknowledge",
                                    "Alright! Let me know if you need anything else.")

        # 9. Context: a short answer to "How are you?"
        if self.last_intent == "how_are_you":
            tokens = cleaned.split()
            if len(tokens) <= 4:
                if contains_phrase(cleaned, ["not good", "not well", "not great",
                                             "not okay", "not ok"]) \
                        or (set(tokens) & NEGATIVE_WORDS and "not bad" not in cleaned):
                    return self._finish("user_feeling_bad", self.rng.choice(RULES_BY_NAME["user_feeling_bad"][2]))
                if set(tokens) & POSITIVE_WORDS or "not bad" in cleaned:
                    return self._finish("user_feeling_good", self.rng.choice(RULES_BY_NAME["user_feeling_good"][2]))

        # 10. Scored rule matching (with typo repair).
        fixed = autocorrect(cleaned)
        matches = []   # (words_matched, -rule_order, name)
        for order, (name, phrases, _reply) in enumerate(RULES):
            score = best_phrase_length(fixed, phrases)
            if score:
                matches.append((score, -order, name))

        if not matches:
            return self._finish(None, self.rng.choice(DEFAULT_REPLIES))

        names = {name for _s, _o, name in matches}
        if "farewell" in names:
            chosen = "farewell"
        else:
            chosen = max(matches)[2]

        _name, _phrases, reply = RULES_BY_NAME[chosen]
        text = reply(self, raw, fixed) if callable(reply) else self.rng.choice(reply)

        # "hello, how are you?" -> greet AND answer.
        if "greeting" in names and chosen not in ("greeting", "farewell", "acknowledge"):
            who = f" {self.user_name}" if self.user_name else ""
            text = f"{self.rng.choice(['Hi', 'Hello', 'Hey'])}{who}! {text}"

        return self._finish(chosen, text)


RULES_BY_NAME = {rule[0]: rule for rule in RULES}


# --------------------------------------------------------------------------
# Backwards-compatible function API (same as v1)
# --------------------------------------------------------------------------
_DEFAULT_BOT = Chatbot()


def is_exit_command(message: str) -> bool:
    return Chatbot.is_exit_command(message)


def get_response(message: str) -> str:
    """Return the chatbot's reply for a user message (stateful, no file saved)."""
    return _DEFAULT_BOT.reply(message)


# --------------------------------------------------------------------------
# Optional: terminal version (uses input(), print() and a while loop)
# --------------------------------------------------------------------------
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "chatbot_memory.json")


def run_console_chat() -> None:
    """Run the chatbot in the terminal. Type 'exit' to quit."""
    bot = Chatbot(memory_path=MEMORY_FILE)
    print(f"{BOT_NAME}: {bot.welcome()}\n")

    while True:
        try:
            user_message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / Ctrl+D: leave politely instead of crashing.
            print(f"\n{BOT_NAME}: {FAREWELL_REPLY}")
            break

        if bot.is_exit_command(user_message):
            print(f"{BOT_NAME}: {FAREWELL_REPLY}")
            break

        print(f"{BOT_NAME}: {bot.reply(user_message)}")


if __name__ == "__main__":
    run_console_chat()
