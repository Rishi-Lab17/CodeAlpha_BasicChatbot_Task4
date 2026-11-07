# SMART CHATBOT — Pure White & Black Edition

A smart, **fully offline** desktop chatbot with a premium, minimal, professional **Pure White & Black** Tkinter interface.
Zero cloud dependencies, zero external packages, **no API keys** required — powered by Python standard library and local vector NLP.

## Pure White & Black Minimalist UI

- **Color Theme**: Pure White (#FFFFFF) & Deep Black (#000000) with Light Grey (#F7F7F7, #F2F2F2, #EAEAEA) surfaces.
- **Branding**: Clean SMART CHATBOT header with minimal black icon badge and status indicator.
- **Sidebar**: Light grey navigation sidebar with active indicator, Core Runtime toggles, One-Click Prompts, and knowledge controls.
- **Chat Workspace**: White canvas, user messages in light grey (#F2F2F2) rounded bubble cards, bot messages on pure white (#FFFFFF) with #E5E5E5 borders.
- **Typing Indicator**: Smooth, subtle animated loading dots ("Thinking ● ○ ○") before each reply.
- **Elevated Input Dock**: White input field with subtle black focus border, placeholder handling, and black primary SEND button.
- **Intent Studio**: Expand vector dictionary in real-time, inspect session entity memory, and explore active vector registry.

## What's new in v2.0

### Smarter brain (`chatbot.py`)
- **Scored matching** - the most specific phrase wins (not just the first rule).
- **Typo tolerance** - `helo`, `thnks`, `wats up`, `heyyyy` still work.
- **Memory** - say `my name is Sam` and it remembers you (even next time you open it).
  Say `forget my name` to erase it.
- **Context** - after a joke, `another one` / `more` gives you another joke; after
  `how are you?` a short `good` or `not good` gets the right reaction.
- **Greeting + question** in one message: `hi, how are you?`
- **Built-in skills**
  - Time / date / day
  - Calculator (safe, no `eval`): `12 * (3 + 4)`, `5 plus 3`, `2^10`, `20% of 150`, `square root of 81`
  - Jokes, fun facts, motivational quotes (no repeats until all have been shown)
  - Coin flip, dice (`roll a 20 sided dice`), random number (`random number 5 to 10`)
  - `repeat` / `say that again`
  - Small talk: compliments, "are you a robot?", "who made you?", "do you need internet?"
- A caring reply if someone mentions self-harm.
- Varied replies instead of one fixed sentence.

### Better GUI (`main.py`)
- **Dark / Light** theme (button or `Ctrl+Shift+T`)
- Animated **typing indicator** before each reply
- **Time stamps** on every message
- **Quick-reply chips** that change with the topic
- **Up / Down arrows** recall your earlier messages
- **Save** the conversation as a `.txt` file (`Ctrl+S`)
- `Ctrl+L` clears the chat; replies stay in order even if you type fast
- Picks a good font for Windows, macOS or Linux

## Project Structure

```
chatbot/
├── main.py            # Tkinter GUI (run this)
├── chatbot.py         # Chatbot logic (no GUI code)
├── README.md
└── requirements.txt
```
(`chatbot_memory.json` is created next to the code only after you tell the bot your name.)

## How to Run

1. Install [Python 3.8+](https://www.python.org/downloads/)
   (Tkinter is included on Windows/macOS; on Linux run `sudo apt install python3-tk`).
2. Open the folder in VS Code (**File > Open Folder**).
3. Open `main.py` and press **Run**, or in a terminal:

```bash
python main.py
```

### Terminal-only version

```bash
python chatbot.py
```

## Example Conversation

```
You: helo
Bot: Hi! How can I help you?
You: my name is Sam
Bot: Nice to meet you, Sam!
You: what is 15% of 240
Bot: 15% of 240 = 36
You: tell me a joke
Bot: Why do programmers prefer dark mode? Because light attracts bugs!
You: another one
Bot: What's a computer's favourite snack? Microchips!
You: thnks, bye
Bot: See you later! Take care!
```

## How It Works

1. `normalize()` lowercases the text and strips punctuation.
2. Special requests are handled first: exit, name memory, calculator, "repeat", follow-ups.
3. `autocorrect()` repairs typos using the words the bot already knows (`difflib`).
4. Every rule in `RULES` is scored by the length of its best matching phrase; the highest score wins.
5. Matching uses whole words, so `hi` will not trigger inside `this`.
6. If nothing matches, a friendly default reply is returned.

## Adding Your Own Rules

Open `chatbot.py` and add an entry to `RULES`. A reply can be a list (one is picked at random):

```python
("pizza", ["pizza", "i want pizza"], ["Pizza is great!", "Extra cheese, please!"]),
```

or a function `def my_handler(bot, raw, cleaned): ... return "text"`.
Longer, more specific phrases automatically beat shorter ones.

## Concepts Used

functions - classes - `while` loops - `input()` / `print()` - regular expressions -
`ast` (safe calculator) - `difflib` (typo repair) - JSON file storage - exception handling - Tkinter
