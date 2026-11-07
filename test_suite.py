import os
import sys
import tkinter as tk
from tkinter import messagebox
messagebox.showinfo = lambda title, msg: True
messagebox.showwarning = lambda title, msg: True

# Add chatbot directory to sys.path
sys.path.insert(0, os.path.abspath("chatbot"))

import main

print("==========================================")
print("TESTING 1: LocalChatbot Brain & Intent Logic")
print("==========================================")

bot = main.LocalChatbot()

# Test 1: Greetings
resp_hello = bot.get_response("hello")
print("Q: hello ->", resp_hello)
assert any(g in resp_hello.lower() for g in ["hello", "help", "ready", "assist"]), "Hello response failed"

# Test 2: Status
resp_status = bot.get_response("how are you")
print("Q: how are you ->", resp_status)
assert "operating" in resp_status.lower() or "efficiency" in resp_status.lower(), "Status response failed"

# Test 3: Capabilities
resp_cap = bot.get_response("what can you do")
print("Q: what can you do ->", resp_cap)
assert "nlp" in resp_cap.lower() or "vector" in resp_cap.lower() or "offline" in resp_cap.lower(), "Capabilities response failed"

# Test 4: Dynamic Time
resp_time = bot.get_response("what time is it")
print("Q: what time is it ->", resp_time)
assert "time" in resp_time.lower(), "Time response failed"

# Test 5: Math Calculation
resp_math1 = bot.get_response("Calculate (150 * 4) + (250 / 5)")
print("Q: Calculate (150 * 4) + (250 / 5) ->", resp_math1)
assert "650" in resp_math1, "Math calculation failed"

# Test 6: Offline Math
resp_math2 = bot.get_response("20% of 150")
print("Q: 20% of 150 ->", resp_math2)
assert "30" in resp_math2, "Percentage math failed"

# Test 7: Memory Retention
resp_mem_set = bot.get_response("My name is Commander")
print("Q: My name is Commander ->", resp_mem_set)
assert "Commander" in resp_mem_set, "Set name failed"
assert bot.memory.get("user_name") == "Commander", "Memory user_name failed"

resp_mem_check = bot.get_response("What is my name?")
print("Q: What is my name? ->", resp_mem_check)
assert "Commander" in resp_mem_check, "Check name failed"

# Test 8: Offline Brain Fallback (Joke / Fact)
resp_joke = bot.get_response("tell me a joke")
print("Q: tell me a joke ->", resp_joke)
assert len(resp_joke) > 10, "Joke response failed"

resp_fact = bot.get_response("tell me a fun fact")
print("Q: tell me a fun fact ->", resp_fact)
assert len(resp_fact) > 10, "Fact response failed"

# Test 9: Intent Learning (Intent Studio)
bot.learn("flight_status", "check flight 402", "Flight 402 is on schedule.")
resp_custom = bot.get_response("check flight 402")
print("Q: check flight 402 ->", resp_custom)
assert "on schedule" in resp_custom.lower(), "Custom learned intent failed"

# Test 10: Goodbye
resp_bye = bot.get_response("bye")
print("Q: bye ->", resp_bye)
assert any(w in resp_bye.lower() for w in ["ahead", "anytime", "day", "great", "later"]), "Goodbye response failed"

print("\n--- All LocalChatbot Brain Tests PASSED! ---\n")

print("==========================================")
print("TESTING 2: Tkinter Pure White & Black UI")
print("==========================================")

root = tk.Tk()
root.withdraw()

app = main.SmartChatbotUI(root)
print("Window Title:", root.title())
assert "SMART CHATBOT" in root.title()

# Verify exact color theme requirements
required_palette = {
    "app_bg": "#FFFFFF",
    "chat_bg": "#FFFFFF",
    "sidebar_bg": "#F7F7F7",
    "card_bg": "#FFFFFF",
    "btn_primary_bg": "#000000",
    "btn_primary_fg": "#FFFFFF",
    "btn_secondary_bg": "#F2F2F2",
    "btn_secondary_fg": "#000000",
    "text_main": "#000000",
    "text_secondary": "#555555",
    "border": "#E5E5E5",
    "user_bubble_bg": "#F2F2F2",
    "user_bubble_fg": "#000000",
    "bot_bubble_bg": "#FFFFFF",
    "bot_bubble_fg": "#000000",
    "input_bg": "#FFFFFF",
    "input_fg": "#000000",
    "placeholder": "#888888",
    "hover_bg": "#EAEAEA",
    "focus_border": "#000000",
}

for k, expected_color in required_palette.items():
    actual_color = app.colors[k]
    assert actual_color == expected_color, f"Color mismatch for {k}: expected {expected_color}, got {actual_color}"
print("All 20 exact palette colors VERIFIED!")

# Verify no unwanted accent colors exist in app.colors
disallowed_colors = ["#06b6d4", "#8b5cf6", "#10b981", "#0b0f19", "#121826", "#0f2e4a"]
for k, v in app.colors.items():
    assert str(v).lower() not in disallowed_colors, f"Disallowed color {v} found in {k}"
print("Zero cyan/purple/green/dark accent colors verified in palette!")

# Test Sending Message
app.user_entry.delete(0, "end")
app.user_entry.insert(0, "hello")
app._handle_send()
root.update_idletasks()

# Test Quick Prompt
app._send_quick_prompt("What time is it right now?")
root.update_idletasks()

# Test Backward Compatibility with ModernLandscapeUI alias
assert main.ModernLandscapeUI is main.SmartChatbotUI
print("ModernLandscapeUI alias confirmed!")

# Test Intent Studio Training through UI
app.pattern_entry.insert(0, "open cargo bay")
app.response_entry.insert(0, "Cargo bay doors are now open.")
app._train_bot()
root.update_idletasks()

# Verify inspectors updated
assert "open cargo bay" in [p for intent in app.bot.intents for p in intent.get("patterns", [])]
print("Intent Studio training through UI verified!")

# Test Clear Chat
app._clear_chat()
root.update_idletasks()
assert len(app._bubble_labels) == 0
print("Clear chat verified!")

# Test Save Knowledge
app._save_knowledge()
assert os.path.exists("knowledge.json")
print("Save knowledge verified! Clean up test knowledge file...")
if os.path.exists("knowledge.json"):
    os.remove("knowledge.json")

root.destroy()
print("\n==========================================")
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("==========================================")
