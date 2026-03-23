import json
import re
import os
import datetime
from collections import Counter

from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

DEFAULT_MODEL = "gemma3:4b"
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def save_log(filename: str, text: str, stats: dict, analysis: dict):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(filename)[0]

    raw_path = os.path.join(LOGS_DIR, f"{timestamp}_{base}_original.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)

    result_path = os.path.join(LOGS_DIR, f"{timestamp}_{base}_analysis.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "analysis": analysis}, f, ensure_ascii=False, indent=2)

    print(f"[LOG] Saved: logs/{timestamp}_{base}_*")


def parse_telegram_json(text: str) -> list[dict]:
    data = json.loads(text)
    messages = []
    for msg in data.get("messages", []):
        if msg.get("type") != "message":
            continue
        t = msg.get("text", "")
        if isinstance(t, list):
            t = "".join(p if isinstance(p, str) else p.get("text", "") for p in t)
        if not t.strip():
            continue
        messages.append({
            "author": msg.get("from") or "Unknown",
            "text": t.strip(),
            "date": msg.get("date", ""),
        })
    return messages


def parse_whatsapp(text: str) -> list[dict]:
    pattern = re.compile(
        r"[\[\(]?(\d{1,2}[./]\d{1,2}[./]\d{2,4}),?\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)"
        r"[\]\)]?\s*[-–]?\s*([^:]+):\s*(.*)"
    )
    messages = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            messages.append({
                "author": m.group(3).strip(),
                "text": m.group(4).strip(),
                "date": f"{m.group(1)} {m.group(2)}",
            })
    return messages


def parse_plain(text: str) -> list[dict]:
    messages = []
    for line in text.splitlines():
        if ":" in line:
            author, _, body = line.partition(":")
            if author.strip() and body.strip():
                messages.append({"author": author.strip(), "text": body.strip(), "date": ""})
    return messages


def load_chat(filename: str, text: str) -> list[dict]:
    if filename.endswith(".json"):
        return parse_telegram_json(text)
    msgs = parse_whatsapp(text)
    return msgs if msgs else parse_plain(text)


def calc_stats(messages: list[dict]) -> dict:
    counts: Counter = Counter()
    words: Counter = Counter()
    chars: Counter = Counter()

    for m in messages:
        a = m["author"] or "Unknown"
        counts[a] += 1
        words[a] += len(m["text"].split())
        chars[a] += len(m["text"])

    total = len(messages)
    authors = list(counts.keys())
    return {
        "total": total,
        "authors": authors,
        "counts": dict(counts),
        "words": dict(words),
        "chars": dict(chars),
        "avg_len": {a: round(chars[a] / counts[a]) if counts[a] else 0 for a in authors},
        "share": {a: round(counts[a] / total * 100, 1) if total else 0 for a in authors},
    }


def build_sample(messages: list[dict], max_msgs: int) -> str:
    total = len(messages)
    if total <= max_msgs:
        sample = messages
    else:
        step = total / max_msgs
        sample = [messages[int(i * step)] for i in range(max_msgs)]

    return "\n".join(
        f"{m['author']}: {m['text']}"
        for m in sample
    )


PROMPT = """Read the conversation below and write a short, natural characterization of it.
Respond ONLY with a valid JSON object, no markdown, no extra text.

Describe the chat as you would to someone who hasn't read it — what is it about,
what is the vibe, how do the participants communicate, what are the main themes.
Be honest and observational. No need to focus on conflicts specifically — just describe what you see.

Return this JSON structure:
{{
  "summary": "3-5 sentences describing the overall character of this conversation — topics, mood, communication style, dynamics",
  "topics": ["main topic 1", "main topic 2", "main topic 3"],
  "vibe": "one short phrase describing the general atmosphere (e.g. friendly and casual, tense and brief, warm but chaotic)",
  "participants": {{
    "NAME": "1-2 sentences about how this person communicates in the chat"
  }}
}}

CONVERSATION:
{chat_text}"""


def analyze(chat_text: str, model: str) -> dict:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(chat_text=chat_text)}],
        options={"temperature": 0.4, "num_predict": 1024, "num_ctx": 8192},
    )
    raw = response.message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON. Response: {raw[:200]}")
    return json.loads(m.group(0))


@app.after_request
def skip_ngrok_warning(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(html_path):
        return "index.html not found next to server.py", 404
    with open(html_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/health")
def health():
    try:
        models = [m.model for m in ollama.list().models]
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 503


@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "file" not in request.files:
        return jsonify({"error": "File not provided"}), 400

    file = request.files["file"]
    model = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    max_msgs = int(request.form.get("max_msgs", 300))

    try:
        text = file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            file.seek(0)
            text = file.read().decode("cp1251")
        except Exception:
            return jsonify({"error": "Failed to read file (try UTF-8)"}), 400

    try:
        messages = load_chat(file.filename, text)
    except Exception as e:
        return jsonify({"error": f"Parsing error: {e}"}), 400

    if not messages:
        return jsonify({"error": "Chat is empty or format not recognized"}), 400

    stats = calc_stats(messages)

    chat_text = build_sample(messages, max_msgs)
    try:
        analysis = analyze(chat_text, model)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}"}), 500
    except Exception as e:
        err = str(e)
        if "model" in err.lower() and "not found" in err.lower():
            return jsonify({"error": f"Model '{model}' not found. Run: ollama pull {model}"}), 400
        if "connection" in err.lower() or "refused" in err.lower():
            return jsonify({"error": "Ollama is not running. Run: ollama serve"}), 503
        return jsonify({"error": err}), 500

    save_log(file.filename, text, stats, analysis)

    return jsonify({"stats": stats, "analysis": analysis})


if __name__ == "__main__":
    import threading
    import webbrowser

    PORT = 5000
    URL = f"http://localhost:{PORT}"

    print("\n  Chat Analyzer Server")
    print(f"  {URL}")
    print(f"  Logs: {LOGS_DIR}")
    print("  Ctrl+C to stop\n")

    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(URL)

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)