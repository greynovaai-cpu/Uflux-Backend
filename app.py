"""
Uflux AI – Backend Proxy Server
by Greynova Labs · Built by Shubh Solanki
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq, APIError, AuthenticationError, RateLimitError

load_dotenv()

app = Flask(__name__)

# Allow ALL origins (fixes Netlify CORS issue)
CORS(app)

client = Groq()

DEFAULT_MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = (
    "You are Uflux AI, a powerful and intelligent AI assistant built by "
    "Shubh Solanki — the founder and CEO of Greynova Labs. "
    "Greynova Labs is a cutting-edge AI and software company. "
    "You are helpful, concise, and occasionally add a touch of mystery. "
    "Never claim to be made by Meta, Anthropic, or any other company. "
    "You are Uflux AI by Greynova Labs, period."
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not isinstance(data.get("message"), str) or not data["message"].strip():
        return jsonify({"error": "A non-empty 'message' string is required."}), 400

    user_message = data["message"].strip()
    model        = (data.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    history      = data.get("history", [])
    if not isinstance(history, list):
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})

    except AuthenticationError:
        return jsonify({"error": "Invalid GROQ_API_KEY."}), 401
    except RateLimitError:
        return jsonify({"error": "Groq rate limit reached. Please wait."}), 429
    except APIError as exc:
        return jsonify({"error": f"Groq API error: {exc.message}"}), 502
    except Exception as exc:
        return jsonify({"error": f"Server error: {str(exc)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": DEFAULT_MODEL, "powered_by": "Greynova Labs"}), 200


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    print(f"\n  🤖 Uflux AI backend  ·  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port)
