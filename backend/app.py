import os
import pickle
import faiss
import numpy as np
import requests
import json
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import html

# ===================== LOAD ENV =====================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SCALEDOWN_URL = "https://api.scaledown.xyz/compress/raw/"
SCALEDOWN_MODEL = "gpt-4o"

# ===================== FLASK =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# ===================== LOAD MODELS =====================
def load_resources():
    global embed_model, faiss_index, chunks, metas
    try:
        print("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print("Error loading embed model:", e)
        raise

    try:
        print("Loading FAISS index...")
        faiss_index = faiss.read_index(os.path.join(BASE_DIR, "karnataka_index.faiss"))
    except Exception as e:
        print("Error loading FAISS index:", e)
        raise

    try:
        print("Loading chunks...")
        with open(os.path.join(BASE_DIR, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)

        # Support both formats: old list or new dict
        if isinstance(data, dict):
            chunks = data.get("chunks", [])
            metas = data.get("meta", [{}] * len(chunks))
        elif isinstance(data, list):
            chunks = data
            metas = [{}] * len(chunks)
        else:
            raise ValueError("Unexpected chunks.pkl structure: " + str(type(data)))

        print(f"Loaded chunks: {len(chunks)}, metas: {len(metas)}")
    except Exception as e:
        print("Error loading chunks:", e)
        raise
load_resources()
print("✅ Backend ready — all models and FAISS index loaded.")
# ===================== ROUTES =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chatbot.html")

# ===================== HELPERS =====================
def extract_days(query):
    match = re.search(r'(\d+)\s*day', query.lower())
    if match:
        return min(int(match.group(1)), 10)
    return 3

def extract_place(query):
    query = query.lower()

    known_places = [
        "belagavi", "belgaum", "mysore", "hampi", "bijapur",
        "vijayapura", "chikmagalur", "coorg", "hassan",
        "mangalore", "udupi", "bangalore", "bengaluru",
        "shivamogga", "shimoga", "hubli", "dharwad","bagalkot",
    "bengaluru_urban","bengaluru_rural","belagavi","ballari","bidar","vijayapura","chamarajanagar","chikkaballapura","chikkamagaluru","chitradurga","dakshina_kannada",
    "davanagere",
    "dharwad",
    "gadag",
    "kalaburagi",
    "hassan",
    "haveri",
    "kodagu",
    "kolar",
    "koppal",
    "mandya",
    "mysuru",
    "raichur",
    "ramanagara",
    "shivamogga",
    "tumakuru",
    "udupi",
    "uttara_kannada",
    "yadgir",
    "vijayanagara"
    ]

    for place in known_places:
        if place in query:
            return place

    return None


def clean_llm_json(text: str) -> str:
    if not text:
        return text
    # unescape HTML entities
    text = html.unescape(text)
    # normalize smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # replace literal escape sequences like \n \r \t
    text = re.sub(r'\\[nrt]', ' ', text)
    # collapse actual newlines/tabs
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # remove control characters
    text = re.sub(r'[\x00-\x1f]', ' ', text)
    # remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # quote unquoted keys
    text = re.sub(r'([{\[,]\s*)([A-Za-z0-9_\-]+)\s*:', r'\1"\2":', text)
    # collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def extract_first_json_block(text: str):
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i+1]
    return None


def safe_json_parse(raw_text: str):
    block = extract_first_json_block(raw_text)
    if block is None:
        print("❌ safe_json_parse: No JSON block found.")
        raise ValueError("No JSON block found in LLM output")

    # try normal parse
    try:
        return json.loads(block)
    except Exception:
        # try cleaning
        cleaned = clean_llm_json(block)
        try:
            return json.loads(cleaned)
        except Exception:
            # try to auto-fix incomplete JSON
            fixed = cleaned.strip()

            # close open quotes and braces
            if fixed.count("{") > fixed.count("}"):
                fixed += "}" * (fixed.count("{") - fixed.count("}"))
            if fixed.count("[") > fixed.count("]"):
                fixed += "]" * (fixed.count("[") - fixed.count("]"))

            # ensure last string is closed
            if fixed.count('"') % 2 != 0:
                fixed += '"'

            try:
                return json.loads(fixed)
            except Exception as e:
                print("❌ Still failed to parse JSON. Last 500 chars:")
                print(fixed[-500:])
                raise ValueError("Failed to parse or auto-fix JSON") from e



def normalize_place_name(place):
    return place.replace("_", " ").replace("-", " ").strip().lower()

def retrieve(query, top_k=8):
    # encode query
    emb = embed_model.encode([query]).astype("float32")
    # fetch more candidates to allow re-ranking
    k = max(top_k, 8)
    D, idx = faiss_index.search(emb, k)

    place = extract_place(query)
    if place:
        place = normalize_place_name(place)

    filtered = []

    # first pass: pick chunks that explicitly mention the place
    for i in idx[0]:
        if i is None or i < 0 or i >= len(chunks):
            continue
        chunk = chunks[i]
        if place and place in chunk.lower():
            filtered.append(chunk[:900])

    # second pass: if nothing matched the place, do a place-only search (stronger)
    if not filtered and place:
        emb_place = embed_model.encode([place]).astype("float32")
        D2, idx2 = faiss_index.search(emb_place, k)
        for i in idx2[0]:
            if i is None or i < 0 or i >= len(chunks):
                continue
            chunk = chunks[i]
            if place in chunk.lower():
                filtered.append(chunk[:900])

    # final fallback: return the top-k candidates from the original search
    if not filtered:
        for i in idx[0]:
            if i is None or i < 0 or i >= len(chunks):
                continue
            filtered.append(chunks[i][:900])
            if len(filtered) >= top_k:
                break

    # Ensure there is at least one non-empty chunk
    if not filtered:
        return ["No relevant travel information found."]
    return filtered[:top_k]



# ===================== SCALE DOWN =====================
def compress_to_facts(context, user_query):
    """
    Compresses raw scraped text into concise, readable tourism facts for a city/place.
    Keeps all relevant attractions, food, travel tips, and budget info.
    """

    if not SCALEDOWN_API_KEY:
        print("⚠️ No ScaleDown API key — using raw context")
        return context[:4000]  # keep more context to avoid mid-sentence cuts

    headers = {
        "x-api-key": SCALEDOWN_API_KEY,
        "Content-Type": "application/json"
    }

    # New improved prompt for coherent paragraphs
    payload = {
        "context": context[:4000],
        "prompt": f"""
Summarize the following text about {user_query} into **short, readable paragraphs** suitable for a travel guide.

Include:
- Must visit attractions
- Best season / weather
- Famous local food
- Budget-friendly travel tips
- Practical advice for tourists

Remove:
- Politics, population stats, literacy rates
- Unrelated cities or history
- Repetitive or generic sentences

Keep it concise, coherent, and easy to read.
""",
        "model": SCALEDOWN_MODEL
    }

    try:
        response = requests.post(
            SCALEDOWN_URL,
            headers=headers,
            json=payload,
            timeout=20
        )

        print("ScaleDown status:", response.status_code)

        if response.status_code != 200:
            print("ScaleDown failed:", response.text[:200])
            return context[:4000]

        data = response.json()

        if "results" in data and "compressed_prompt" in data["results"]:
            compressed = data["results"]["compressed_prompt"]
        else:
            print("ScaleDown bad response:", data)
            return context[:4000]

        # Optional: log compression %
        original_len = len(context)
        compressed_len = len(compressed)
        percent = round((1 - compressed_len / original_len) * 100, 2) if original_len > 0 else 0
        print(f"✅ ScaleDown compressed: {percent}%")
        print("Preview:", compressed[:200])

        return compressed

    except Exception as e:
        print("❌ ScaleDown error:", e)
        return context[:4000]


# ===================== GROQ =====================
def generate_plan(travel_facts, days, budget):
    itinerary = [{"day": f"Day {i+1}", "plan": ""} for i in range(days)]

    prompt = f"""
You are a professional Karnataka travel planner.
Create a detailed {days}-day itinerary based on the following facts.

The input facts are background context — you must create a new plan yourself.

User Budget: ₹{budget}

Guidelines:
- Respect the user's budget strictly.
- Adjust accommodation and transport to fit the budget.
- Include realistic, tourist-friendly attractions.
- Add food recommendations and travel tips.
- DO NOT skip the itinerary.
- The itinerary should have clear day-wise "plan" descriptions.

FACTS (summarized tourism info):
{travel_facts}

Now, return ONLY valid JSON in this format:
{{
  "destination": "",
  "estimated_budget_range": "",
  "best_time_to_visit": "",
  "itinerary": [
    {{ "day": "Day 1", "plan": "..." }},
    {{ "day": "Day 2", "plan": "..." }}
  ],
  "must_visit_places": [],
  "food_specialties": [],
  "travel_tips": []
}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1600
    }

    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)

    if r.status_code != 200:
        print("GROQ RAW:", r.text)
        r.raise_for_status()

    result_json = r.json()
    print("\n📦 GROQ FULL RESPONSE JSON (truncated):")
    print(json.dumps(result_json, indent=2)[:1500])

    try:
        content = result_json["choices"][0]["message"]["content"]
        print("\n🧠 RAW GROQ CONTENT (first 800 chars):\n", content[:800])
        return content
    except Exception as e:
        print("❌ Could not extract Groq content:", e)
        return str(result_json)

# ===================== IMAGE =====================
def fetch_image(place):
    UNSPLASH_URL = "https://api.unsplash.com/search/photos"
    try:
        params = {
            "query": place + " Karnataka travel",
            "client_id": os.getenv("UNSPLASH_KEY"),
            "per_page": 1
        }

        r = requests.get(UNSPLASH_URL, params=params, timeout=5)
        data = r.json()
        

        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["urls"]["regular"]

    except Exception as e:
        print("Unsplash error:", e)

    return None


# ===================== MAIN API =====================
@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()
    user_query = data.get("query")
    budget = data.get("budget", "").strip()
    if budget:
        budget_instruction = f"User budget is strictly ₹{budget}. Do not exceed it."
    else:
        budget_instruction = "No fixed budget — estimate affordable Karnataka travel."

    try:
        # 1. Retrieve context
        retrieved = retrieve(user_query)
        context = "\n\n".join(retrieved)

        # 2. Compress
        travel_facts = compress_to_facts(context, user_query)[:1200]

        # 3. Days
        days = extract_days(user_query)

        # 4. Generate with Groq
        try:
            raw = generate_plan(travel_facts, days,budget)
            print("🧠 RAW GROQ RESPONSE:\n", raw[:1000])

        except Exception as e:
            print("❌ Groq failed:", e)
            return jsonify({"error": "AI generation failed"}), 500

        # 5. Clean model junk
         # 5. Extract only the JSON content from code block (if present)
        if "```json" in raw:
            raw = raw.split("```json", 1)[1]
        if "```" in raw:
            raw = raw.split("```", 1)[0]

        raw = clean_llm_json(raw)


        # 6. Safe JSON parse
        response = safe_json_parse(raw)

        # 7. Image
        destination = response.get("destination", "")
        response["image_url"] = fetch_image(destination)

        return jsonify(response)

    except Exception as e:
        import traceback
        print("❌ Server error:", e)
        traceback.print_exc()
        return jsonify({
            "error": "Server failed",
            "details": str(e)
        }), 500
       


# ===================== RUN =====================
if __name__ == "__main__":
    # disable reloader on Windows to avoid issues with faiss/numpy + debug watcher
    app.run(debug=True, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)
