import os
import datetime
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local .env variables
load_dotenv()

# ---------------------------------------------------------------------------
# WEEKLY PROMPT SUITE (BULGARIAN TAUGHT IN PORTUGUESE)
# ---------------------------------------------------------------------------

PROMPTS = {
    0: """
    You are an expert Bulgarian language teacher. Create a Telegram post in HTML format.
    EXPLAIN BULGARIAN NOUN GENDERS (Masculine, Feminine, Neuter) AND DEFINITE ARTICLES (postfixed articles: -ът/-а, -та, -то, -те) TO A 5-YEAR-OLD CHILD IN PORTUGUESE.
    
    Requirements:
    - Language of explanation: Portuguese.
    - Style: Extremely simple, fun, as if explaining to a 5-year-old child.
    - Explain how Bulgarian articles attach to the end of words (postfixes).
    - Provide 1-2 ultra-short example words/sentences in Bulgarian with Portuguese translations for each gender.
    - Output MUST be valid Telegram HTML (use <b>, <i>, <code>).
    """,

    1: """
    You are a Bulgarian journalist and linguist. Create a Telegram post in HTML format.
    WRITE AN INTERESTING WORLD NEWS ARTICLE IN BULGARIAN.
    
    Requirements:
    - Length: Short article (150-250 words).
    - Target vocabulary: Pick EXACTLY 10 words outside the top 500 most common Bulgarian words, but within the top 3000 used in everyday Bulgaria.
    - Format in the article: Wrap those 10 specific words in <b>BOLD</b>.
    - Below the article, add a section labeled "<b>Vocabulário / Речник:</b>".
    - List the 10 bolded words in a numbered list with their direct Portuguese translation and a brief context note.
    """,

    2: """
    You are an expert Bulgarian teacher. Create a Telegram post in HTML format.
    PROVIDE A CLEAR OVERVIEW OF EVERYDAY BULGARIAN TENSES (Presente - Сегашно време, Pretérito Perfeito - Минало свършено време, Futuro - Бъдеще време).
    
    Requirements:
    - Language of explanation: Portuguese.
    - Focus strictly on tenses used in daily spoken/written Bulgarian.
    - Show clear, simple mathematical-like formulas for each tense (e.g., Sujeito + ще + Verbo).
    - Provide 1 clear short example per tense in Bulgarian with Portuguese translation.
    - Keep explanations clean, practical, and highly visual.
    """,

    3: """
    You are a native Bulgarian speaker from Sofia. Create a Telegram post in HTML format.
    CREATE A SHORT COLLOQUIAL BULGARIAN DIALOGUE OR PHRASE (2-3 SENTENCES) DENSE WITH SPOKEN PARTICLES.
    
    Requirements:
    - Include native particles like: бе (be), ма (ma), де (de), па (pa), я (ja), беле, etc.
    - Sound 100% like native friends talking casually on the street.
    - Below the dialogue, EXPLAIN IT TO A 5-YEAR-OLD CHILD IN PORTUGUESE.
    - Break down how each particle changes the emotion or tone of the sentence, translated to Portuguese.
    """,

    4: """
    You are an expert Bulgarian grammar tutor. Create a Telegram post in HTML format.
    EXPLAIN BULGARIAN SENTENCE STRUCTURE AND SHORT PRONOUN CLITICS (го, му, се, си, ги, им) TO A 5-YEAR-OLD CHILD IN PORTUGUESE.
    
    Requirements:
    - Language of explanation: Portuguese.
    - Cover basic SVO word order and explain where short pronoun clitics go in a sentence.
    - Use dead-simple metaphors a child understands.
    - Include short Bulgarian example sentences with Portuguese translations.
    """,

    5: """
    You are a Bulgarian language coach. Create a Telegram post in HTML format.
    PROVIDE THE 10 MOST POPULAR BULGARIAN WORD COMBINATIONS / IDIOMS (e.g., "Няма проблеми", "Имаш ли предвид", "С един куршум два заека").
    
    Requirements:
    - Language of explanation: Portuguese.
    - Format: Numbered list 1 to 10.
    - For each phrase: Show the Bulgarian phrase, direct/figurative Portuguese translation, and a 1-sentence grammar/usage tip explained as if for a 5-year-old.
    - Keep it short, actionable, and punchy.
    """,

    6: """
    You are a Bulgarian language coach. Create a Telegram post in HTML format.
    PROVIDE ALL BULGARIAN QUESTION WORDS AND PREPOSITIONAL QUESTIONS (Кой, Какво, Къде, Защо, Кога, С кого, За кого, etc.).
    
    Requirements:
    - Order: From most frequently used in daily life to least frequently used.
    - Language of explanation: Portuguese.
    - Include basic question words AND prepositional questions.
    - Provide direct Portuguese translations for every single question word/phrase.
    - Format as a clean, structured reference list.
    """
}

DAY_TITLES = {
    0: "Segunda-feira: Gêneros e Artigos Definidos (Para Crianças)",
    1: "Terça-feira: Notícias Mundiais + Vocabulário Búlgaro",
    2: "Quarta-feira: Tempos Verbais Práticos (Fórmulas)",
    3: "Quinta-feira: Búlgaro Coloquial & Partículas (Бе, Де, Ма)",
    4: "Sexta-feira: Estrutura de Frases e Pronomes Curtos",
    5: "Sábado: Top 10 Expressões e Idiomas em Búlgaro",
    6: "Domingo: Todas as Palavras Interrogativas em Búlgaro"
}

# ---------------------------------------------------------------------------
# CORE LOGIC WITH FALLBACK ENGINE
# ---------------------------------------------------------------------------

def generate_with_deepseek(prompt: str, system_instruction: str) -> str:
    """Fallback generation engine using DeepSeek API via REST."""
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    
    if not deepseek_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    res = requests.post(url, json=payload, timeout=30)
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"]

def generate_daily_lesson(day_index: int) -> str:
    """Attempts Gemini 3.6 Flash first; falls back to DeepSeek on failure."""
    system_instruction = (
    "You are an expert Bulgarian language educator producing daily learning posts for a Telegram group. "
    "Explanations MUST be written in strict European Portuguese (Português de Portugal / PT-PT), "
    "using European Portuguese vocabulary, grammar, and syntax (e.g., avoid Brazilian gerunds, use PT-PT terms). "
    "Outputs MUST be formatted exclusively using standard Telegram HTML tags (<b>, <i>, <code>). "
    "Do NOT enclose the entire response inside markdown ```html codeblocks. Return clean HTML content."
)
    prompt = PROMPTS[day_index]

    # Primary attempt: Gemini 3.6 Flash
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        print("Attempting generation via Gemini 3.6 Flash...")
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                temperature=0.4,
            )
        )
        return response.text

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        print("🔄 Switching to DeepSeek fallback...")
        return generate_with_deepseek(prompt, system_instruction)

def format_telegram_message(day_index: int, content: str) -> str:
    """Prepends a formatted header and strips code fences."""
    header = f"<b>🇧🇬 Língua Búlgara — Dia {day_index + 1}/7</b>\n<b>📌 Tópico: {DAY_TITLES[day_index]}</b>\n\n"
    cleaned_content = content.replace("```html", "").replace("```", "").strip()
    return header + cleaned_content

def send_telegram_message(message_text: str):
    """Sends payload to Telegram API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    res = requests.post(url, json=payload, timeout=15)
    res.raise_for_status()
    print("Successfully delivered daily lesson to Telegram!")

def main():
    today_index = datetime.datetime.now(datetime.timezone.utc).weekday()
    print(f"Executing automation for Day Index {today_index} ({DAY_TITLES[today_index]})")

    raw_lesson = generate_daily_lesson(today_index)
    final_message = format_telegram_message(today_index, raw_lesson)
    send_telegram_message(final_message)

if __name__ == "__main__":
    main()