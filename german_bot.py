import os
import datetime
from dotenv import load_dotenv
import requests
from google import genai
from google.genai import types


# Load local .env file variables into os.environ
load_dotenv()

# ---------------------------------------------------------------------------
# WEEKLY PROMPT SUITE
# ---------------------------------------------------------------------------

PROMPTS = {
    # 0 = MONDAY (#1: Cases & Genders)
    0: """
    You are an expert German language teacher. Create a Telegram post in HTML format.
    EXPLAIN ALL 4 GERMAN CASES (Nominativ, Akkusativ, Dativ, Genitiv) AND GENDERS (der, die, das) TO A 5-YEAR-OLD CHILD IN BULGARIAN.
    
    Requirements:
    - Language of explanation: Bulgarian.
    - Style: Extremely simple, fun, as if explaining to a 5-year-old child.
    - Include der, die, das explanations briefly.
    - For each case, provide maximum 1-2 ultra-short example sentences in German with Bulgarian translation.
    - Output MUST be valid Telegram HTML (use <b>, <i>, <code>).
    """,

    # 1 = TUESDAY (#2: Global News with Oberbayern Vocab Focus)
    1: """
    You are a German journalist and linguist. Create a Telegram post in HTML format.
    WRITE AN INTERESTING WORLD NEWS ARTICLE IN GERMAN.
    
    Requirements:
    - Length: Short article (150-250 words).
    - Target vocabulary: Pick EXACTLY 10 words outside the top 500 most common German words, but within the top 3000 used in Germany (specifically common in Bavaria / Upper Bavaria / Oberbayern).
    - Format in the article: Wrap those 10 specific words in <b>BOLD</b>.
    - Below the article, add a section labeled "<b>Vocabulary / Речник (Oberbayern focus):</b>".
    - List the 10 bolded words in a numbered list with their direct Bulgarian translation and a brief context note.
    """,

    # 2 = WEDNESDAY (#3: Tenses Overview)
    2: """
    You are an expert German teacher. Create a Telegram post in HTML format.
    PROVIDE A CLEAR OVERVIEW OF ALL EVERYDAY GERMAN TENSES (Präsens, Perfekt, Präteritum, Futur I).
    
    Requirements:
    - Language of explanation: Bulgarian.
    - Focus strictly on tenses used in daily spoken/written German.
    - Show clear, universal mathematical-like formulas/structures for each tense (e.g., Subjekt + haben/sein + ... + Partizip II).
    - Provide 1 clear short example per tense with Bulgarian translation.
    - Keep explanations clean, practical, and highly visual.
    """,

    # 3 = THURSDAY (#4: Colloquial Modal Particle Heavy Dialogue)
    3: """
    You are a native German speaker from Bavaria. Create a Telegram post in HTML format.
    CREATE A SHORT COLLOQUIAL GERMAN DIALOGUE OR PHRASE (2-3 SENTENCES) DENSE WITH MODAL PARTICLES.
    
    Requirements:
    - Include particles like: da, mal, doch, aber, nach, um, schon, zwar, allerdings, anlässlich, gib, etc.
    - Sound 100% like native friends talking casually.
    - Below the phrase, EXPLAIN IT TO A 5-YEAR-OLD CHILD IN BULGARIAN.
    - Break down how each particle changes the emotion or nuance of the sentence, translated to Bulgarian.
    """,

    # 4 = FRIDAY (#5: Sentence Structures & Verb Position)
    4: """
    You are an expert German grammar tutor. Create a Telegram post in HTML format.
    EXPLAIN GERMAN SENTENCE TYPES ORDERED FROM MOST COMMON TO LEAST COMMON TO A 5-YEAR-OLD CHILD IN BULGARIAN.
    
    Requirements:
    - Language of explanation: Bulgarian.
    - Cover: Hauptsatz (V2 rule), Nebensatz (verb at the end), Relativsatz, Question structures (V1 rule), etc.
    - Highlight clearly WHERE THE VERB GOES in each structure.
    - Use dead-simple metaphors a child understands.
    - Include short German example sentences with Bulgarian translations.
    """,

    # 5 = SATURDAY (#6: Top 10 Idioms / Fixed Phrases)
    5: """
    You are a German language coach. Create a Telegram post in HTML format.
    PROVIDE THE 10 MOST POPULAR GERMAN WORD COMBINATIONS / IDIOMS (e.g., "Im Großen und Ganzen", "Auf jeden Fall", "Ab und zu").
    
    Requirements:
    - Language of explanation: Bulgarian.
    - Format: Numbered list 1 to 10.
    - For each phrase: Show the German phrase, direct/figurative Bulgarian translation, and a 1-sentence grammar/usage tip explained as if for a 5-year-old.
    - Keep it short, actionable, and punchy.
    """,

    # 6 = SUNDAY (#7: Question Words & Case Questions)
    6: """
    You are a German language coach. Create a Telegram post in HTML format.
    PROVIDE ALL GERMAN QUESTION WORDS AND PREPOSITIONAL QUESTIONS (W-Fragen + Präpositionalfragen: wofür, womit, an wen, worauf, etc.).
    
    Requirements:
    - Order: From most frequently used in daily life to least frequently used.
    - Language of explanation: Bulgarian.
    - Include basic W-questions AND prepositional questions with cases.
    - Provide direct Bulgarian translations for every single question word/phrase.
    - Format as a clean, structured reference list.
    """
}

DAY_TITLES = {
    0: " понеделник: Падежите и родовете (За 5-годишни)",
    1: " вторник: Световни новини + Баварски речник",
    2: " сряда: Всички времена в немския (Формули)",
    3: " четвъртък: Разговорен немски & Партикли",
    4: " петък: Видове изречения & Място на глагола",
    5: " събота: Топ 10 немски фрази и идиоми",
    6: " неделя: Всички въпросителни думи в немския"
}

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def generate_daily_lesson(day_index: int) -> str:
    """Invokes Gemini API with the selected day's prompt using Gemini 3.6 Flash thinking."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=gemini_key)
    
    system_instruction = (
        "You are an expert German language educator producing daily learning posts for a Telegram group. "
        "Outputs MUST be formatted exclusively using standard Telegram HTML tags (<b>, <i>, <code>). "
        "Do NOT enclose the entire response inside markdown ```html codeblocks. Return clean HTML content."
    )

    prompt = PROMPTS[day_index]

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # Configures Gemini 3 reasoning/thinking behavior
            thinking_config=types.ThinkingConfig(
                thinking_level="high"  # Options: "high" (deeper reasoning) or "low" (faster execution)
            ),
            temperature=0.4,
        )
    )
    return response.text

def format_telegram_message(day_index: int, content: str) -> str:
    """Prepend clean header and wrap content."""
    header = f"<b>🇩🇪 Немски език — Ден {day_index + 1}/7</b>\n<b>📌 Topic: {DAY_TITLES[day_index]}</b>\n\n"
    
    # Strip accidental code fences if LLM includes them
    cleaned_content = content.replace("```html", "").replace("```", "").strip()
    return header + cleaned_content

def send_telegram_message(message_text: str):
    """Sends HTML message to Telegram group or chat."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

    # Clean raw URL string (NO markdown brackets):
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
    # Detect current day of week UTC (0 = Monday, 6 = Sunday)
    today_index = datetime.datetime.now(datetime.timezone.utc).weekday()
    print(f"Executing automation for Day Index {today_index} ({DAY_TITLES[today_index]})")

    raw_lesson = generate_daily_lesson(today_index)
    final_message = format_telegram_message(today_index, raw_lesson)
    send_telegram_message(final_message)

if __name__ == "__main__":
    main()