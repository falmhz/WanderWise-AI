"""
WanderWise AI — Smart Travel & Culinary Guide
======================================================================
Aplikasi SaaS cerdas untuk perencanaan perjalanan, eksplorasi kuliner lokal,
tanya jawab dokumen perjalanan berbasis RAG, dan analitik database otonom
bertenaga Google Gemini, LangChain, FAISS, dan SQLite.

Modul Utama:
1. Mode 1: Travel Chat & Persona Eksploratif
2. Mode 2: RAG Dokumen Perjalanan (Ekstraksi PDF & Semantic Search)
3. Mode 3: AI Travel Data Analyst Agent (ReAct Tool-Calling SQL)

Cara Menjalankan:
>>> streamlit run app_integrated.py
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pymupdf
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import create_agent

# ── Konfigurasi Halaman Streamlit ─────────────────────────────────────────────
st.set_page_config(
    page_title="WanderWise AI — Smart Travel & Culinary Guide",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS: DESAIN SAAS MODERN & PALET EMERALD / OCEAN BLUE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Import Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Primary Color Theme Variables (Emerald & Ocean Blue) ───────── */
    :root {
        --ww-primary: #0d9488;         /* Emerald Teal */
        --ww-primary-hover: #0f766e;
        --ww-secondary: #0284c7;       /* Ocean Blue */
        --ww-secondary-hover: #0369a1;
        --ww-accent: #10b981;          /* Bright Emerald */
        --ww-bg-card: rgba(255, 255, 255, 0.95);
        --ww-border: rgba(226, 232, 240, 0.9);
        --ww-text-main: #0f172a;
        --ww-text-muted: #64748b;
    }

    /* ── Override Streamlit Slider Red Accent to Emerald/Ocean ──────── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #0d9488 !important;
        border-color: #0d9488 !important;
        box-shadow: 0 0 8px rgba(13, 148, 136, 0.4) !important;
    }
    .stSlider [data-baseweb="slider"] > div:first-child > div:nth-child(1) {
        background: linear-gradient(90deg, #0d9488 0%, #0284c7 100%) !important;
    }
    .stSlider [data-baseweb="slider"] div div {
        background-color: #0d9488 !important;
    }

    /* ── Custom Buttons (Modern Gradient & Subtle Lift) ─────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 3px 10px rgba(13, 148, 136, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(13, 148, 136, 0.4) !important;
        color: #ffffff !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Metric Cards SaaS Styling ──────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(13, 148, 136, 0.3) !important;
        padding: 0.9rem 1.15rem;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(13, 148, 136, 0.55);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        color: #2dd4bf !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin-top: 0.2rem !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
    }

    /* ── Radio & Tabs Accent ─────────────────────────────────────────── */
    div[data-testid="stRadio"] label[data-checked="true"] > div:first-child {
        border-color: #0d9488 !important;
        background-color: #0d9488 !important;
    }
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        font-size: 0.92rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2dd4bf !important;
        border-bottom-color: #0d9488 !important;
    }

    /* ── Chat Messages Elegance (Dark Mode - Abu-Abu Gelap) ─────────── */
    .stChatMessage {
        border-radius: 14px !important;
        padding: 1.1rem 1.35rem !important;
        margin-bottom: 0.95rem !important;
        border: 1px solid rgba(71, 85, 105, 0.5) !important;
        line-height: 1.68 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    }
    /* Asisten AI: Abu-abu gelap (Slate 800) beraksen Emerald */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        background-color: #1e293b !important;
        border-left: 4px solid #0d9488 !important;
    }
    /* Pesan User: Abu-abu gelap pekat (Slate 900) beraksen Ocean Blue */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        background-color: #0f172a !important;
        border-left: 4px solid #0284c7 !important;
    }
    /* Tipografi teks di dalam Chat agar kontras & nyaman dibaca */
    .stChatMessage p, .stChatMessage li, .stChatMessage span {
        color: #e2e8f0 !important;
    }
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
        color: #38bdf8 !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
    }
    .stChatMessage strong {
        color: #f8fafc !important;
    }

    /* ── Sidebar Branding & Styling ──────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #0B192C !important;
        border-right: 1px solid rgba(51, 65, 85, 0.6) !important;
    }
    .sidebar-brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.2), rgba(2, 132, 199, 0.2));
        color: #2dd4bf;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #10b981;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10b981;
    }

    /* ── Expander & Status Polish ────────────────────────────────────── */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid rgba(51, 65, 85, 0.6) !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stStatusWidget"] {
        border-radius: 12px !important;
        border: 1px solid rgba(13, 148, 136, 0.4) !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION (wanderwise_travel.db)
# ══════════════════════════════════════════════════════════════════════════════
DB_FILE = "wanderwise_travel.db"

def init_sqlite_database(db_path: str = DB_FILE):
    """Membuat tabel dan data awal untuk WanderWise Travel Knowledge Base."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS destinations (
        destination_id INTEGER PRIMARY KEY AUTOINCREMENT,
        city VARCHAR(255) NOT NULL,
        country VARCHAR(100) NOT NULL,
        region VARCHAR(100) NOT NULL,
        best_season VARCHAR(50) NOT NULL,
        avg_daily_budget_usd DECIMAL(10, 2) NOT NULL,
        rating DECIMAL(3, 1) NOT NULL,
        highlights TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS accommodations (
        accommodation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        destination_id INTEGER NOT NULL,
        type VARCHAR(50) NOT NULL,
        price_per_night_usd DECIMAL(10, 2) NOT NULL,
        rating DECIMAL(3, 1) NOT NULL,
        amenities TEXT NOT NULL,
        FOREIGN KEY (destination_id) REFERENCES destinations (destination_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS local_cuisines (
        cuisine_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dish_name VARCHAR(255) NOT NULL,
        destination_id INTEGER NOT NULL,
        cuisine_type VARCHAR(100) NOT NULL,
        avg_price_usd DECIMAL(10, 2) NOT NULL,
        spice_level VARCHAR(20) NOT NULL,
        is_vegetarian BOOLEAN NOT NULL,
        description TEXT NOT NULL,
        FOREIGN KEY (destination_id) REFERENCES destinations (destination_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
        traveler_name VARCHAR(255) NOT NULL,
        destination_id INTEGER NOT NULL,
        accommodation_id INTEGER NOT NULL,
        check_in DATE NOT NULL,
        check_out DATE NOT NULL,
        num_guests INTEGER NOT NULL,
        total_cost_usd DECIMAL(10, 2) NOT NULL,
        status VARCHAR(50) NOT NULL,
        FOREIGN KEY (destination_id) REFERENCES destinations (destination_id),
        FOREIGN KEY (accommodation_id) REFERENCES accommodations (accommodation_id)
    );
    """)

    cursor.execute("SELECT COUNT(*) FROM destinations;")
    if cursor.fetchone()[0] == 0:
        destinations_data = [
            ("Ubud", "Indonesia", "Southeast Asia", "Apr-Oct", 45.00, 4.7, "Rice terraces, temples, art galleries, yoga retreats"),
            ("Kyoto", "Japan", "East Asia", "Mar-May / Oct-Nov", 120.00, 4.9, "Bamboo grove, Fushimi Inari, geisha district, tea ceremonies"),
            ("Hallstatt", "Austria", "Europe", "Jun-Sep", 95.00, 4.6, "Alpine lake, salt mines, charming village, hiking trails"),
            ("Marrakech", "Morocco", "North Africa", "Mar-May / Sep-Nov", 40.00, 4.5, "Souks, Jardin Majorelle, Djemaa el-Fna, riads"),
            ("Queenstown", "New Zealand", "Oceania", "Dec-Feb", 110.00, 4.8, "Bungee jumping, Milford Sound, wine tours, skiing"),
            ("Cartagena", "Colombia", "South America", "Dec-Apr", 55.00, 4.4, "Old walled city, beaches, salsa dancing, street food"),
            ("Chiang Mai", "Thailand", "Southeast Asia", "Nov-Feb", 30.00, 4.6, "Night bazaar, elephant sanctuaries, temples, cooking classes"),
            ("Santorini", "Greece", "Europe", "May-Oct", 130.00, 4.8, "Caldera views, blue domes, wine tasting, sunset at Oia"),
            ("Cusco", "Peru", "South America", "May-Sep", 35.00, 4.5, "Machu Picchu gateway, Inca ruins, alpaca textiles, pisco sour"),
            ("Hoi An", "Vietnam", "Southeast Asia", "Feb-May", 25.00, 4.7, "Lantern-lit old town, tailor shops, banh mi, cooking classes"),
        ]
        cursor.executemany(
            "INSERT INTO destinations (city, country, region, best_season, avg_daily_budget_usd, rating, highlights) VALUES (?, ?, ?, ?, ?, ?, ?);",
            destinations_data,
        )

        accommodations_data = [
            ("Kamandalu Ubud Resort", 1, "Resort", 180.00, 4.8, "Infinity pool, spa, rice paddy view, free yoga"),
            ("Ubud Backpacker Hostel", 1, "Hostel", 12.00, 4.2, "Shared dorm, free breakfast, bike rental"),
            ("Kyoto Machiya Townhouse", 2, "Traditional Inn", 95.00, 4.7, "Tatami rooms, garden view, tea set"),
            ("Hotel Granvia Kyoto", 2, "Hotel", 150.00, 4.5, "Station-connected, rooftop bar, concierge"),
            ("Heritage Hotel Hallstatt", 3, "Boutique Hotel", 130.00, 4.6, "Lake view, Austrian breakfast, sauna"),
            ("Riad Yasmine Marrakech", 4, "Riad", 75.00, 4.7, "Rooftop plunge pool, courtyard, moroccan breakfast"),
            ("Sofitel Queenstown", 5, "Hotel", 220.00, 4.8, "Lake Wakatipu view, spa, fine dining"),
            ("Casa del Alma Cartagena", 6, "Boutique Hotel", 65.00, 4.5, "Rooftop pool, colonial architecture, breakfast included"),
            ("Zela Chiang Mai", 7, "Boutique Hotel", 40.00, 4.6, "Garden pool, cooking class, tuk-tuk service"),
            ("Canaves Oia Santorini", 8, "Luxury Villa", 350.00, 4.9, "Private pool, caldera view, sunset terrace, butler service"),
            ("Palacio Nazarenas Cusco", 9, "Luxury Hotel", 200.00, 4.7, "Oxygen-enriched rooms, Inca walls, heated pool"),
            ("An Bang Seaside Village", 10, "Homestay", 18.00, 4.4, "Beach access, family-run, bicycle included"),
        ]
        cursor.executemany(
            "INSERT INTO accommodations (name, destination_id, type, price_per_night_usd, rating, amenities) VALUES (?, ?, ?, ?, ?, ?);",
            accommodations_data,
        )

        cuisines_data = [
            ("Nasi Campur Bali", 1, "Indonesian", 3.50, "Medium", False, "Mixed rice plate with lawar, sate lilit, sambal matah, and crispy tempeh"),
            ("Babi Guling", 1, "Indonesian", 5.00, "Medium", False, "Balinese-style roasted suckling pig with spiced rice and lawar"),
            ("Kaiseki Ryori", 2, "Japanese", 80.00, "Mild", False, "Multi-course traditional Japanese dinner with seasonal ingredients"),
            ("Matcha Parfait", 2, "Japanese", 8.00, "None", True, "Layered dessert with Uji matcha ice cream, mochi, and red bean"),
            ("Wiener Schnitzel", 3, "Austrian", 15.00, "Mild", False, "Breaded and pan-fried veal cutlet served with lingonberry jam"),
            ("Tagine Lamb", 4, "Moroccan", 7.00, "Medium", False, "Slow-cooked lamb stew with apricots, almonds, and saffron in clay pot"),
            ("Mint Tea", 4, "Moroccan", 1.00, "None", True, "Traditional Moroccan sweet mint tea poured from height"),
            ("Fergburger", 5, "New Zealand", 12.00, "Mild", False, "Legendary gourmet burger from Queenstown's iconic burger joint"),
            ("Ceviche Cartagenero", 6, "Colombian", 6.00, "Medium", False, "Fresh seafood marinated in lime juice with coconut milk and cilantro"),
            ("Khao Soi", 7, "Thai", 2.50, "Hot", False, "Northern Thai coconut curry noodle soup with crispy noodle topping"),
            ("Mango Sticky Rice", 7, "Thai", 1.50, "None", True, "Sweet glutinous rice with fresh mango and coconut cream"),
            ("Moussaka", 8, "Greek", 10.00, "Mild", True, "Layered eggplant, potato, and bechamel sauce baked casserole"),
            ("Cuy al Horno", 9, "Peruvian", 9.00, "Medium", False, "Oven-roasted guinea pig, a traditional Andean delicacy"),
            ("Cao Lau", 10, "Vietnamese", 2.00, "Mild", False, "Hoi An specialty noodles with pork, herbs, and crispy croutons"),
        ]
        cursor.executemany(
            "INSERT INTO local_cuisines (dish_name, destination_id, cuisine_type, avg_price_usd, spice_level, is_vegetarian, description) VALUES (?, ?, ?, ?, ?, ?, ?);",
            cuisines_data,
        )

        bookings_data = [
            ("Ayu Lestari", 1, 1, "2026-10-15", "2026-10-20", 2, 900.00, "Confirmed"),
            ("Takeshi Yamamoto", 2, 3, "2026-11-01", "2026-11-05", 1, 380.00, "Confirmed"),
            ("Emma Müller", 3, 5, "2026-07-10", "2026-07-14", 2, 520.00, "Confirmed"),
            ("Youssef Amine", 4, 6, "2026-04-05", "2026-04-09", 3, 300.00, "Confirmed"),
            ("Sophie Clark", 5, 7, "2026-01-20", "2026-01-25", 2, 1100.00, "Confirmed"),
            ("Carlos Rivera", 6, 8, "2026-03-12", "2026-03-16", 2, 260.00, "Cancelled"),
            ("Mia Jensen", 7, 9, "2026-12-01", "2026-12-07", 1, 240.00, "Confirmed"),
            ("Olivia Rossi", 8, 10, "2026-06-18", "2026-06-23", 2, 1750.00, "Confirmed"),
            ("Raul Gutierrez", 9, 11, "2026-08-01", "2026-08-04", 2, 600.00, "Pending"),
            ("Linh Nguyen", 10, 12, "2026-03-20", "2026-03-25", 1, 90.00, "Confirmed"),
            ("Ayu Lestari", 7, 9, "2026-12-15", "2026-12-20", 2, 200.00, "Pending"),
            ("Takeshi Yamamoto", 8, 10, "2026-05-10", "2026-05-15", 2, 1750.00, "Confirmed"),
        ]
        cursor.executemany(
            "INSERT INTO bookings (traveler_name, destination_id, accommodation_id, check_in, check_out, num_guests, total_cost_usd, status) VALUES (?, ?, ?, ?, ?, ?, ?);",
            bookings_data,
        )
        conn.commit()

    conn.close()

init_sqlite_database()

# ══════════════════════════════════════════════════════════════════════════════
# SQL DATABASE HELPER TOOLS
# ══════════════════════════════════════════════════════════════════════════════
def sql_list_tables() -> List[str]:
    """Mengembalikan daftar semua nama tabel dalam database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in cursor.fetchall()]

def sql_describe_table(table_name: str) -> List[Dict[str, Any]]:
    """Mengambil skema kolom dari tabel tertentu."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        cols = cursor.fetchall()
        return [{"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3], "pk": c[5]} for c in cols]

def sql_execute_query(query: str) -> Dict[str, Any]:
    """Mengeksekusi query read-only SELECT ke database SQLite."""
    query_clean = query.strip()
    if not query_clean.lower().startswith("select") and not query_clean.lower().startswith("with"):
        return {"error": "Hanya perintah SELECT yang diizinkan demi keamanan database."}
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query_clean)
            col_names = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {"columns": col_names, "rows": rows, "row_count": len(rows)}
        except Exception as err:
            return {"error": str(err)}

# ══════════════════════════════════════════════════════════════════════════════
# CONTENT FORMATTER (Format Response & Multipart Handling)
# ══════════════════════════════════════════════════════════════════════════════
def format_llm_content(content: Any) -> str:
    """Mengekstrak string teks bersih dari format response model."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "thought":
                    continue
                if "text" in part:
                    texts.append(part["text"])
                elif "content" in part:
                    texts.append(str(part["content"]))
            elif hasattr(part, "text"):
                texts.append(part.text)
        if not texts and content:
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
        return "\n\n".join(texts).strip()
    return str(content).strip()

# ══════════════════════════════════════════════════════════════════════════════
# RAG DOCUMENT LOADER & VECTOR STORE (PyMuPDF + FAISS)
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """Mengekstrak teks halaman demi halaman menggunakan PyMuPDF."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    documents = []
    for page_idx, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_idx + 1}
                )
            )
    return documents

def build_vector_store(documents: List[Document], api_key: str, provider: str = "Google Gemini"):
    """Membagi teks menjadi chunks dan membangun FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    if not split_docs:
        return None, 0
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore, len(split_docs)
    except Exception as e:
        from langchain_community.embeddings import FakeEmbeddings
        st.warning(f"Menggunakan fallback embeddings: {e}")
        fallback = FakeEmbeddings(size=768)
        vectorstore = FAISS.from_documents(split_docs, fallback)
        return vectorstore, len(split_docs)

# ══════════════════════════════════════════════════════════════════════════════
# INISIALISASI LLM CLIENT (TANPA @st.cache_resource)
# ══════════════════════════════════════════════════════════════════════════════
def get_llm_client(provider_choice: str, key: str, model_id: str, temp: float, p: float, max_tok: int):
    """
    Membuat instance ChatModel dari LangChain.
    Decorator @st.cache_resource sengaja dihapus agar perubahan parameter
    (seperti max_tokens dan temperature) langsung berefek secara instan.
    """
    if "Gemini" in provider_choice:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=key,
            temperature=temp,
            top_p=p,
            max_output_tokens=max_tok,
        )
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_id,
            api_key=key,
            temperature=temp,
            max_tokens=max_tok,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: KONTROL, PARAMETER & NAVIGASI
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand-badge">Travel Intelligence Suite</div>', unsafe_allow_html=True)
    st.markdown("## 🌏 WanderWise AI")
    st.caption("Asisten Perjalanan & Kuliner Berbasis AI")

    st.markdown("---")

    # Section 1: Konfigurasi Model
    st.markdown("### ⚙️ Konfigurasi Model")
    provider = st.selectbox(
        "Penyedia Layanan (LLM Provider)",
        ["Google Gemini (Rekomendasi)", "Groq API (Alternatif)"],
        index=0,
    )

    if "Gemini" in provider:
        api_key = st.text_input(
            "Google AI Studio API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Dapatkan API Key di https://aistudio.google.com"
        )
        model_name = st.selectbox(
            "Pilih Model Gemini",
            ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
            index=0
        )
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Dapatkan API Key di https://console.groq.com"
        )
        model_name = st.selectbox(
            "Pilih Model Groq",
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
            index=0
        )

    st.markdown("---")

    # Section 2: Hyperparameter Tuning
    st.markdown("### Parameter Model")
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="0.0 = Presisi & faktual. 1.0 = Kreatif & eksploratif."
    )
    top_p = st.slider(
        "Top-P Sampling",
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
        help="Probabilitas kumulatif untuk pemilihan kata."
    )
    # Default 2560 token untuk mencegah teks terpotong
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=512, max_value=4096, value=2560, step=256,
        help="Batas maksimum panjang respon. Nilai 2560 memastikan itinerary dan analisis komprehensif tidak terpotong."
    )

    st.markdown("---")

    # Section 3: Mode Aplikasi
    st.markdown("### Kemampuan AI")
    app_mode = st.radio(
        "Pilih Mode Operasional:",
        [
            "Mode 1: Travel Chat & Persona",
            "Mode 2: RAG Dokumen Perjalanan",
            "Mode 3: AI Travel Data Analyst",
        ],
        index=0,
    )

    st.markdown("---")

    # Section 4: Aksi & Status
    col_btn, col_status = st.columns([1.5, 1])
    with col_btn:
        if st.button("🔄 Reset Sesi", help="Bersihkan riwayat chat dan memori sementara"):
            st.session_state["messages"] = []
            st.session_state.pop("rag_vectorstore", None)
            st.session_state.pop("rag_source", None)
            st.session_state.pop("rag_chunks", None)
            st.rerun()
    with col_status:
        st.markdown(
            '<div class="status-indicator"><span class="status-dot"></span> Online</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = app_mode
elif st.session_state["current_mode"] != app_mode:
    st.session_state["current_mode"] = app_mode

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER & SAAS OVERVIEW BANNER (MENGGANTIKAN EXPANDER BERTUMPUK)
# ══════════════════════════════════════════════════════════════════════════════
col_header_text, col_header_badge = st.columns([3.5, 1])
with col_header_text:
    st.title("🧭 **Smart Travel & Culinary Intelligence**")
    st.markdown(
        "Asisten perjalanan cerdas untuk perencanaan itinerary, "
        "rekomendasi kuliner autentik, Q&A dokumen wisata (RAG), dan analitik data pariwisata real-time."
    )

# Banner Kartu Ringkas (Product Highlights)
col_card1, col_card2, col_card3 = st.columns(3)
with col_card1:
    st.metric(
        label="Destinasi Terkurasi",
        value="10+ Destinasi",
        delta="Asia, Eropa, Afrika, Amerika",
        delta_color="normal"
    )
with col_card2:
    st.metric(
        label="Knowledge Hub (RAG)",
        value="FAISS Vector",
        delta="PDF Semantic Search & Citations",
        delta_color="normal"
    )
with col_card3:
    st.metric(
        label="Autonomous SQL Agent",
        value="SQLite ReAct",
        delta="Direct 1-Turn Analytical Query",
        delta_color="normal"
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# Validasi API Key
if not api_key:
    st.info(" **Mulai Menjelajah:** Masukkan API Key Anda pada panel navigasi di sebelah kiri untuk mengaktifkan seluruh fitur WanderWise AI.", icon="🔑")
    st.stop()

# Instansiasi Model
try:
    llm = get_llm_client(provider, api_key, model_name, temperature, top_p, max_tokens)
except Exception as e:
    st.error(f"Gagal menginisialisasi model: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: TRAVEL CHAT & PERSONA EKSPLORATIF
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Mode 1: Travel Chat & Persona":
    st.markdown("### 💬 Percakapan Bebas & Persona Spesialis")
    st.caption("Konsultasikan rencana perjalanan, eksplorasi kuliner lokal, strategi hemat, atau sejarah budaya destinasi impian Anda.")

    # Panel Persona
    with st.expander(" Pengaturan Persona & System Prompt", expanded=False):
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            persona_preset = st.selectbox(
                "Pilihan Persona:",
                [
                    "Travel Expert (Destinasi & Itinerary)",
                    "Culinary Guide (Kuliner Lokal & Rekomendasi)",
                    "Budget Travel Advisor (Hemat & Efisien)",
                    "Cultural Historian (Sejarah & Tradisi)",
                    "Custom Persona"
                ]
            )
        with col_p2:
            if "Travel Expert" in persona_preset:
                default_sys = (
                    "Anda adalah Senior Travel Expert yang telah menjelajahi lebih dari 50 negara. "
                    "Bantu wisatawan merencanakan itinerary yang seimbang, merekomendasikan destinasi hidden gems, "
                    "memberikan saran transportasi lokal terbaik, dan menyusun waktu kunjungan yang optimal. "
                    "Gunakan gaya bahasa Indonesia yang ramah, hangat, inspiratif, dan sertakan estimasi biaya bila relevan."
                )
            elif "Culinary Guide" in persona_preset:
                default_sys = (
                    "Anda adalah Culinary Guide dan Food Critic berpengalaman. "
                    "Bantu pengguna menemukan kuliner lokal autentik, jajanan kaki lima legendaris, "
                    "restoran bernuansa budaya khas, serta jelaskan filosofi cita rasa di balik hidangan tersebut. "
                    "Selalu sertakan opsi ramah vegetarian atau halal jika diminta."
                )
            elif "Budget Travel Advisor" in persona_preset:
                default_sys = (
                    "Anda adalah Budget Travel Advisor yang ahli merencanakan perjalanan hemat namun tetap nyaman dan berkesan. "
                    "Berikan tips akomodasi terjangkau, transportasi publik hemat, kuliner lokal bernilai tinggi, "
                    "serta strategi cerdas menghindari biaya tak terduga. Selalu sertakan estimasi budget dalam USD atau mata uang lokal."
                )
            elif "Cultural Historian" in persona_preset:
                default_sys = (
                    "Anda adalah Sejarawan Budaya dan Pemandu Warisan Wisata. "
                    "Jelaskan latar belakang sejarah, tradisi lokal, arsitektur, dan nilai filosofis tempat wisata. "
                    "Edukasi wisatawan tentang etiket lokal, busana yang pantas, dan cara menghormati tradisi masyarakat setempat."
                )
            else:
                default_sys = "Anda adalah asisten perjalanan AI yang ramah, fleksibel, dan siap membantu merencanakan petualangan."

            system_instruction = st.text_area("System Instruction:", value=default_sys, height=95)

    # Prompt Suggestion Chips
    st.markdown("**💡 Inspirasi Pertanyaan Cepat:**")
    sug_col1, sug_col2, sug_col3 = st.columns(3)
    suggested_query = None
    with sug_col1:
        if st.button("🏝️ Itinerary 3 Hari di Ubud", key="sug1"):
            suggested_query = "Buatkan itinerary perjalanan santai 3 hari di Ubud, Bali untuk pasangan yang menyukai alam, seni, dan yoga."
    with sug_col2:
        if st.button("🍜 Rekomendasi Kuliner Autentik Kyoto", key="sug2"):
            suggested_query = "Apa saja kuliner lokal autentik yang wajib dicoba di Kyoto selain ramen, beserta estimasi harganya?"
    with sug_col3:
        if st.button("💰 Tips Backpacker Hemat di Eropa", key="sug3"):
            suggested_query = "Bagaimana tips backpacker hemat keliling Eropa dengan budget harian maksimal $50 per hari?"

    # Render History Chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else "👤"):
            st.markdown(format_llm_content(msg["content"]))

    # Input User
    user_input = st.chat_input("Tanyakan rekomendasi destinasi, itinerary, kuliner, atau tips perjalanan...")
    query_to_send = user_input or suggested_query

    if query_to_send:
        st.session_state["messages"].append({"role": "user", "content": query_to_send})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query_to_send)

        langchain_msgs = [SystemMessage(content=system_instruction)]
        for m in st.session_state["messages"]:
            clean_text = format_llm_content(m["content"])
            if m["role"] == "user":
                langchain_msgs.append(HumanMessage(content=clean_text))
            else:
                langchain_msgs.append(AIMessage(content=clean_text))

        with st.chat_message("assistant", avatar="🧭"):
            with st.spinner("WanderWise sedang menyusun rekomendasi perjalanan terbaik..."):
                try:
                    response = llm.invoke(langchain_msgs)
                    reply_text = format_llm_content(response.content)
                    st.markdown(reply_text)
                    st.session_state["messages"].append({"role": "assistant", "content": reply_text})
                except Exception as ex:
                    err_msg = f"Terjadi kendala saat memproses permintaan: {ex}"
                    st.error(err_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": err_msg})


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: RAG DOKUMEN PERJALANAN (PDF KNOWLEDGE BASE)
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "Mode 2: RAG Dokumen Perjalanan":
    st.markdown("### 📚 Tanya Jawab Dokumen Perjalanan (RAG)")
    st.caption("Unggah panduan wisata, itinerary, atau brosur hotel PDF. AI akan menjawab secara akurat dengan merujuk langsung ke halaman dokumen.")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_pdf = st.file_uploader(
            "Unggah Berkas PDF Perjalanan:",
            type=["pdf"],
            help="Mendukung format PDF e-book panduan wisata, dokumen itinerary, brosur hotel, dsb."
        )
    with col_up2:
        st.markdown("**Atau Gunakan Panduan Sampel:**")
        load_sample_guide = st.button(
            "📖 Muat Sampel Panduan Wisata Bali",
            help="Muat contoh e-book panduan Bali untuk uji coba instan tanpa perlu unggah manual."
        )

    docs_to_index = None
    doc_source_name = ""

    if uploaded_pdf is not None:
        docs_to_index = extract_text_from_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
        doc_source_name = uploaded_pdf.name
    elif load_sample_guide:
        candidate_paths = [
            Path("../../Sesi 2/baliguide-sample-pdf.pdf"),
            Path("Sesi 2/baliguide-sample-pdf.pdf"),
            Path("../baliguide-sample-pdf.pdf"),
            Path("baliguide-sample-pdf.pdf"),
        ]
        sample_path = next((p for p in candidate_paths if p.exists()), None)
        if sample_path:
            with open(sample_path, "rb") as f:
                docs_to_index = extract_text_from_pdf(f.read(), "Bali Travel Guide.pdf")
                doc_source_name = "Bali Travel Guide.pdf"
        else:
            st.warning("Berkas sampel tidak ditemukan di direktori lokal. Silakan unggah PDF Anda sendiri.")

    # Indeks Dokumen ke Vector Store
    if docs_to_index and ("rag_source" not in st.session_state or st.session_state.get("rag_source") != doc_source_name):
        with st.spinner(f"Mengindeks dokumen '{doc_source_name}' ke FAISS Vector Store..."):
            vstore, n_chunks = build_vector_store(docs_to_index, api_key, provider)
            if vstore:
                st.session_state["rag_vectorstore"] = vstore
                st.session_state["rag_source"] = doc_source_name
                st.session_state["rag_chunks"] = n_chunks
                st.success(f"Berhasil mengindeks '{doc_source_name}' ({n_chunks} potongan teks semantik)!")
            else:
                st.error("Gagal mengekstrak teks dari berkas PDF.")

    # Status Dokumen Aktif
    if "rag_vectorstore" in st.session_state:
        st.info(f"📂 **Dokumen Aktif:** {st.session_state.get('rag_source')} — **{st.session_state.get('rag_chunks')} Chunks Terindeks di FAISS**", icon="📄")
    else:
        st.warning("Silakan unggah dokumen PDF perjalanan Anda atau klik tombol sampel di atas untuk memulai.", icon="ℹ️")

    # Render History Chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else "👤"):
            st.markdown(format_llm_content(msg["content"]))

    rag_query = st.chat_input("Tanyakan informasi seputar dokumen perjalanan Anda...")
    if rag_query:
        st.session_state["messages"].append({"role": "user", "content": rag_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(rag_query)

        with st.chat_message("assistant", avatar="🧭"):
            if "rag_vectorstore" not in st.session_state:
                res_warn = "Mohon unggah dan indeks dokumen PDF terlebih dahulu sebelum mengajukan pertanyaan."
                st.warning(res_warn)
                st.session_state["messages"].append({"role": "assistant", "content": res_warn})
            else:
                with st.spinner("Mencari referensi relevan di dalam dokumen..."):
                    vectorstore = st.session_state["rag_vectorstore"]
                    relevant_docs = vectorstore.similarity_search(rag_query, k=3)

                    context_text = ""
                    with st.expander("🔍 Bukti Rujukan Dokumen (Retrieved Chunks):", expanded=False):
                        for idx, d in enumerate(relevant_docs):
                            page_num = d.metadata.get("page", "-")
                            src = d.metadata.get("source", "PDF")
                            st.markdown(f"**Kutipan #{idx+1} (Sumber: {src}, Halaman: {page_num})**")
                            st.text(d.page_content[:350] + ("..." if len(d.page_content) > 350 else ""))
                            context_text += f"\n--- Halaman {page_num} ({src}) ---\n{d.page_content}\n"

                    rag_prompt = f"""Anda adalah WanderWise AI, asisten perjalanan berbasis Retrieval-Augmented Generation (RAG).
Jawab pertanyaan pengguna secara akurat dan HANYA berdasarkan konteks dokumen perjalanan di bawah ini.
Jika informasi tidak tersedia di dalam konteks, sampaikan dengan terus terang bahwa dokumen tidak memuat informasi tersebut.
Format jawaban dalam Bahasa Indonesia yang ramah, jelas, dan profesional.

Konteks Dokumen:
{context_text}

Pertanyaan Pengguna:
{rag_query}

Jawaban Profesional Anda:"""

                    try:
                        resp = llm.invoke([HumanMessage(content=rag_prompt)])
                        ans_content = format_llm_content(resp.content)
                        st.markdown(ans_content)
                        st.session_state["messages"].append({"role": "assistant", "content": ans_content})
                    except Exception as err:
                        st.error(f"Terjadi kesalahan saat mengeksekusi RAG: {err}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: AI TRAVEL DATA ANALYST AGENT (ReAct SQL TOOL-CALLING)
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "Mode 3: AI Travel Data Analyst":
    st.markdown("### 📊 AI Travel Data Analyst Agent")
    st.caption("AI Agent otonom yang langsung menganalisis database SQLite destinasi wisata, akomodasi, kuliner lokal, dan booking untuk menjawab pertanyaan analitik Anda.")

    # Tampilan Database Modern Berbasis Tab (Bukan Stacked Expander yang Berantakan)
    with st.expander("🗄️ Eksplorasi Database WanderWise Travel (Preview)", expanded=False):
        tab_dest, tab_accom, tab_cuis, tab_book = st.tabs([
            "🌍 Destinasi Wisata", "🏨 Akomodasi", "🍜 Kuliner Lokal", "📋 Booking Wisatawan"
        ])
        with tab_dest:
            d_res = sql_execute_query("SELECT destination_id, city, country, region, best_season, avg_daily_budget_usd, rating, highlights FROM destinations;")
            if "columns" in d_res:
                st.dataframe(pd.DataFrame(d_res["rows"], columns=d_res["columns"]), use_container_width=True, hide_index=True)
        with tab_accom:
            a_res = sql_execute_query("SELECT accommodation_id, name, destination_id, type, price_per_night_usd, rating, amenities FROM accommodations;")
            if "columns" in a_res:
                st.dataframe(pd.DataFrame(a_res["rows"], columns=a_res["columns"]), use_container_width=True, hide_index=True)
        with tab_cuis:
            c_res = sql_execute_query("SELECT cuisine_id, dish_name, destination_id, cuisine_type, avg_price_usd, spice_level, is_vegetarian, description FROM local_cuisines;")
            if "columns" in c_res:
                st.dataframe(pd.DataFrame(c_res["rows"], columns=c_res["columns"]), use_container_width=True, hide_index=True)
        with tab_book:
            b_res = sql_execute_query("SELECT booking_id, traveler_name, destination_id, accommodation_id, check_in, check_out, num_guests, total_cost_usd, status FROM bookings;")
            if "columns" in b_res:
                st.dataframe(pd.DataFrame(b_res["rows"], columns=b_res["columns"]), use_container_width=True, hide_index=True)

    # Prompt Suggestion Chips untuk Analisis Data
    st.markdown("**💡 Pertanyaan Analitik Populer:**")
    chip1, chip2, chip3, chip4 = st.columns(4)
    suggested_sql_query = None
    with chip1:
        if st.button("🏷️ Destinasi Termurah", key="chip1"):
            suggested_sql_query = "Destinasi mana yang memiliki rata-rata budget harian paling murah beserta ratingnya?"
    with chip2:
        if st.button("🥗 Kuliner Vegetarian", key="chip2"):
            suggested_sql_query = "Tampilkan daftar makanan vegetarian beserta kota asal dan harganya."
    with chip3:
        if st.button("💵 Pendapatan Booking", key="chip3"):
            suggested_sql_query = "Berapa total pendapatan (total_cost_usd) dari booking yang berstatus 'Confirmed'?"
    with chip4:
        if st.button("⭐ Akomodasi Terbaik", key="chip4"):
            suggested_sql_query = "Tampilkan akomodasi dengan rating tertinggi di atas 4.6 beserta tarif per malamnya."

    # Render History Chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else "👤"):
            st.markdown(format_llm_content(msg["content"]))

    agent_input = st.chat_input("Ajukan pertanyaan analitik destinasi, akomodasi, kuliner, atau booking...")
    active_agent_query = agent_input or suggested_sql_query

    if active_agent_query:
        st.session_state["messages"].append({"role": "user", "content": active_agent_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_agent_query)

        with st.chat_message("assistant", avatar="🧭"):
            # ── Definisi Tool ReAct SQL ──────────────────────────────────────
            @tool
            def execute_query(sql_query: str) -> str:
                """Execute a read-only SELECT SQL query on the travel SQLite database and return results in JSON format."""
                result = sql_execute_query(sql_query)
                return json.dumps(result, ensure_ascii=False)

            @tool
            def list_tables() -> List[str]:
                """Retrieve names of all tables in the travel database."""
                return sql_list_tables()

            @tool
            def describe_table(table_name: str) -> List[Dict[str, Any]]:
                """Look up the schema of a specific table in the database."""
                return sql_describe_table(table_name)

            db_tools = [execute_query, list_tables, describe_table]

            # ── Instruction dengan Skema Lengkap & Larangan Tool Berulang ─────
            agent_instruction = (
                "Anda adalah Senior Travel Data Analyst AI dari WanderWise AI. "
                "Tugas Anda adalah menganalisis database SQLite 'wanderwise_travel.db' untuk menjawab pertanyaan pengguna dengan akurat, komprehensif, dan berbasis data.\n\n"
                "SKEMA LENGKAP DATABASE WANDERWISE TRAVEL:\n"
                "1. destinations\n"
                "   - destination_id INTEGER PRIMARY KEY\n"
                "   - city VARCHAR(255) (contoh: 'Ubud', 'Kyoto', 'Hallstatt', 'Santorini', dll)\n"
                "   - country VARCHAR(100) (contoh: 'Indonesia', 'Japan', 'Austria', 'Greece', dll)\n"
                "   - region VARCHAR(100) (contoh: 'Southeast Asia', 'East Asia', 'Europe', dll)\n"
                "   - best_season VARCHAR(50) (contoh: 'Apr-Oct', 'Mar-May / Oct-Nov', dll)\n"
                "   - avg_daily_budget_usd DECIMAL(10, 2)\n"
                "   - rating DECIMAL(3, 1)\n"
                "   - highlights TEXT\n\n"
                "2. accommodations\n"
                "   - accommodation_id INTEGER PRIMARY KEY\n"
                "   - name VARCHAR(255)\n"
                "   - destination_id INTEGER (FOREIGN KEY -> destinations.destination_id)\n"
                "   - type VARCHAR(50) (contoh: 'Resort', 'Hostel', 'Traditional Inn', 'Hotel', 'Boutique Hotel', 'Luxury Villa')\n"
                "   - price_per_night_usd DECIMAL(10, 2)\n"
                "   - rating DECIMAL(3, 1)\n"
                "   - amenities TEXT\n\n"
                "3. local_cuisines\n"
                "   - cuisine_id INTEGER PRIMARY KEY\n"
                "   - dish_name VARCHAR(255)\n"
                "   - destination_id INTEGER (FOREIGN KEY -> destinations.destination_id)\n"
                "   - cuisine_type VARCHAR(100) (contoh: 'Indonesian', 'Japanese', 'Austrian', 'Greek', dll)\n"
                "   - avg_price_usd DECIMAL(10, 2)\n"
                "   - spice_level VARCHAR(20) (contoh: 'None', 'Mild', 'Medium', 'Hot')\n"
                "   - is_vegetarian BOOLEAN (1 = True/Vegetarian, 0 = False)\n"
                "   - description TEXT\n\n"
                "4. bookings\n"
                "   - booking_id INTEGER PRIMARY KEY\n"
                "   - traveler_name VARCHAR(255)\n"
                "   - destination_id INTEGER (FOREIGN KEY -> destinations.destination_id)\n"
                "   - accommodation_id INTEGER (FOREIGN KEY -> accommodations.accommodation_id)\n"
                "   - check_in DATE\n"
                "   - check_out DATE\n"
                "   - num_guests INTEGER\n"
                "   - total_cost_usd DECIMAL(10, 2)\n"
                "   - status VARCHAR(50) ('Confirmed', 'Pending', 'Cancelled')\n\n"
                "ATURAN OPERASIONAL PENTING (MINIMALISIR TOOL CALLS):\n"
                "1. JANGAN memanggil tool 'list_tables' atau 'describe_table' karena skema di atas sudah LENGKAP dan PASTI BENAR.\n"
                "2. LANGSUNG buat satu query SQL SELECT yang efisien dan akurat, lalu eksekusi menggunakan tool 'execute_query'.\n"
                "3. Hanya gunakan perintah SELECT (read-only query).\n"
                "4. Setelah menerima hasil data, berikan jawaban analisis yang jelas, informatif, dan terstruktur dalam Bahasa Indonesia yang profesional (gunakan tabel Markdown jika menyajikan banyak data)."
            )

            with st.status("🔍 WanderWise Agent sedang menganalisis data...", expanded=True) as status_box:
                try:
                    agent_executor = create_agent(
                        model=llm,
                        tools=db_tools,
                        system_prompt=agent_instruction
                    )

                    st.write("⚙️ Menganalisis parameter pertanyaan dan menyusun query...")
                    agent_result = agent_executor.invoke(
                        {"messages": [HumanMessage(content=active_agent_query)]}
                    )

                    all_agent_messages = agent_result.get("messages", [])
                    for msg in all_agent_messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                st.write(f"⚡ **Eksekusi Tool:** `{tc['name']}`")
                                if "sql_query" in tc.get("args", {}):
                                    st.code(tc["args"]["sql_query"], language="sql")
                        elif getattr(msg, "type", "") == "tool":
                            st.write("📊 Data berhasil diekstrak dari database SQLite.")

                    status_box.update(label="Analisis Database Selesai", state="complete", expanded=False)

                    final_response_text = format_llm_content(all_agent_messages[-1].content)
                    st.markdown(final_response_text)
                    st.session_state["messages"].append({"role": "assistant", "content": final_response_text})

                except Exception as agent_err:
                    err_str = str(agent_err)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        status_box.update(label="Kuota API Penuh (Rate Limit 429)", state="error")
                        err_text = (
                            "⚠️ **Gemini API Rate Limit Tercapai (Error 429):**\n"
                            "Batas permintaan per menit dari API Key gratis Anda sedang penuh. "
                            "Harap tunggu sekitar 10-20 detik sebelum mencoba lagi, atau ganti model ke `gemini-2.0-flash` di sidebar."
                        )
                    else:
                        status_box.update(label="Terjadi Kendala Eksekusi", state="error")
                        err_text = f"Terjadi kendala pada AI Agent: {agent_err}"

                    st.error(err_text)
                    st.session_state["messages"].append({"role": "assistant", "content": err_text})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
col_ft1, col_ft2 = st.columns([3, 1])
with col_ft1:
    st.caption("🌏 **WanderWise AI**  Smart Travel & Culinary Intelligence Platform")
with col_ft2:
    st.caption("Powered by Google Gemini & Groq")
