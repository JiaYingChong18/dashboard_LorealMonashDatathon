import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from collections import Counter
import re
import json
from groq import Groq
import nltk
from nltk.corpus import stopwords
import ast
import emoji
import os
import gdown
import kagglehub

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Put this at the start of your Streamlit file
st.set_page_config(
    page_title="L'Oréal Comment Quality Dashboard",
    layout="wide",  
    initial_sidebar_state="expanded"
)
# ---------------- Load CSV ----------------
@st.cache_data
def load_data():

    # Download latest version of your Kaggle dataset
    dataset_path = kagglehub.dataset_download("jiayinggggggg18/first-data")
    
    # Assuming your CSV is inside the zip file
    # Replace 'your_file.csv' with the actual filename in the dataset
    df = pd.read_csv(f"{dataset_path}/finalise_cleaned_text2100k.csv")

    st.write("Columns in dataframe:", df.columns.tolist())
    # Relevance
    if "is_relevant" in df.columns:
        df["is_relevant"] = df["is_relevant"].astype(bool)
    else:
        df["is_relevant"] = False

    # Dates
    if "comment_publishedAt" in df.columns:
        df["comment_publishedAt"] = pd.to_datetime(df["comment_publishedAt"], errors="coerce")

    return df


# ---------------- Date Filter ----------------
st.sidebar.header("⏳ Filter by Date")
df = load_data()

if "commentPublishedAt" in df.columns:
    df["commentPublishedAt"] = pd.to_datetime(df["commentPublishedAt"])
    min_date = df["commentPublishedAt"].min()
    max_date = df["commentPublishedAt"].max()
    start_date, end_date = st.sidebar.slider(
        "Select date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD"
    )
    mask = (df["commentPublishedAt"] >= start_date) & (df["commentPublishedAt"] <= end_date)
    df = df.loc[mask]
    st.sidebar.write("Start date:", start_date.date())
    st.sidebar.write("End date:", end_date.date())


# ---------------- Functions Calling ---------------
def get_kpi_summary(df):
    return {
        "total_comments": int(df["commentId"].nunique()),
        "pct_relevant": round(100*df["is_relevant"].mean(), 2),
        "avg_sentiment": round(df.loc[df["is_relevant"], "sentiment_score"].mean(), 2)
    }

def get_top_videos(df, n=3):
    df_quality = df[(df["is_relevant"]) & (df["comment_is_spam"] == 0)]
    video_stats = df_quality.groupby("videoId").agg(
        avg_quality_score=("quality_score", "mean"),
        total_comments=("commentId", "count")
    ).reset_index()
    top_videos = video_stats.sort_values("avg_quality_score", ascending=False).head(n)
    return top_videos.to_dict(orient="records")

def get_top_categories(df, n=3):
    return df[df["is_relevant"] & (df["comment_is_spam"] == 0)]["label"].value_counts().head(n).to_dict()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

def generate_response_groq(user_input, user_role):
    """
    Generates a response from Groq GPT-OSS-20B
    using a structured prompt.
    """
    dashboard_state = {
        "kpis": get_kpi_summary(df),
        "top_videos": get_top_videos(df, n=5),
        "top_categories": get_top_categories(df, n=5)
    }
   
    prompt = f"""
    You are a professional business insights assistant for L’Oréal.
You may use the dashboard information that is relevant to answer the user’s question.
The dashboard includes: relevance filtering, spam detection, comment categorization, sentiment analysis, quality scoring, and trend detection.
Your job is to:
    1. Explain dashboard results in plain, simple language anyone can understand.
    2. Provide actionable, role-specific suggestions tailored to the user’s role.
    - Marketing Teams → budget allocation, ROI.
    - Product Development / R&D → Product feedback, improvements, customer needs.
    - Trend Forecasting / Insights Team → New beauty trends, emerging consumer behaviors.
    - PR & Corporate Communications → Negative spikes, complaints, brand risk.
    - Customer Care → Urgent issues, trust-building opportunities.
    - Top Management / Strategy → Clear summaries, revenue/brand impact.
    - Influencers & Brand Ambassadors → Trend-driven content ideas, engagement boosters.
    3. Avoid jargon unless necessary. If you use a technical term (e.g., “sentiment score”), define it simply.
    4. Always structure your response as: Explanation → Insight → Recommendation.
    5. Keep answers clear, concise, and ideally under 200 words.
    6. Ensure the response is complete.
    User role: {user_role}

    Dashboard summary (JSON): {json.dumps(dashboard_state, indent=2)}
    User question: {user_input}

    Respond in the following structure:
    - 📊 Explanation
    - 🔍 Insights
    - ✅ Recommendations
    """

    # Call Groq
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=700
    )

    response_text = completion.choices[0].message.content
    return response_text

# Example usage in Streamlit sidebar
st.sidebar.header("💬 L'Oréal Insights Chatbot")
user_role = st.sidebar.selectbox(
    "Select your role:",
    ["Marketing Teams", "Product Development/R&D", "Trend Forecasting / Insights Team", "PR & Corporate Communications", "Customer Care", "Top Management / Strategy", "Influencers & Brand Ambassadors"],
    key="user_role_selectbox"
)
user_question = st.sidebar.text_area("Enter your question:")

if user_question:
    with st.spinner("Generating response ..."):
        answer = generate_response_groq(user_question, user_role)
        st.sidebar.markdown(f"**Chatbot Response:**\n\n{answer}")


# ---------------- KPI Cards ----------------
st.title("L'Oréal Comment Quality Dashboard")


# --- KPI calculations ---
total_comments = df["commentId"].nunique() if "commentId" in df.columns else len(df)
pct_relevant = 100 * df["is_relevant"].mean()
avg_sentiment = df.loc[df["is_relevant"], "sentiment_score"].mean() if df["is_relevant"].any() else 0
pct_spam = 100 * df["comment_is_spam"].mean()
avg_quality = df["quality_score"].mean() if "quality_score" in df.columns else 0.0

# --- KPI display ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Comments", f"{total_comments:,}")
k2.metric("% Relevant to L'Oréal", f"{pct_relevant:.1f}%")
k3.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")
k4.metric("% Spam Detected", f"{pct_spam:.1f}%")
k5.metric("Avg Quality Score", f"{avg_quality:.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Relevance & Sentiment Breakdown",
    "💄 Category Insights",
    "🚨 Spam Detection",
    "⭐ Quality Score"
])

with tab1:
    st.header("📊 Relevance & Sentiment Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart: Relevant vs Non-relevant
        relevance_counts = df["is_relevant"].value_counts().rename({True: "Relevant", False: "Not Relevant"})
        fig_rel = px.pie(
            names=relevance_counts.index,
            values=relevance_counts.values,
            title="Relevant vs Non-relevant Comments"
        )
        st.plotly_chart(fig_rel)

    with col2:
        # Bar chart: Sentiment distribution (Relevant only, not spam)
        sent_counts = df[(df["is_relevant"]) & (df["comment_is_spam"] == 0)]["sentiment_y"].value_counts()
        fig_sent = px.bar(
            x=sent_counts.index,
            y=sent_counts.values,
            labels={"x": "Sentiment", "y": "Count"},
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent)

    # Time-series: Sentiment trend
    if "commentPublishedAt" in df.columns:
        df["commentPublishedAt"] = pd.to_datetime(df["commentPublishedAt"], errors="coerce", utc=True)
        trend = (
            df[(df["is_relevant"]) & (df["comment_is_spam"] == 0)]
            .groupby(df["commentPublishedAt"].dt.to_period("M"))["sentiment_score"]
            .mean()
            .reset_index()
        )
        trend["commentPublishedAt"] = trend["commentPublishedAt"].dt.to_timestamp()
        chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(x="commentPublishedAt:T", y="sentiment_score:Q")
            .properties(title="Avg Sentiment Over Time")
        )
        st.altair_chart(chart, use_container_width=True)

with tab2:
    st.header("💄 Category Insights")

    if "label" in df.columns:
        df_cat = df[(df["is_relevant"]) & (df["comment_is_spam"] == 0)]

        # ---- 1. Sentiment Distribution by Category ----
        cat_sent = (
            df_cat.groupby(["label", "sentiment_y"])
            .size()
            .reset_index(name="count")
        )

        total_comments = cat_sent["count"].sum()
        cat_sent["percent"] = cat_sent["count"] / total_comments * 100

        fig_cat = px.bar(
            cat_sent,
            x="label",
            y="percent",
            color="sentiment_y",
            barmode="stack",
            labels={
                "percent": "Percentage of All Comments (%)",
                "label": "Category",
                "sentiment_y": "Sentiment"
            },
            title="📊 Sentiment Distribution by Category",
            color_discrete_map={
                "positive": "#4CAF50",  
                "negative": "#F44336",  
                "neutral": "#2196F3",  
            },
        )

        # ---- 2. Engagement Comparison (Avg Likes per Comment) ----
        if "comment_likeCount" in df_cat.columns:
            cat_eng = (
                df_cat.groupby("label")["comment_likeCount"]
                .mean()
                .reset_index()
            )

            fig_eng = px.bar(
                cat_eng,
                x="label",
                y="comment_likeCount",
                color="label",
                text_auto=".1f",
                color_discrete_sequence=px.colors.sequential.Viridis,
                labels={
                    "comment_likeCount": "Avg Likes per Comment",
                    "label": "Category"
                },
                title="👍 Engagement Comparison by Category"
            )

        # ---- 3. Quality Score Analysis ----
        if "quality_score" in df_cat.columns:
            label_stats = df_cat.groupby("label").agg(
                avg_quality_score=("quality_score", "mean"),
                total_comments=("commentId", "count"),
                avg_sentiment=("sentiment_score", "mean")
            ).reset_index()

            label_stats = label_stats.sort_values("avg_quality_score", ascending=False)

            fig_quality = px.bar(
                label_stats,
                x="label",
                y="avg_quality_score",
                color="avg_quality_score",
                title="⭐ Average Quality Score by Category",
                text="avg_quality_score"
            )
            fig_quality.update_traces(
                texttemplate='%{text:.2f}',
                textposition="outside"
            )

        # ---- Tabs for switching between charts ----
        cat_tab1, cat_tab2, cat_tab3 = st.tabs([
            "📊 Sentiment Distribution",
            "👍 Engagement",
            "⭐ Quality Score"
        ])

        with cat_tab1:
            st.plotly_chart(fig_cat, use_container_width=True)

        with cat_tab2:
            st.plotly_chart(fig_eng, use_container_width=True)

        with cat_tab3:
            st.plotly_chart(fig_quality, use_container_width=True)


        selected_label = st.selectbox("Choose a label:", label_stats["label"].unique())

        top_comments = (
            df_cat[df_cat["label"] == selected_label]
            .sort_values("quality_score", ascending=False)
            .head(5)
        )

        display_comments = top_comments[["comment_clean_text", "quality_score", "sentiment_y", "comment_likeCount"]].rename(
            columns={
                "comment_clean_text": "Comment",
                "quality_score": "Quality Score",
                "sentiment_y": "Sentiment",
                "comment_likeCount": "Engagement (Likes)"
            }
        )

        st.dataframe(
            display_comments.style.background_gradient(cmap="YlGnBu", subset=["Quality Score"])
                            .format({"Quality Score": "{:.2f}", "Engagement (Likes)": "{:d}"})
        )

with tab3:
    st.header("🚨 Spam Detection")

    # --- Spam % per Category ---
    if "label" in df.columns:
        spam_per_cat = (
            df.groupby("label")["comment_is_spam"]
            .mean()
            .reset_index()
        )
        spam_per_cat["comment_is_spam"] *= 100  
        fig_spam_cat = px.bar(
            spam_per_cat,
            x="label",
            y="comment_is_spam",
            text_auto=".1f",
            color="label",
            color_discrete_sequence=px.colors.sequential.Blues,
            labels={"comment_is_spam": "% Spam Comments", "label": "Category"},
            title="📌 Spam % per Category"
        )
        fig_spam_cat.update_traces(textposition="outside")
        st.plotly_chart(fig_spam_cat, use_container_width=True)

    # --- Keywords by Category ---

    st.header("🔥 Most Common Keywords by Category")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    emoji_noise = {
        "red_heart", "blue_heart", "purple_heart", "yellow_heart", "green_heart",
        "smiling_face_with_smiling_eyes", "smiling_face_with_heart_eyes",
        "rolling_on_the_floor_laughing", "face_with_tears_of_joy",
        "crying_face", "clown_face", "relieved_face", "smiling_face_with_sunglasses",
        "ok_hand", "thumbs_up", "thumbs_down","face_vomiting","loudly_crying_face","pink_heart",
        "smilling_face_with_smiling_eyes","sparkles","hundred_points","pray","see_no_evil",
        "party_popper","two_hearts","collision","smiling_face_with_heart","face_blowing_a_kiss", "grinning_face_with_sweat", "much", "always", 
        "really", "smiling_face_with_hearts", "face_with_open_mouth", "face_with_hand_over_mouth",
        "like", "get", "good", "great", "well", "know", "one", "go", "even", "time", "day"
    }
    extra_noise = {"like", "thank", "thanks", "love", "wow", "wowwww"}
    stop_words.update(emoji_noise)
    stop_words.update(extra_noise)

    def tokenize_and_clean(text: str):
        """Tokenize, lowercase, remove stopwords + emojis"""
        tokens = re.findall(r"\b\w+\b", text)  
        cleaned = [t for t in tokens if t not in stop_words and len(t) > 2]
        return cleaned

    results = {}
    all_rows = []

    spam_df = df[(df["is_relevant"]) & (df["comment_is_spam"] == 0)]
    for category, group in spam_df.groupby("label"):
        tokens = []

        for text in group["comment_clean_text"]:
            tokens.extend(tokenize_and_clean(text))

        counter = Counter(tokens)
        top_keywords = counter.most_common(5)
        results[category] = top_keywords

        for kw, cnt in top_keywords:
            all_rows.append({"Category": category, "Keyword": kw, "Count": cnt})

    df_keywords = pd.DataFrame(all_rows)

    # --------------------------
    # Streamlit Filter + View Mode
    # --------------------------
    if not df_keywords.empty:
        categories = df_keywords["Category"].unique().tolist()
        categories_with_all = ["All"] + categories  

        if "selected_category" not in st.session_state:
            st.session_state.selected_category = categories[0]

        selected_category = st.selectbox(
            "Select a category",
            categories_with_all,
            index=categories_with_all.index(st.session_state.selected_category),
            key="selected_category"
        )

        st.subheader(f"🔥 Most Common Keywords in {selected_category}")
        if selected_category == "All":
            st.table(df_keywords.groupby("Category")
                    .apply(lambda x: x.nlargest(5, "Count"))
                    .reset_index(drop=True))
        else:
            keywords = results.get(selected_category, [])
            if keywords:
                st.table(pd.DataFrame(keywords, columns=["Keyword", "Count"]))
            else:
                st.write("No keywords found for this category.")

        # --------------------------
        # View mode for bar chart
        # --------------------------
        view_mode = st.radio("Bar Chart View Mode", ["Per Category", "Combined"], horizontal=True)

        if view_mode == "Per Category" and selected_category != "All":
            st.subheader(f"📊 Keywords Chart ({selected_category})")
            chart_df = df_keywords[df_keywords["Category"] == selected_category]
            st.bar_chart(chart_df.set_index("Keyword")["Count"])

        elif view_mode == "Combined":
            st.subheader("📊 Combined Common Keywords by Category")
            pivot_df = df_keywords.pivot_table(
                index="Keyword", columns="Category", values="Count", fill_value=0
            )
            st.bar_chart(pivot_df)



with tab4:
    # ---------------- Section 5: Comment Quality Score ----------------
    st.header("⭐ Comment Quality Score")
    df_quality = df[(df["comment_is_spam"] == 0) & (df["is_relevant"] == True)]

    # --- Histogram of Quality Scores ---
    fig_hist = px.histogram(
        df_quality,
        x="quality_score",
        nbins=20,
        color_discrete_sequence=["#2E86AB"],
        title="Distribution of Quality Scores"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 🌟 Top 10 High-Quality Positive Comments
    st.subheader("🌟 Top 10 High-Quality Positive Comments")

    top10_pos = (
        df_quality[df_quality["sentiment_y"] == "positive"]
        .sort_values("quality_score", ascending=False)
        .head(10)
    )

    display_pos = top10_pos[["comment_clean_text", "comment_likeCount", "quality_score"]].rename(
        columns={
            "comment_clean_text": "Comment",
            "comment_likeCount": "Engagement (Likes)",
            "quality_score": "Quality Score"
        }
    )

    st.dataframe(
        display_pos.style.background_gradient(cmap="Greens", subset=["Quality Score"])
                        .format({"Quality Score": "{:.2f}"})
    )

    # 📉 Top 10 High-Quality Negative Comments
    st.subheader("📉 Top 10 High-Quality Negative Comments")

    top10_neg = (
        df_quality[df_quality["sentiment_y"] == "negative"]
        .sort_values("quality_score", ascending=False)
        .head(10)
    )

    display_neg = top10_neg[["comment_clean_text", "comment_likeCount", "quality_score"]].rename(
        columns={
            "comment_clean_text": "Comment",
            "comment_likeCount": "Engagement (Likes)",
            "quality_score": "Quality Score"
        }
    )

    st.dataframe(
        display_neg.style.background_gradient(cmap="Oranges", subset=["Quality Score"])
                        .format({"Quality Score": "{:.2f}"})
    )

    # ---Top 10 Best Videos, Best Content Creator---
    df_filtered = df[(df["is_relevant"] == True) & (df["comment_is_spam"] == 0)]
    # ---- Aggregate per video ----
    video_stats = df_filtered.groupby("videoId").agg(
        avg_quality_score=("quality_score", "mean"),
        engagement=("commentId", "count"),   
        video_likeCount=("video_likeCount", "max"),
        video_viewCount=("video_viewCount", "max"),
        video_channelId=("video_channelId", "first"),
        contentDuration=("contentDuration", "first"),
        video_title=("video_clean_text", "first")  
    ).reset_index()

    # ---- Ranking logic (fixed) ----
    video_stats["score"] = (
        video_stats["engagement"].rank(ascending=False) +
        video_stats["avg_quality_score"].rank(ascending=False) +
        video_stats["video_likeCount"].rank(ascending=False) +
        video_stats["video_viewCount"].rank(ascending=False)
    )

    # ---- Top 10 videos ----
    top_videos = video_stats.sort_values("score", ascending=True).head(10)
    top_videos["video_likeCount"] = top_videos["video_likeCount"].astype(int)
    top_videos["video_viewCount"] = top_videos["video_viewCount"].astype(int)

    # ---- Display ----
    st.subheader("🎬 Top 10 Videos by Quality & Engagement")
    st.dataframe(
        top_videos[["video_title", "avg_quality_score", "engagement", "video_likeCount", "video_viewCount", "score"]]
        .rename(columns={
            "video_title": "Video Title",
            "avg_quality_score": "Avg Quality Score",
            "engagement": "Engagement (#Comments)",
            "video_likeCount": "Likes",
            "video_viewCount": "Views",
            "score": "Ranking Score"
        })
        .style.background_gradient(cmap="YlGnBu", subset=[ "Ranking Score"])
        .format({"Ranking Score": "{:.2f}","Avg Quality Score":"{:.2f}",
            "Likes": "{:d}",
            "Views": "{:d}"})
    )

    # 🌍 Mapping language codes → representative countries/regions
    lang_to_country = {
        "es": "Spain / Latin America",
        "fr": "France",
        "id": "Indonesia",
        "de": "Germany",
        "pt": "Portugal / Brazil",
        "ru": "Russia",
        "zh": "China",
        "ja": "Japan",
        "ar": "Middle East (Arabic-speaking)",
        "hi": "India",
        "bn": "Bangladesh / India",
        "tr": "Turkey",
        "nl": "Netherlands",
        "it": "Italy",
        "ko": "South Korea",
        "th": "Thailand",
        "pl": "Poland",
        "vi": "Vietnam",
        "fa": "Iran",
        "uk": "Ukraine",
        "ms": "Malaysia",
        "ta": "India (Tamil Nadu) / Sri Lanka",
        "ur": "Pakistan",
        "el": "Greece",
        "he": "Israel",
        "sv": "Sweden",
        "no": "Norway",
        "fi": "Finland",
        "cs": "Czech Republic",
        "ro": "Romania",
        "hu": "Hungary",
        "bg": "Bulgaria",
        "sr": "Serbia",
        "hr": "Croatia",
        "sk": "Slovakia",
        "da": "Denmark",
        "az": "Azerbaijan",
        "uz": "Uzbekistan",
        "ka": "Georgia",
        "mn": "Mongolia",
        "kk": "Kazakhstan",
        "pa": "India / Pakistan (Punjab)",
        "si": "Sri Lanka (Sinhala)",
        "my": "Myanmar",
        "km": "Cambodia",
    }

    # 🔤 Mapping language codes → full language names
    lang_to_name = {
        "es": "Spanish",
        "fr": "French",
        "id": "Indonesian",
        "de": "German",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ar": "Arabic",
        "hi": "Hindi",
        "bn": "Bengali",
        "tr": "Turkish",
        "nl": "Dutch",
        "it": "Italian",
        "ko": "Korean",
        "th": "Thai",
        "pl": "Polish",
        "vi": "Vietnamese",
        "fa": "Persian (Farsi)",
        "uk": "Ukrainian",
        "ms": "Malay",
        "ta": "Tamil",
        "ur": "Urdu",
        "el": "Greek",
        "he": "Hebrew",
        "sv": "Swedish",
        "no": "Norwegian",
        "fi": "Finnish",
        "cs": "Czech",
        "ro": "Romanian",
        "hu": "Hungarian",
        "bg": "Bulgarian",
        "sr": "Serbian",
        "hr": "Croatian",
        "sk": "Slovak",
        "da": "Danish",
        "az": "Azerbaijani",
        "uz": "Uzbek",
        "ka": "Georgian",
        "mn": "Mongolian",
        "kk": "Kazakh",
        "pa": "Punjabi",
        "si": "Sinhala",
        "my": "Burmese",
        "km": "Khmer",
    }

    # ---- Filter out English ----
    foreign_langs = df[df["lang_prefix_comment"] != "en"]

    # ---- Count language frequency ----
    lang_counts = (
        foreign_langs["lang_prefix_comment"]
        .value_counts()
        .reset_index()
    )
    lang_counts.columns = ["lang_code", "count"]

    # ---- Map to country & full language name ----
    lang_counts["country"] = lang_counts["lang_code"].map(lang_to_country).fillna("Other / Unknown")
    lang_counts["language"] = lang_counts["lang_code"].map(lang_to_name).fillna("Other / Unknown")

    # ---- Get top 5 ----
    top5_langs = lang_counts.head(5)

    # ---- Display Table ----
    st.subheader("🌐 Top 5 Foreign Languages in Comments")
    st.dataframe(
        top5_langs.rename(columns={
            "language": "Language",
            "country": "Representative Country",
            "count": "Comment Count"
        })[["Language", "Representative Country", "Comment Count"]]
    )

    # ---- Bubble Chart ----
    fig = px.scatter(
        top5_langs,
        x="language",
        y="count",
        size="count",           # bubble size = number of comments
        color="language",
        hover_name="country",
        size_max=100,
        title="Top 5 Foreign Languages by Country (Comment Count)"
    )

    st.plotly_chart(fig, use_container_width=True)



