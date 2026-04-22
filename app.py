# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# ================= CONFIG =================
st.set_page_config(
    page_title="AI Intelligence Studio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).resolve().parent

# ================= STYLE =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color:white;
}
.header {
    text-align:center;
    font-size:48px;
    font-weight:900;
    background: linear-gradient(90deg,#00f5ff,#3a86ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.kpi {
    background: rgba(255,255,255,0.06);
    padding:18px;
    border-radius:14px;
    text-align:center;
}
.banner {
    height:260px;border-radius:18px;overflow:hidden;position:relative;margin-bottom:20px;
}
.banner img {
    width:100%;height:100%;object-fit:cover;
    filter:brightness(35%);
}
.banner-text {
    position:absolute;bottom:20px;left:30px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="header">AI Intelligence Studio</div>', unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    try:
        anime = pd.read_csv(BASE / "anime.csv")
        social = pd.read_csv(BASE / "03_Clustering_Marketing.csv")
    except:
        st.error("❌ CSV files not found. Keep them in same folder as app.py")
        st.stop()

    # Clean anime
    anime['rating'] = pd.to_numeric(anime.get('rating'), errors='coerce')
    anime['name'] = anime.get('name', "").astype(str)
    anime['content'] = anime.get('genre', "").astype(str) + " " + anime.get('type', "").astype(str)
    anime = anime.dropna(subset=['name']).reset_index(drop=True)

    # Clean social
    social = social.fillna(social.median(numeric_only=True))

    return anime, social

anime, social = load_data()

# ================= KPI =================
k1,k2,k3,k4 = st.columns(4)
k1.markdown(f'<div class="kpi">Anime<br><h2>{len(anime)}</h2></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi">Avg Rating<br><h2>{round(anime["rating"].mean(),2)}</h2></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi">Users<br><h2>{len(social)}</h2></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi">Features<br><h2>{social.shape[1]}</h2></div>', unsafe_allow_html=True)

# =====================================================
# 🎬 ANIME RECOMMENDER
# =====================================================
st.markdown("""
<div class="banner">
<img src="https://images.unsplash.com/photo-1511512578047-dfb367046420">
<div class="banner-text"><h1>Anime Intelligence</h1></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    mat = tfidf.fit_transform(df['content'])
    sim = cosine_similarity(mat)
    idx = pd.Series(df.index, index=df['name']).drop_duplicates()
    return sim, idx

sim, idx = build_model(anime)

def recommend(name, n):
    if name not in idx:
        return pd.DataFrame()

    i = idx[name]
    scores = sorted(
        list(enumerate(sim[i])),
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]

    return anime.iloc[[x[0] for x in scores]]

def poster(title):
    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        res = requests.get(url, timeout=5).json()
        return res['data'][0]['images']['jpg']['image_url']
    except:
        return "https://via.placeholder.com/300x420"

c1,c2,c3 = st.columns([4,1,1])
sel = c1.selectbox("Select Anime", anime['name'].head(300))
n = c2.selectbox("Top N",[5,10])
go = c3.button("🚀 Recommend")

if go:
    res = recommend(sel, n)

    if res.empty:
        st.warning("No recommendations found")
    else:
        cols = st.columns(5)
        for i, (_, r) in enumerate(res.iterrows()):
            with cols[i % 5]:
                st.image(poster(r['name']), use_container_width=True)
                st.markdown(f"**{r['name']}**")
                st.caption(f"⭐ {round(r['rating'],2)}")

# =====================================================
# 👥 SOCIAL CLUSTERING
# =====================================================
st.markdown("""
<div class="banner">
<img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71">
<div class="banner-text"><h1>Social Clustering</h1></div>
</div>
""", unsafe_allow_html=True)

X = social.select_dtypes(include=[np.number])
X = SimpleImputer(strategy='median').fit_transform(X)

X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

model = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = model.fit_predict(X_pca)

st.success("✅ Clustering Completed")

df_cluster = social.copy()
df_cluster["Cluster"] = labels

selected_cluster = st.selectbox("Select Segment", sorted(set(labels)))
mask = labels == selected_cluster

fig = px.scatter(
    x=X_pca[:,0],
    y=X_pca[:,1],
    color=labels.astype(str),
    template="plotly_dark"
)

fig.add_scatter(
    x=X_pca[mask,0],
    y=X_pca[mask,1],
    mode='markers',
    marker=dict(size=10)
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_cluster[mask].head(20))

# ================= FOOTER =================
st.markdown("⚠️ This app is for demo purposes. Validate results before use.")
