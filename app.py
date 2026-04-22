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
st.set_page_config(page_title="AI Intelligence Studio", layout="wide")
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
    font-size:50px;
    font-weight:900;
    background: linear-gradient(90deg,#00f5ff,#3a86ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.kpi {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:14px;
    text-align:center;
}
.banner {
    height:300px;border-radius:18px;overflow:hidden;position:relative;margin-bottom:20px;
}
.banner img {
    width:100%;height:100%;object-fit:cover;
    filter:brightness(30%);
    transition:0.6s;
}
.banner:hover img {transform:scale(1.08);}
.banner-text {
    position:absolute;bottom:30px;left:40px;
}
.card {
    position:relative;border-radius:12px;overflow:hidden;transition:0.3s;
}
.card:hover {transform:scale(1.08);}
.card img {width:100%;}
.overlay {
    position:absolute;bottom:0;width:100%;padding:10px;
    background:linear-gradient(to top, rgba(0,0,0,0.9), transparent);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="header">AI Intelligence Studio</div>', unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    anime = pd.read_csv(BASE / "anime.csv")
    social = pd.read_csv(BASE / "03_Clustering_Marketing.csv")

    anime['rating'] = pd.to_numeric(anime['rating'], errors='coerce')
    anime['content'] = anime['genre'].fillna('') + " " + anime['type'].fillna('')

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
# 🎬 ANIME SECTION
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
    i = idx[name]
    scores = sorted(list(enumerate(sim[i])), key=lambda x:x[1], reverse=True)[1:n+1]
    return anime.iloc[[x[0] for x in scores]]

def poster(title):
    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        return requests.get(url, timeout=5).json()['data'][0]['images']['jpg']['image_url']
    except:
        return "https://via.placeholder.com/300x420"

# ===== INPUT UI =====
with st.container():
    c1, c2, c3 = st.columns([4,1,1])

    sel = c1.selectbox("Select Anime", anime['name'].head(300), key="anime")
    n = c2.selectbox("Top N", [5,10], key="topn")
    go = c3.button("🚀 Recommend", key="btn")

# ===== STATE =====
if "result" not in st.session_state:
    st.session_state.result = None

if go:
    st.session_state.result = recommend(sel, n)

# ===== SEPARATOR =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ===== OUTPUT =====
if st.session_state.result is not None:

    st.markdown("### 🎬 Recommended Anime")

    cols = st.columns(5)

    for i,(_,r) in enumerate(st.session_state.result.iterrows()):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="card">
                <img src="{poster(r['name'])}">
                <div class="overlay">
                    <b>{r['name']}</b><br>⭐ {round(r['rating'],2)}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# 👥 SOCIAL CLUSTERING
# =====================================================
st.markdown("""
<div class="banner">
<img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71">
<div class="banner-text">
<h1>Social Clustering</h1>
</div>
</div>
""", unsafe_allow_html=True)

X = social.select_dtypes(include=[np.number])
X = SimpleImputer(strategy='median').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)

labels = KMeans(n_clusters=3, n_init=20, random_state=42).fit_predict(X_scaled)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

df_cluster = social.copy()
df_cluster["Cluster"] = labels

st.success("AI Clustering Completed")

selected_cluster = st.selectbox("Select Segment", sorted(set(labels)), key="cluster")

mask = labels == selected_cluster

fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str), template="plotly_dark")
fig.add_scatter(x=X_pca[mask,0], y=X_pca[mask,1], mode='markers')

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_cluster[mask].head(20))

# ================= FOOTER =================
st.markdown("This app can make mistakes. Check important information before use.")
