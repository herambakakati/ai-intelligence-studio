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

# ================= STYLE (UNCHANGED) =================
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
        res = requests.get(url, timeout=5).json()
        return res['data'][0]['images']['jpg']['image_url']
    except:
        return "https://via.placeholder.com/300x420"

c1,c2,c3 = st.columns([4,1,1])
sel = c1.selectbox("Select Anime", anime['name'].head(300))
n = c2.selectbox("Top N",[5,10])
go = c3.button("🚀 Recommend")

if go:
    res = recommend(sel,n)
    cols = st.columns(5)
    for i,(_,r) in enumerate(res.iterrows()):
        with cols[i%5]:
            st.markdown(f"""
            <div class="card">
                <img src="{poster(r['name'])}">
                <div class="overlay">
                    <b>{r['name']}</b><br>⭐ {round(r['rating'],2)}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# 👥 SOCIAL CLUSTERING (FINAL + AI INSIGHT)
# =====================================================
st.markdown("""
<div class="banner">
<img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71">
<div class="banner-text">
<h1>Social Clustering</h1>
</div>
</div>
""", unsafe_allow_html=True)

# ===== AUTO CLUSTER =====
X = social.select_dtypes(include=[np.number])
X = SimpleImputer(strategy='median').fit_transform(X)

var = np.var(X, axis=0)
X = X[:, var > 0]

X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

model = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = model.fit_predict(X_pca)

st.success("AI Clustering Completed")

# ===== PERSONA =====
df_cluster = social.copy()
df_cluster["Cluster"] = labels

numeric_df = df_cluster.select_dtypes(include=[np.number])
summary = numeric_df.groupby(df_cluster["Cluster"]).mean()

global_mean = numeric_df.mean().mean()

persona = {}
for c in sorted(summary.index):
    cm = summary.loc[c].mean()

    if cm > global_mean:
        persona[c] = f"Segment {c} • High Engagement"
    elif cm > global_mean * 0.7:
        persona[c] = f"Segment {c} • Active Users"
    else:
        persona[c] = f"Segment {c} • Low Activity"

# ===== SEARCH =====
st.markdown("### 🧠 Search User Segment")

selected_cluster = st.selectbox(
    "Select Segment",
    list(persona.keys()),
    format_func=lambda x: persona[x]
)

mask = labels == selected_cluster

# ===== VISUAL =====
fig = px.scatter(
    x=X_pca[:,0],
    y=X_pca[:,1],
    color=labels.astype(str),
    opacity=0.2,
    template="plotly_dark"
)

fig.add_scatter(
    x=X_pca[mask,0],
    y=X_pca[mask,1],
    mode='markers',
    marker=dict(size=10)
)

st.plotly_chart(fig, use_container_width=True)

# ===== AI INSIGHT =====
cluster_data = df_cluster[mask]

numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
cluster_mean = cluster_data[numeric_cols].mean()
overall_mean = df_cluster[numeric_cols].mean()

diff = (cluster_mean - overall_mean).sort_values(ascending=False)
top_features = diff.head(3).index.tolist()

if "High Engagement" in persona[selected_cluster]:
    insight = f"This segment shows strong activity in {', '.join(top_features)}. These users are highly engaged."
elif "Active Users" in persona[selected_cluster]:
    insight = f"This group is moderately active, especially in {', '.join(top_features)}."
else:
    insight = f"This segment has lower activity, particularly in {', '.join(top_features)}."

st.markdown(f"""
<div style="background:#020617;padding:18px;border-radius:12px;margin-top:10px;">
<h4>🧠 AI Insight</h4>
<p>{insight}</p>
</div>
""", unsafe_allow_html=True)

# ===== TABLE =====
st.dataframe(df_cluster[mask].head(20))

# ================= FOOTER =================
st.markdown("This app can make mistakes. Check important information before use.")
