from flask import Flask, request, redirect
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

df_global = None


# 🎨 ESTILOS REUTILIZABLES
BASE_HTML = """
<style>
body {{
    font-family: Arial;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    margin: 0;
    padding: 20px;
}}

h1 {{
    text-align: center;
}}

.nav {{
    text-align: center;
    margin-bottom: 20px;
}}

.nav a {{
    color: white;
    text-decoration: none;
    margin: 0 10px;
    padding: 8px 15px;
    border-radius: 10px;
    background: #00c6ff;
    color: black;
    font-weight: bold;
}}

.card {{
    background: white;
    color: black;
    padding: 15px;
    border-radius: 15px;
    margin: 20px auto;
    width: 90%;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}}

.center {{
    text-align: center;
}}

button {{
    padding: 10px 20px;
    border: none;
    border-radius: 10px;
    background: #00c6ff;
    font-weight: bold;
    cursor: pointer;
}}

input {{
    padding: 10px;
    border-radius: 8px;
    border: none;
}}
</style>
"""


# ------------------ HOME ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global df_global

    if request.method == "POST":
        file = request.files["file"]

        try:
            try:
                df_global = pd.read_csv(file, encoding="utf-8")
            except:
                file.seek(0)
                df_global = pd.read_csv(file, encoding="latin-1")

            return redirect("/dashboard")

        except Exception as e:
            return str(e)

    return f"""
    {BASE_HTML}
    <h1>📊 Subir dataset</h1>

    <div class="card center">
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br><br>
            <button type="submit">Subir</button>
        </form>
    </div>
    """


# ------------------ DASHBOARD ------------------
@app.route("/dashboard")
def dashboard():
    global df_global

    if df_global is None:
        return f"{BASE_HTML}<h2>Primero sube un dataset</h2><a href='/'>Volver</a>"

    df = df_global
    numeric_df = df.select_dtypes(include="number").dropna()

    scatter = px.scatter_matrix(numeric_df.iloc[:, :8])
    scatter_html = scatter.to_html(full_html=False)

    if "SALES" in df.columns:
        df_group = df.groupby(df.columns[0]).sum(numeric_only=True)
        line = px.line(x=df_group.index, y=df_group["SALES"], title="Sales")
        line_html = line.to_html(full_html=False)
    else:
        line_html = "<h3>No hay columna SALES</h3>"

    return f"""
    {BASE_HTML}
    <h1>📊 Dashboard</h1>

    <div class="nav">
        <a href="/analisis">Analisis</a>
        <a href="/clusters">Clusters</a>
        <a href="/pca">PCA</a>
    </div>

    <div class="card">{line_html}</div>
    <div class="card">{scatter_html}</div>
    """


# ------------------ ANALISIS ------------------
@app.route("/analisis")
def analisis():
    global df_global

    if df_global is None:
        return f"{BASE_HTML}<h2>Primero sube un dataset</h2><a href='/'>Volver</a>"

    df = df_global
    numeric_df = df.select_dtypes(include="number").dropna()

    corr = numeric_df.corr()
    heatmap = px.imshow(corr, text_auto=True)
    heatmap_html = heatmap.to_html(full_html=False)

    dist_html = ""
    for col in numeric_df.columns[:5]:
        fig = ff.create_distplot([numeric_df[col]], [col])
        dist_html += f'<div class="card">{fig.to_html(full_html=False)}</div>'

    return f"""
    {BASE_HTML}
    <h1>📈 Analisis</h1>

    <div class="nav">
        <a href="/dashboard">Dashboard</a>
    </div>

    <div class="card">{heatmap_html}</div>
    {dist_html}
    """


# ------------------ CLUSTERS ------------------
@app.route("/clusters")
def clusters():
    global df_global

    if df_global is None:
        return f"{BASE_HTML}<h2>Primero sube un dataset</h2><a href='/'>Volver</a>"

    df = df_global
    numeric_df = df.select_dtypes(include="number").dropna()

    scores = []
    r = range(1, 10)

    for i in r:
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit(numeric_df)
        scores.append(kmeans.inertia_)

    elbow = px.line(x=list(r), y=scores, title="Elbow Method")
    elbow_html = elbow.to_html(full_html=False)

    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(numeric_df)

    df_cluster = numeric_df.copy()
    df_cluster["cluster"] = labels

    hist_html = ""
    for col in numeric_df.columns[:3]:
        fig = px.histogram(df_cluster, x=col, color="cluster")
        hist_html += f'<div class="card">{fig.to_html(full_html=False)}</div>'

    return f"""
    {BASE_HTML}
    <h1>🤖 Clusters</h1>

    <div class="nav">
        <a href="/dashboard">Dashboard</a>
    </div>

    <div class="card">{elbow_html}</div>
    {hist_html}
    """


# ------------------ PCA ------------------
@app.route("/pca")
def pca_view():
    global df_global

    if df_global is None:
        return f"{BASE_HTML}<h2>Primero sube un dataset</h2><a href='/'>Volver</a>"

    df = df_global
    numeric_df = df.select_dtypes(include="number").dropna()

    pca = PCA(n_components=2)
    result = pca.fit_transform(numeric_df)

    pca_df = pd.DataFrame(result, columns=["pca1", "pca2"])

    kmeans = KMeans(n_clusters=3, n_init=10)
    pca_df["cluster"] = kmeans.fit_predict(pca_df)

    fig = px.scatter(pca_df, x="pca1", y="pca2", color="cluster")
    html = fig.to_html(full_html=False)

    return f"""
    {BASE_HTML}
    <h1>📉 PCA</h1>

    <div class="nav">
        <a href="/dashboard">Dashboard</a>
    </div>

    <div class="card">{html}</div>
    """


# ------------------ RUN LOCAL ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)