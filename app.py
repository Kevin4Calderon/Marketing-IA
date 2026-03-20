from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)

# 🔹 variable global simple (para práctica)
df_global = None


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

    return """
    <h1>Subir dataset</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Subir</button>
    </form>
    """


# ------------------ DASHBOARD ------------------
@app.route("/dashboard")
def dashboard():
    global df_global
    df = df_global

    numeric_df = df.select_dtypes(include="number").dropna()

    # 🔹 Scatter matrix
    scatter = px.scatter_matrix(numeric_df.iloc[:, :8])
    scatter_html = scatter.to_html(full_html=False)

    # 🔹 Línea (sales)
    if "SALES" in df.columns:
        df_group = df.groupby(df.columns[0]).sum(numeric_only=True)
        line = px.line(x=df_group.index, y=df_group["SALES"], title="Sales")
        line_html = line.to_html(full_html=False)
    else:
        line_html = "<h3>No hay columna SALES</h3>"

    return f"""
    <h1>Dashboard</h1>
    <a href='/analisis'>Analisis</a> |
    <a href='/clusters'>Clusters</a> |
    <a href='/pca'>PCA</a>
    {line_html}
    {scatter_html}
    """


# ------------------ ANALISIS ------------------
@app.route("/analisis")
def analisis():
    global df_global
    df = df_global

    numeric_df = df.select_dtypes(include="number").dropna()

    # 🔹 Heatmap
    corr = numeric_df.corr()
    heatmap = px.imshow(corr, text_auto=True)
    heatmap_html = heatmap.to_html(full_html=False)

    # 🔹 Distribuciones
    dist_html = ""
    for col in numeric_df.columns[:5]:
        fig = ff.create_distplot([numeric_df[col]], [col])
        dist_html += fig.to_html(full_html=False)

    return f"""
    <h1>Analisis</h1>
    <a href='/dashboard'>Dashboard</a>
    {heatmap_html}
    {dist_html}
    """


# ------------------ CLUSTERS ------------------
@app.route("/clusters")
def clusters():
    global df_global
    df = df_global

    numeric_df = df.select_dtypes(include="number").dropna()

    # 🔹 Elbow method
    scores = []
    r = range(1, 10)

    for i in r:
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit(numeric_df)
        scores.append(kmeans.inertia_)

    elbow = px.line(x=list(r), y=scores, title="Elbow Method")
    elbow_html = elbow.to_html(full_html=False)

    # 🔹 Clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(numeric_df)

    df_cluster = numeric_df.copy()
    df_cluster["cluster"] = labels

    hist_html = ""
    for col in numeric_df.columns[:3]:
        fig = px.histogram(df_cluster, x=col, color="cluster")
        hist_html += fig.to_html(full_html=False)

    return f"""
    <h1>Clusters</h1>
    <a href='/dashboard'>Dashboard</a>
    {elbow_html}
    {hist_html}
    """


# ------------------ PCA ------------------
@app.route("/pca")
def pca_view():
    global df_global
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
    <h1>PCA</h1>
    <a href='/dashboard'>Dashboard</a>
    {html}
    """


# ------------------ RUN ------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)