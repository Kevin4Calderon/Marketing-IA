from flask import Flask, request, render_template_string
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Dashboard</title>
</head>
<body style="font-family: Arial; background:#111; color:white; text-align:center;">

<h1>📊 Data Dashboard</h1>

<form method="POST" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button type="submit">Subir dataset</button>
</form>

<br><br>

<div>{{heatmap|safe}}</div>
<div>{{scatter|safe}}</div>
<div>{{pca3d|safe}}</div>
<div>{{clusters|safe}}</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files["file"]

            if file.filename == "":
                return "Error: No se seleccionó archivo"

            df = pd.read_csv(file)

            # 🔥 SOLO COLUMNAS NUMÉRICAS
            numeric_df = df.select_dtypes(include='number').dropna()

            if numeric_df.shape[1] < 3:
                return "Error: Se necesitan al menos 3 columnas numéricas"

            # 🔹 1. Heatmap
            corr = numeric_df.corr()
            heatmap = px.imshow(corr, text_auto=True)
            heatmap_html = heatmap.to_html(full_html=False)

            # 🔹 2. Scatter matrix
            scatter = px.scatter_matrix(numeric_df.iloc[:, :8])
            scatter_html = scatter.to_html(full_html=False)

            # 🔹 3. PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(numeric_df)

            pca_df = pd.DataFrame(pca_result, columns=['pca1','pca2','pca3'])

            # 🔹 4. Clustering
            kmeans = KMeans(n_clusters=3, n_init=10)
            pca_df['cluster'] = kmeans.fit_predict(pca_df)

            # 🔹 5. PCA 3D
            pca3d = px.scatter_3d(
                pca_df,
                x='pca1',
                y='pca2',
                z='pca3',
                color='cluster',
                opacity=0.7
            )
            pca3d_html = pca3d.to_html(full_html=False)

            # 🔹 6. Clusters 3D
            clusters = px.scatter_3d(
                pca_df,
                x='pca1',
                y='pca2',
                z='pca3',
                color='cluster',
                symbol='cluster'
            )
            clusters_html = clusters.to_html(full_html=False)

            # ✅ IMPORTANTE: render_template_string
            return render_template_string(
                HTML,
                heatmap=heatmap_html,
                scatter=scatter_html,
                pca3d=pca3d_html,
                clusters=clusters_html
            )

        except Exception as e:
            return f"<h2>Error interno:</h2><pre>{str(e)}</pre>"

    # ✅ IMPORTANTE: también aquí
    return render_template_string(HTML)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)