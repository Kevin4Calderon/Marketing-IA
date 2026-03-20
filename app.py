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
    <title>Dashboard</title>
</head>
<body style="font-family: Arial; background:#111; color:white; text-align:center;">

<h1>📊 Data Dashboard</h1>

<form method="POST" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button type="submit">Subir dataset</button>
</form>

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
        file = request.files["file"]
        df = pd.read_csv(file)

        # 🔥 Heatmap
        corr = df.corr(numeric_only=True)
        heatmap = px.imshow(corr, text_auto=True)
        heatmap_html = heatmap.to_html(full_html=False)

        # 📊 Scatter matrix
        scatter = px.scatter_matrix(df, dimensions=df.columns[:8])
        scatter_html = scatter.to_html(full_html=False)

        # 🌐 PCA
        numeric_df = df.select_dtypes(include='number').dropna()
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(numeric_df)

        pca_df = pd.DataFrame(pca_result, columns=['pca1','pca2','pca3'])

        # 🧠 Clustering
        kmeans = KMeans(n_clusters=3, n_init=10)
        pca_df['cluster'] = kmeans.fit_predict(pca_df)

        # 🌐 PCA 3D
        pca3d = px.scatter_3d(
            pca_df, x='pca1', y='pca2', z='pca3',
            color='cluster', opacity=0.7
        )
        pca3d_html = pca3d.to_html(full_html=False)

        # 🧠 Clusters 3D
        clusters = px.scatter_3d(
            pca_df, x='pca1', y='pca2', z='pca3',
            color='cluster', symbol='cluster'
        )
        clusters_html = clusters.to_html(full_html=False)

        return render_template_string(
            HTML,
            heatmap=heatmap_html,
            scatter=scatter_html,
            pca3d=pca3d_html,
            clusters=clusters_html
        )

    return HTML

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)