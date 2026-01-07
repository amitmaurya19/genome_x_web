import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import joblib
import os

from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from Bio import SeqIO

# ============================================================
# PATH & STORAGE CONFIG (PRODUCTION SAFE)
# ============================================================

BASE_DIR = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "genome_x_xgboost.model")

# Ensure media directory exists (IMPORTANT for Render)
os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

# ============================================================
# BIOLOGY HELPERS
# ============================================================

def calculate_gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) * 100


def get_molecular_weight(seq):
    weights = {'A': 313.2, 'T': 304.2, 'G': 329.2, 'C': 289.2}
    return sum(weights.get(base, 0) for base in seq)


def encode_sequence(seq_list):
    features = []
    mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4}

    for seq in seq_list:
        gc = calculate_gc(seq)
        mw = get_molecular_weight(seq)
        num_seq = [mapping.get(base, 0) for base in seq]
        features.append([gc, mw] + num_seq)

    return features

# ============================================================
# INTERACTIVE CHARTS
# ============================================================

def generate_interactive_charts(df):
    charts = {}

    fig_dist = px.histogram(
        df,
        x="Predicted_Efficiency",
        nbins=20,
        title="AI Confidence Distribution",
        template="plotly_dark"
    )

    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    charts["score_dist"] = pio.to_html(fig_dist, full_html=False)

    fig_scatter = px.scatter(
        df,
        x="GC_Content",
        y="Predicted_Efficiency",
        color="Predicted_Efficiency",
        hover_data=["seq", "pam"],
        title="GC Content vs Efficiency",
        template="plotly_dark"
    )

    fig_scatter.add_vline(x=40, line_dash="dash", line_color="red")
    fig_scatter.add_vline(x=60, line_dash="dash", line_color="red")

    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    charts["gc_scatter"] = pio.to_html(fig_scatter, full_html=False)

    counts = pd.DataFrame({
        "Base": ["A", "T", "G", "C"],
        "Count": [
            "".join(df["seq"]).count("A"),
            "".join(df["seq"]).count("T"),
            "".join(df["seq"]).count("G"),
            "".join(df["seq"]).count("C"),
        ]
    })

    fig_pie = px.pie(
        counts,
        values="Count",
        names="Base",
        title="Nucleotide Composition",
        template="plotly_dark"
    )

    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    charts["composition"] = pio.to_html(fig_pie, full_html=False)

    return charts

# ============================================================
# VIEWS
# ============================================================

def home(request):
    if request.method == "POST" and request.FILES.get("fasta_file"):
        try:
            uploaded_file = request.FILES["fasta_file"]

            # Save file safely
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            # Parse FASTA
            records = list(SeqIO.parse(file_path, "fasta"))
            candidates = []

            for rec in records:
                seq = str(rec.seq).upper()
                for i in range(len(seq) - 23):
                    pam = seq[i + 20:i + 23]
                    if pam.endswith("GG"):
                        candidates.append({
                            "id": rec.id,
                            "pos": i,
                            "seq": seq[i:i + 20],
                            "pam": pam
                        })

            if not candidates:
                return render(request, "index.html", {
                    "error": "No CRISPR targets (NGG) found."
                })

            if not os.path.exists(MODEL_PATH):
                return render(request, "index.html", {
                    "error": "ML model file missing on server."
                })

            model = joblib.load(MODEL_PATH)

            df = pd.DataFrame(candidates)
            X = encode_sequence(df["seq"])
            df["Predicted_Efficiency"] = model.predict(X)
            df["GC_Content"] = df["seq"].apply(calculate_gc)

            df = df.sort_values("Predicted_Efficiency", ascending=False)
            request.session["results_csv"] = df.to_csv(index=False)

            display_df = df.copy()
            display_df["Predicted_Efficiency"] = display_df["Predicted_Efficiency"].round(4)
            display_df["GC_Content"] = display_df["GC_Content"].round(1)

            top_candidates = display_df[
                (display_df["Predicted_Efficiency"] > 0.8) &
                (display_df["GC_Content"].between(40, 60))
            ].head(20).to_dict("records")

            charts = generate_interactive_charts(df)

            return render(request, "dashboard.html", {
                "candidates": top_candidates,
                "charts": charts,
                "total": len(df),
                "qualified": len(top_candidates)
            })

        except Exception as e:
            return render(request, "index.html", {
                "error": f"System Error: {str(e)}"
            })

    return render(request, "index.html")


def download_csv(request):
    csv_data = request.session.get("results_csv")
    if csv_data:
        response = HttpResponse(csv_data, content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="genome_x_report.csv"'
        return response

    return HttpResponse("No data available.")
