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
from Bio import SeqIO

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'genome_x_xgboost.model')

# --- 1. BIOLOGY HELPERS ---
def calculate_gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) * 100

def get_molecular_weight(seq):
    weights = {'A': 313.2, 'T': 304.2, 'G': 329.2, 'C': 289.2}
    return sum(weights.get(base, 0) for base in seq)

def encode_sequence(seq_list):
    """Converts DNA to Model-Ready Features"""
    features = []
    mapping = {'A':1, 'T':2, 'G':3, 'C':4}
    for seq in seq_list:
        gc = calculate_gc(seq)
        mw = get_molecular_weight(seq)
        num_seq = [mapping.get(base, 0) for base in seq]
        features.append([gc, mw] + num_seq)
    return features

# --- 2. INTERACTIVE CHART GENERATOR (PLOTLY) ---
def generate_interactive_charts(df):
    charts = {}

    # Chart A: Confidence Distribution (Interactive Histogram)
    fig_dist = px.histogram(
        df, x='Predicted_Efficiency', nbins=20, 
        title='AI Confidence Distribution',
        labels={'Predicted_Efficiency': 'Efficiency Score (0-1)'},
        color_discrete_sequence=['#00f2c3'],
        template='plotly_dark'
    )
    # FIX: Replaced 'bg_color' with 'paper_bgcolor' and 'plot_bgcolor'
    fig_dist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color="white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    charts['score_dist'] = pio.to_html(fig_dist, full_html=False, config={'displayModeBar': False})

    # Chart B: Thermodynamic Safety Zone (Interactive Scatter)
    fig_scatter = px.scatter(
        df, x='GC_Content', y='Predicted_Efficiency',
        color='Predicted_Efficiency',
        title='Thermodynamic Stability vs. Efficiency',
        labels={'GC_Content': 'GC Content (%)', 'Predicted_Efficiency': 'AI Score'},
        color_continuous_scale='Bluered',
        hover_data=['seq', 'pam'],
        template='plotly_dark'
    )
    # Add Red Zones (Lines)
    fig_scatter.add_vline(x=40, line_dash="dash", line_color="#ff0055", annotation_text="Min Stability")
    fig_scatter.add_vline(x=60, line_dash="dash", line_color="#ff0055", annotation_text="Max Stability")
    
    # FIX: Replaced 'bg_color'
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color="white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    charts['gc_scatter'] = pio.to_html(fig_scatter, full_html=False, config={'displayModeBar': False})

    # Chart C: Nucleotide Composition (Interactive Pie)
    all_seq = "".join(df['seq'].tolist())
    counts = pd.DataFrame({
        'Base': ['A', 'T', 'G', 'C'], 
        'Count': [all_seq.count(b) for b in 'ATGC']
    })
    
    fig_pie = px.pie(
        counts, values='Count', names='Base',
        title='Global Nucleotide Composition',
        color='Base',
        color_discrete_map={'A':'#ff0055', 'T':'#ffe600', 'G':'#00f2c3', 'C':'#0099ff'},
        template='plotly_dark'
    )
    # FIX: Replaced 'bg_color'
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color="white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    charts['composition'] = pio.to_html(fig_pie, full_html=False, config={'displayModeBar': False})

    return charts

# --- 3. CONTROLLERS ---
def home(request):
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        try:
            # Handle File
            myfile = request.FILES['fasta_file']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            file_path = fs.path(filename)

            # Parse DNA
            records = list(SeqIO.parse(file_path, "fasta"))
            candidates = []
            for rec in records:
                seq = str(rec.seq).upper()
                for i in range(len(seq) - 23):
                    pam = seq[i+20:i+23]
                    if pam.endswith("GG"):
                        candidates.append({'id': rec.id, 'pos': i, 'seq': seq[i:i+20], 'pam': pam})

            if not candidates:
                return render(request, 'index.html', {'error': "Invalid DNA Data. No CRISPR targets (NGG) found."})

            # AI Processing
            df = pd.DataFrame(candidates)
            if not os.path.exists(MODEL_PATH):
                return render(request, 'index.html', {'error': "Model Missing! Place 'genome_x_xgboost.model' in root folder."})
            
            model = joblib.load(MODEL_PATH)
            X_input = encode_sequence(df['seq'])
            df['Predicted_Efficiency'] = model.predict(X_input)
            df['GC_Content'] = df['seq'].apply(calculate_gc)

            # Format Results
            df = df.sort_values(by='Predicted_Efficiency', ascending=False)
            request.session['results_csv'] = df.to_csv(index=False)
            
            # Display Data (Rounding)
            display_df = df.copy()
            display_df['GC_Content'] = display_df['GC_Content'].round(1)
            display_df['Predicted_Efficiency'] = display_df['Predicted_Efficiency'].round(4)

            top_candidates = display_df[
                (display_df['Predicted_Efficiency'] > 0.80) & 
                (display_df['GC_Content'] >= 40) & 
                (display_df['GC_Content'] <= 60)
            ].head(20).to_dict('records')

            charts = generate_interactive_charts(df)

            return render(request, 'dashboard.html', {
                'candidates': top_candidates,
                'charts': charts,
                'total': len(df),
                'qualified': len(top_candidates)
            })
            
        except Exception as e:
            print(f"SYSTEM ERROR: {e}")
            return render(request, 'index.html', {'error': f"System Error: {str(e)}"})

    return render(request, 'index.html')

def download_csv(request):
    csv_data = request.session.get('results_csv')
    if csv_data:
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="genome_x_report.csv"'
        return response
    return HttpResponse("No data available.")