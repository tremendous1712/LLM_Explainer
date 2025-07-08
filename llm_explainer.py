"""
LLM Explainer: Interactive Language Model Visualization
======================================================

This Streamlit app demonstrates how Large Language Models (LLMs) work internally.
Users can explore tokenization, embeddings, attention mechanisms, and sentence completion.

Requirements:
- streamlit
- transformers
- torch
- matplotlib
- seaborn
- scikit-learn
- plotly

Run with: streamlit run llm_explainer.py
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

# Set page config
st.set_page_config(
    page_title="LLM Explainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem !important;
}
.flowchart-container {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    padding: 10px 10px 10px 10px !important;
}
.flowchart-btn {
    white-space: normal !important;
    word-break: break-word !important;
    font-size: 1.05rem !important;
    min-height: 48px !important;
    min-width: 120px !important;
    max-width: 180px !important;
    line-height: 1.2 !important;
    margin-bottom: 0 !important;
}
.content-area {
    margin-top: 0.5rem !important;
}
.generated-text {
    background: #f3f3f3 !important;
    color: #222 !important;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    font-family: 'Georgia', serif;
    font-size: 1.1rem;
    line-height: 1.6;
    box-shadow: 0 5px 20px rgba(0,0,0,0.07);
    word-break: break-word;
}
.token-pill {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff !important;
    border-radius: 20px;
    padding: 8px 15px;
    margin: 5px;
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.insight-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #f1f8ff 100%);
    color: #222 !important;
    border-left: 4px solid #667eea;
    padding: 20px;
    margin: 20px 0;
    border-radius: 8px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-size: 1.2rem;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 2rem;
}

.flowchart-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.flowchart-step {
    background: white;
    border: 2px solid #e1e8ed;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.flowchart-step:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    border-color: #667eea;
}

.flowchart-step.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
}

.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.step-description {
    font-size: 0.9rem;
    opacity: 0.8;
    line-height: 1.4;
}

.flowchart-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: #667eea;
    margin: 5px 0;
}

.content-area {
    background: white;
    border-radius: 15px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    min-height: 600px;
}

    .step-header {
        font-size: 2rem;
        font-weight: 700;
        color: #3366cc;
        margin-bottom: 1.5rem;
        padding-bottom: 10px;
        border-bottom: 3px solid #3366cc;
    }

.token-pill {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 20px;
    padding: 8px 15px;
    margin: 5px;
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.insight-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #f1f8ff 100%);
    border-left: 4px solid #667eea;
    padding: 20px;
    margin: 20px 0;
    border-radius: 8px;
    font-style: italic;
}

.generated-text {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    font-family: 'Georgia', serif;
    font-size: 1.1rem;
    line-height: 1.6;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.highlight-completion {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 600;
}

.metrics-container {
    display: flex;
    justify-content: space-around;
    margin: 20px 0;
}

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    min-width: 120px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.metric-label {
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-top: 5px;
}

.attention-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.attention-head {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    cursor: pointer;
}

.attention-head:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.preview-tooltip {
    position: absolute;
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 10px;
    border-radius: 5px;
    font-size: 0.8rem;
    z-index: 1000;
    max-width: 200px;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'tokenization'

@st.cache_resource
def load_model():
    # Load the GPT-2 model and tokenizer for better performance
    with st.spinner("Loading model... Please wait"):
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium",
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        # Set pad tokenx
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

def create_flowchart():
    """Create interactive flowchart navigation"""
    steps = [
        {
            'id': 'tokenization',
            'title': 'Tokenization',
            'description': 'Breaking text into tokens',
            'preview': 'Convert "Hello world" → ["Hello", " world"]'
        },
        {
            'id': 'embeddings',
            'title': 'Vector Embeddings',
            'description': 'Converting tokens to vectors',
            'preview': 'Map tokens to high-dimensional space'
        },
        {
            'id': 'attention',
            'title': 'Attention Mechanism',
            'description': 'Understanding token relationships',
            'preview': 'Visualize all attention heads with zoom'
        },
        {
            'id': 'probabilities',
            'title': 'Next-Token Probabilities',
            'description': 'Predicting what comes next',
            'preview': 'Show probability distribution over vocabulary'
        },
        {
            'id': 'completion',
            'title': 'Sentence Completion',
            'description': 'Complete the input sentence',
            'preview': 'Generate meaningful sentence endings'
        }
    ]
    
    # Visually appealing horizontal flowchart bar with compact layout and no extra white bar above
    # Horizontal flowchart using st.columns for true horizontal layout, no extra white bar
    steps_len = len(steps)
    cols = st.columns(steps_len * 2 - 1, gap="small")
    for i, step in enumerate(steps):
        col = cols[i * 2]
        with col:
            is_active = st.session_state.current_step == step['id']
            btn_label = step['title']
            btn = st.button(
                btn_label,
                key=f"step_{step['id']}",
                help=step['preview'],
                use_container_width=True,
                type="primary" if is_active else "secondary"
            )
            st.markdown(f'<style>div[data-testid="stButton"] button {{white-space:normal;word-break:break-word;min-width:115px;max-width:175px;min-height:44px;font-size:1.05rem;line-height:1.2;}}</style>', unsafe_allow_html=True)
            if btn:
                st.session_state.current_step = step['id']
                st.rerun()
        if i < steps_len - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:2rem;color:#667eea;">→</div>', unsafe_allow_html=True)

def display_tokens(tokens, token_ids):
    # Display tokens in a modern pill-style format
    import re
    st.markdown("**Original Input:**")
    # Show the original string in a monospace font
    st.markdown('<div style="font-family:monospace;font-size:1.1rem;padding:8px 0 16px 0;">'+st.session_state.user_input+'</div>', unsafe_allow_html=True)

    # Step 1: Highlight how tokenization splits the string
    # We'll reconstruct the string with each token highlighted in color
    colored_string = ""
    input_text = st.session_state.user_input
    input_pointer = 0
    for i, token in enumerate(tokens):
        decoded = st.session_state.tokenizer.decode([token_ids[i]], clean_up_tokenization_spaces=False)
        # Classify token type and set translucent background
        if re.fullmatch(r"[\s\.,;:!?]+", decoded):
            color = "rgba(255,209,102,0.45)"  # punctuation
            border = "#FFD166"
        elif re.fullmatch(r"\d+", decoded.strip()):
            color = "rgba(6,214,160,0.45)"  # number
            border = "#06D6A0"
        elif re.fullmatch(r"[A-Za-z\-']+", decoded.strip()):
            color = "rgba(17,138,178,0.45)"  # word
            border = "#118AB2"
        else:
            color = "rgba(239,71,111,0.45)"  # special/other
            border = "#EF476F"
        idx = input_text.find(decoded, input_pointer)
        if idx == -1:
            idx = input_pointer
        if idx > input_pointer:
            colored_string += input_text[input_pointer:idx]
        colored_string += f'<span style="background:{color};border:1.5px solid {border};color:#222;padding:2px 4px 2px 4px;border-radius:7px;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin:0 1px;">{decoded}</span>'
        input_pointer = idx + len(decoded)
    if input_pointer < len(input_text):
        colored_string += input_text[input_pointer:]
    st.markdown("**Tokenization Process:**")
    st.markdown(f'<div style="font-family:monospace;font-size:1.1rem;line-height:2;">{colored_string}</div>', unsafe_allow_html=True)

    # Step 2: Show token table with info
    st.markdown("**Tokens:**")
    table = "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:separate;border-spacing:0 4px;font-size:1rem;background:#fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.03);'>"
    table += "<tr style='background:#f5f5f5;color:#222;'><th style='padding:8px 10px;border-radius:8px 0 0 8px;'>#</th><th style='padding:8px 10px;'>Token</th><th style='padding:8px 10px;'>ID</th><th style='padding:8px 10px;'>Type</th><th style='padding:8px 10px;border-radius:0 8px 8px 0;'>Decoded</th></tr>"
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        decoded = st.session_state.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if re.fullmatch(r"[\s\.,;:!?]+", decoded):
            color = "#FFD166"; typ = "punctuation"
        elif re.fullmatch(r"\d+", decoded.strip()):
            color = "#06D6A0"; typ = "number"
        elif re.fullmatch(r"[A-Za-z\-']+", decoded.strip()):
            color = "#118AB2"; typ = "word"
        else:
            color = "#EF476F"; typ = "special/other"
        table += f"<tr style='background:rgba(245,245,245,0.85);color:#222;'>"
        table += f"<td style='padding:6px 10px;text-align:center;font-weight:600;'>{i+1}</td>"
        table += f"<td style='padding:6px 10px;'><span class='token-pill' style='background:{color};color:#222;font-weight:600;'>{token}</span></td>"
        table += f"<td style='padding:6px 10px;text-align:center;'>{token_id}</td>"
        table += f"<td style='padding:6px 10px;text-align:center;'>{typ}</td>"
        table += f"<td style='padding:6px 10px;font-family:monospace;'>{decoded}</td></tr>"
    table += "</table></div>"
    st.markdown(table, unsafe_allow_html=True)

def visualize_embeddings(hidden_states, tokens, max_tokens=10):
    # Visualize token embeddings in 3D using PCA
    embeddings = hidden_states[0][0].detach().numpy()
    
    # Limit tokens for clarity
    embeddings = embeddings[:max_tokens]
    tokens_subset = tokens[:max_tokens]
    
    # Apply PCA to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Clean tokens for display
    clean_tokens = [token.replace("Ġ", " ").replace("</w>", "") for token in tokens_subset]
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers+text',
        text=clean_tokens,
        textposition='top center',
        marker=dict(
            size=12,
            color=list(range(len(clean_tokens))),
            colorscale='plasma',
            showscale=True,
            colorbar=dict(title="Token Position"),
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        hovertemplate='<b>Token:</b> %{text}<br><b>PC1:</b> %{x:.3f}<br><b>PC2:</b> %{y:.3f}<br><b>PC3:</b> %{z:.3f}<extra></extra>'
    ))
    
    # Add connecting lines to show sequence order
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='lines',
        line=dict(
            color='rgba(102, 126, 234, 0.5)',
            width=3
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Token Embeddings in 3D Space (PCA Projection)",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'),
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'),
            zaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)')
        ),
        showlegend=False,
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add controls for 3D view
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pca.explained_variance_ratio_[:3].sum():.1%}</div><div class="metric-label">Total Variance Explained</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{embeddings.shape[1]}</div><div class="metric-label">Original Dimensions</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">Insight: The 3D visualization shows how tokens occupy different regions in the high-dimensional embedding space. Connected lines show the sequence order, and similar tokens cluster together in 3D space. You can rotate and zoom the plot to explore different angles!</div>', unsafe_allow_html=True)

def visualize_attention_all_heads(attention_weights, tokens, layer_idx=0):
    # Visualize all attention heads in a grid with hover zoom functionality
    attention = attention_weights[layer_idx][0].detach().numpy()
    num_heads = attention.shape[0]
    
    # Clean tokens for display
    clean_tokens = [token.replace("Ġ", " ").replace("</w>", "") for token in tokens]
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    # Create subplots
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f"Head {i+1}" for i in range(num_heads)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    for head in range(num_heads):
        row = head // cols + 1
        col = head % cols + 1
        
        heatmap = go.Heatmap(
            z=attention[head],
            x=clean_tokens,
            y=clean_tokens,
            colorscale='Viridis',
            showscale=(head == 0),
            hovertemplate='<b>Head:</b> %{meta}<br><b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Attention:</b> %{z:.3f}<extra></extra>',
            meta=f"Head {head+1}"
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    # Update layout for better visualization
    fig.update_layout(
        title=f"All Attention Heads - Layer {layer_idx + 1}",
        height=200 * rows,
        showlegend=False
    )
    
    # Update axes
    for i in range(1, num_heads + 1):
        fig.update_xaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)
        fig.update_yaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interactive zoom feature
    st.markdown("**Click on any attention head above to zoom in:**")
    selected_head = st.selectbox("Select head to examine in detail:", 
                                options=range(1, num_heads + 1), 
                                format_func=lambda x: f"Head {x}")
    
    if selected_head:
        # Create detailed view of selected head
        fig_detail = go.Figure(data=go.Heatmap(
            z=attention[selected_head - 1],
            x=clean_tokens,
            y=clean_tokens,
            colorscale='Viridis',
            hovertemplate='<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Attention:</b> %{z:.3f}<extra></extra>'
        ))
        
        fig_detail.update_layout(
            title=f"Detailed View - Head {selected_head}",
            xaxis_title="To Token",
            yaxis_title="From Token",
            height=500
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
    
    st.markdown('<div class="insight-box">Insight: Each attention head learns to focus on different types of relationships. Some heads focus on syntax, others on semantics, and some on positional relationships.</div>', unsafe_allow_html=True)

def show_next_token_probabilities(logits, tokenizer, top_k=15):
    # Display top-k next token predictions with enhanced visualization
    probs = torch.softmax(logits, dim=-1)   
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Convert to tokens
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
    top_probs_list = top_probs[0].tolist()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Token': top_tokens,
        'Probability': top_probs_list
    })
    
    # Create enhanced bar chart
    fig = px.bar(
        df, 
        x='Token', 
        y='Probability',
        title=f"Top {top_k} Next Token Predictions",
        color='Probability',
        color_continuous_scale='plasma',
        text='Probability'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Predicted Token",
        yaxis_title="Probability",
        showlegend=False,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top prediction with enhanced styling
    st.markdown(f'<div class="insight-box">Most likely next token: <strong>"{top_tokens[0]}"</strong> with probability {top_probs_list[0]:.3f}</div>', unsafe_allow_html=True)

def complete_sentence(model, tokenizer, input_text, max_length=30):
    # Complete the sentence with better control and visualization
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    st.markdown("**Original Input:**")
    st.markdown(f'<div class="generated-text">{input_text}</div>', unsafe_allow_html=True)
    
    # Generate completion
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get completed text
    generated_ids = output.sequences[0]
    complete_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract only the completion part, up to the first period
    completion = complete_text[len(input_text):].strip()
    first_period = completion.find('.')
    if first_period != -1:
        completion = completion[:first_period+1]

    st.markdown("**Completed Sentence:**")
    completed_html = f'{input_text}<span class="highlight-completion">{completion}</span>'
    st.markdown(f'<div class="generated-text">{completed_html}</div>', unsafe_allow_html=True)
    
    # Show generation statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(completion.split())}</div><div class="metric-label">Words Added</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(tokenizer.encode(completion))}</div><div class="metric-label">Tokens Added</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(complete_text.split())}</div><div class="metric-label">Total Words</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">The model uses the same process (tokenization → embeddings → attention → prediction) iteratively to generate each new token, building context as it goes.</div>', unsafe_allow_html=True)
    
    return complete_text

def main():
    # Title
    st.markdown('<h1 class="main-header">LLM Explainer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Interactive Language Model Visualization</p>', unsafe_allow_html=True)
    
    # Sidebar for input and settings
    with st.sidebar:
        st.markdown("## Input & Settings")
        # Load model first
        if not st.session_state.model_loaded:
            try:
                st.session_state.tokenizer, st.session_state.model = load_model()
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()
        user_input = st.text_area(
            "Enter your prompt:",
            value="Line 42: segmentation fault. Classic rite of",
            height=100,
            help="Enter text to explore how the model processes it"
        )
        max_tokens_embed = st.slider("Max tokens for embedding", 5, 20, 10)
        attention_layer = st.slider("Attention layer", 1, 12, 1)
        st.session_state.user_input = user_input
        st.session_state.max_tokens_embed = max_tokens_embed
        st.session_state.attention_layer = attention_layer

    # Horizontal flowchart at the top (no white bar above)
    create_flowchart()

    # Main content area: always render inside the white box directly below the flowchart
    user_input = st.session_state.get("user_input", "")
    max_tokens_embed = st.session_state.get("max_tokens_embed", 10)
    attention_layer = st.session_state.get("attention_layer", 1)

    if user_input and st.session_state.model_loaded:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model

        # Process input
        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        step = st.session_state.current_step
        # Adjust for 0-based indexing for attention layer
        attn_layer_idx = attention_layer - 1

        if step == 'tokenization':
            st.markdown('<h2 class="step-header">Tokenization</h2>', unsafe_allow_html=True)
            st.markdown("**How the model breaks down your text into tokens:**")
            display_tokens(tokens, input_ids[0].tolist())
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(tokens)}</div><div class="metric-label">Total Tokens</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{tokenizer.vocab_size:,}</div><div class="metric-label">Vocabulary Size</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box">The model converts your text into numerical tokens using Byte-Pair Encoding (BPE). Each token represents a common word, part of a word, or character that the model has learned during training.</div>', unsafe_allow_html=True)

        elif step == 'embeddings':
            st.markdown('<h2 class="step-header">Vector Embeddings</h2>', unsafe_allow_html=True)
            st.markdown("**How tokens become high-dimensional vectors:**")
            visualize_embeddings(outputs.hidden_states, tokens, max_tokens_embed)
            embedding_dim = outputs.hidden_states[0].shape[-1]
            st.markdown(f'<div class="metric-card"><div class="metric-value">{embedding_dim}</div><div class="metric-label">Embedding Dimension</div></div>', unsafe_allow_html=True)

        elif step == 'attention':
            st.markdown('<h2 class="step-header">Attention Mechanism</h2>', unsafe_allow_html=True)
            st.markdown("**How the model decides which tokens to focus on:**")
            # --- Plotly grid of all heads, hover to zoom ---
            attention = outputs.attentions[attn_layer_idx][0].detach().numpy()
            num_heads = attention.shape[0]
            clean_tokens = [token.replace("Ġ", " ").replace("</w>", "") for token in tokens]
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            # Show all heads in a grid
            cols = 4
            rows = (num_heads + cols - 1) // cols
            subplot_titles = [f"Head {i+1}" for i in range(num_heads)]
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            for i in range(num_heads):
                row = i // cols + 1
                col = i % cols + 1
                fig.add_trace(
                    go.Heatmap(
                        z=attention[i],
                        x=clean_tokens,
                        y=clean_tokens,
                        colorscale='Viridis',
                        showscale=False,
                        hovertemplate=f'<b>Head {i+1}</b><br>From: %{{y}}<br>To: %{{x}}<br>Attention: %{{z:.3f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            fig.update_layout(
                title=f"All Attention Heads - Layer {attn_layer_idx+1}",
                height=220 * rows,
                showlegend=False
            )
            for i in range(1, num_heads + 1):
                fig.update_xaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)
                fig.update_yaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            # Fallback: selectbox for now
            selected_head = st.selectbox("Zoomed Head", options=list(range(num_heads)), format_func=lambda x: f"Head {x+1}")
            fig_detail = go.Figure(data=go.Heatmap(
                z=attention[selected_head],
                x=clean_tokens,
                y=clean_tokens,
                colorscale='Viridis',
                hovertemplate='<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Attention:</b> %{z:.3f}<extra></extra>'
            ))
            fig_detail.update_layout(
                title=f"Zoomed-In: Head {selected_head+1}",
                xaxis_title="To Token",
                yaxis_title="From Token",
                height=500
            )
            st.plotly_chart(fig_detail, use_container_width=True)
            num_layers = len(outputs.attentions)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{num_heads}</div><div class="metric-label">Attention Heads</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{num_layers}</div><div class="metric-label">Attention Layers</div></div>', unsafe_allow_html=True)

        elif step == 'probabilities':
            st.markdown('<h2 class="step-header">Next-Token Prediction</h2>', unsafe_allow_html=True)
            st.markdown("**What the model thinks should come next:**")
            logits = outputs.logits[:, -1, :]
            show_next_token_probabilities(logits, tokenizer)
            st.markdown('<div class="insight-box">The model outputs a probability distribution over its entire vocabulary. The softmax function converts raw logits into probabilities that sum to 1.</div>', unsafe_allow_html=True)

        elif step == 'completion':
            st.markdown('<h2 class="step-header">Sentence Completion</h2>', unsafe_allow_html=True)
            st.markdown("**Complete your input with AI-generated text:**")
            with st.spinner("Generating completion..."):
                completed_text = complete_sentence(
                    model, tokenizer, user_input, 25
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">'
        '<p>Built with Streamlit</p>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()