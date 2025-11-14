import plotly.graph_objects as go

def plot_radar(validity):
    if not validity:
        return
    labels = list(validity.keys())
    omega_values = [validity[d]["omega"] for d in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=omega_values, theta=labels, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    fig.show()
