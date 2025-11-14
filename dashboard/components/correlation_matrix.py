import plotly.express as px
import pandas as pd

def plot_corr(validity):
    if not validity:
        return
    df = pd.DataFrame(validity).T
    fig = px.imshow(df, color_continuous_scale='RdBu', text_auto=True)
    fig.show()
