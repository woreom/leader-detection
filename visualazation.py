
import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot


def plot_data(df):
    """
    Plots the data from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        fig (plotly.graph_objects.Figure): The plotly figure object.
    """
    t = np.arange(0, len(df))
    fig = make_subplots(rows=1, cols=1)
    
    for col in df.columns:
        fig.add_trace(go.Scatter(x=t, y=df[col].to_numpy(), 
                             name=f'{col}', line=dict(width=1.5)), row=1, col=1)
    
    layout = dict(
        title='Processed Data',
        xaxis_rangeslider_visible=False,
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True,
        width=1410,
        height=594
    )
    
    fig.update_layout(layout)
    return fig
    

def plot_heatmap(df, title='Heatmap', xlabel='x', ylabel='y', cmap='greys'):
    """
    Plots a heatmap of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        title (str, optional): The title of the plot. Defaults to 'Heatmap'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'x'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'y'.
        cmap (str, optional): The color map to use. Defaults to 'greys'.

    Returns:
        fig (plotly.graph_objects.Figure): The plotly figure object.
    """
    fig = ff.create_annotated_heatmap(
        np.array(df.values),
        x=list(df.columns),
        y=list(df.index),
        annotation_text=df.values,
        hoverinfo='z',
        colorscale='Viridis'
    )
    
    fig.update_layout(
        title='Lead Lag Matrix',
    )
    
    return fig


def constellation_plot(leading_eigenvector, sequential_order_dict):
    """
    Plots the constellation plot of a leading eigenvector.

    Args:
        leading_eigenvector (np.ndarray): The leading eigenvector.
        sequential_order_dict (dict): A dictionary mapping indices to column names.

    Returns:
        fig (plotly.graph_objects.Figure): The plotly figure object.
    """
    sequential_order_indices = list(sequential_order_dict.keys()) 
    sequential_order_columns = list(sequential_order_dict.values())
    x, y = np.real(leading_eigenvector), np.imag(leading_eigenvector)
    N = len(leading_eigenvector)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, 
        y=y,
        mode='markers',
        marker=dict(
            size=10,
            color='blue'
        ),
        showlegend=False
    ))
    
    for n in range(N-1):
        fig.add_trace(go.Scatter(
            x=[x[sequential_order_indices[n]], x[sequential_order_indices[n+1]]],
            y=[y[sequential_order_indices[n]], y[sequential_order_indices[n+1]]], 
            mode='lines',
            line=dict(
                width=2,
                color='black'
            ),
            showlegend=False
        ))
      
    for n in range(N-1):
        x0, y0 = x[sequential_order_indices[n]], y[sequential_order_indices[n]]  # Tail coordinates
        x1, y1 = x[sequential_order_indices[n+1]], y[sequential_order_indices[n+1]]  # Head coordinates
        
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowcolor='black',
            arrowsize=1.5, 
            arrowwidth=2,
            arrowhead=2
        )
    
    fig.update_layout(
        title='Eigenvector Constellation Plot',
        xaxis_title='Re(z)',
        yaxis_title='Im(z)',
        showlegend=False
    )
    
    for n in range(N):
        fig.add_annotation(
            x=x[sequential_order_indices[n]], 
            y=y[sequential_order_indices[n]],
            text=sequential_order_columns[n],
            font=dict(size=20),
            showarrow=False
        )
    
    return fig


def plot_leadership(df, lead_lag_df, leading_eigenvector, sequential_order_dict, accumulated_oriented_area):
    """
    Plots the leadership analysis.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        lead_lag_df (pd.DataFrame): The DataFrame containing the lead-lag matrix.
        leading_eigenvector (np.ndarray): The leading eigenvector.
        sequential_order_dict (dict): A dictionary mapping indices to column names.
        accumulated_oriented_area (pd.DataFrame): The DataFrame containing the accumulated oriented area.

    Returns:
        fig (plotly.graph_objects.Figure): The plotly figure object.
    """
    fig1 = constellation_plot(leading_eigenvector, sequential_order_dict)
    fig2 = plot_heatmap(lead_lag_df)
    
    fig = make_subplots(rows=3, cols=2,
                        specs=[[{}, {}],
                          [{"colspan": 2}, None],
                          [{"colspan": 2}, None]], shared_xaxes=True, subplot_titles=['Constellation Plot', 'Lead Lag Matrix', 'Processed Data', 'Accumulated Oriented Area'])
    
    fig.add_trace(fig1.data[0], row=1, col=1)
    annotations = fig1.layout.annotations
    for ann in annotations:
        fig.add_annotation(ann)
    fig.update_layout(xaxis_title='Re(z)', yaxis_title='Im(z)') 
    
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    annotations = fig2.layout.annotations
    for ann in annotations:
        fig.add_annotation(ann, row=1, col=2)
    
    fig3 = plot_data(df)
    for trace in fig3.data:
        fig.add_trace(trace, row=2, col=1)
    
    t = np.arange(0, len(df))
    for col in accumulated_oriented_area.columns:
        fig.add_trace(go.Scatter(x=t, y=accumulated_oriented_area[col].to_numpy().reshape(-1,), 
                         name=f"{col.split(sep='Accumulated Oriented Area')[0]}", line=dict(width=1.5)), row=3, col=1)
    
    plot(fig)
    return fig

