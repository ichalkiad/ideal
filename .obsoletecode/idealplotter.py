import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def create_2d_ideal_points_plot(data=None, x_col='dimension_1', y_col='dimension_2', 
                               user_type_col='user_type', title='2D Ideal Points Space',
                               x_label='First Dimension (e.g., Economic)', 
                               y_label='Second Dimension (e.g., Social)'):
    """
    Create a 2D scatter plot of ideal points for users and main users in political space.
    
    Parameters:
    - data: pandas DataFrame with ideal points data
    - x_col: column name for first dimension (x-axis) ideal points
    - y_col: column name for second dimension (y-axis) ideal points
    - user_type_col: column name indicating user type
    - title: plot title
    - x_label: label for x-axis (first dimension)
    - y_label: label for y-axis (second dimension)
    
    Returns:
    - plotly figure object
    """
    
    # Generate realistic political science sample data if none provided
    if data is None:
        np.random.seed(42)
        n_users = 300
        n_main_users = 75
        
        # Generate ideal points for regular users (more dispersed across political space)
        users_dim1 = np.random.normal(0, 1.2, n_users)  # Economic dimension
        users_dim2 = np.random.normal(0, 1.0, n_users)  # Social dimension
        
        # Generate ideal points for main users (more concentrated, slightly different positioning)
        main_users_dim1 = np.random.normal(0.4, 0.8, n_main_users)  # Slightly more economically conservative
        main_users_dim2 = np.random.normal(-0.3, 0.6, n_main_users)  # Slightly more socially conservative
        
        # Create DataFrame
        data = pd.DataFrame({
            x_col: np.concatenate([users_dim1, main_users_dim1]),
            y_col: np.concatenate([users_dim2, main_users_dim2]),
            user_type_col: ['User'] * n_users + ['Main User'] * n_main_users,
            'user_id': range(n_users + n_main_users)
        })
    
    # Define colors and markers for different user types
    color_map = {
        'User': '#2E86AB',           # Blue
        'Main User': '#A23B72',      # Purple-red
        'Regular User': '#2E86AB',   # Alternative naming
        'Power User': '#A23B72',     # Alternative naming
        'Elite': '#F18F01',          # Orange for elites
        'Citizen': '#2E86AB'         # Blue for citizens
    }
    
    marker_map = {
        'User': 'circle',
        'Main User': 'diamond',
        'Regular User': 'circle',
        'Power User': 'diamond',
        'Elite': 'square',
        'Citizen': 'circle'
    }
    
    # Create the 2D scatter plot
    fig = go.Figure()
    
    for user_type in data[user_type_col].unique():
        subset = data[data[user_type_col] == user_type]
        
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            name=user_type,
            marker=dict(
                symbol=marker_map.get(user_type, 'circle'),
                size=10 if user_type in ['Main User', 'Power User', 'Elite'] else 8,
                color=color_map.get(user_type, '#2E86AB'),
                opacity=0.75,
                line=dict(width=1.5, color='white')
            ),
            hovertemplate=
                f'<b>{user_type}</b><br>' +
                f'{x_label}: %{{x:.3f}}<br>' +
                f'{y_label}: %{{y:.3f}}<br>' +
                '<extra></extra>'
        ))
    
    # Add quadrant lines (reference lines at origin)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, 
                  annotation_text="", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, 
                  annotation_text="", line_width=1)
    
    # Add quadrant labels for political interpretation
    max_x = data[x_col].max() * 0.9
    max_y = data[y_col].max() * 0.9
    min_x = data[x_col].min() * 0.9
    min_y = data[y_col].min() * 0.9
    
    # Add subtle quadrant annotations
    fig.add_annotation(x=max_x, y=max_y, text="Liberal/Progressive", 
                      showarrow=False, font=dict(size=10, color="gray"),
                      xanchor="right", yanchor="top")
    fig.add_annotation(x=min_x, y=max_y, text="Conservative/Progressive", 
                      showarrow=False, font=dict(size=10, color="gray"),
                      xanchor="left", yanchor="top")
    fig.add_annotation(x=min_x, y=min_y, text="Conservative/Traditional", 
                      showarrow=False, font=dict(size=10, color="gray"),
                      xanchor="left", yanchor="bottom")
    fig.add_annotation(x=max_x, y=min_y, text="Liberal/Traditional", 
                      showarrow=False, font=dict(size=10, color="gray"),
                      xanchor="right", yanchor="bottom")
    
    # Update layout for publication quality
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial'}
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=900,
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode='closest',
        font=dict(family="Arial", size=12)
    )
    
    # Enhance axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False,
        title_font_size=14
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False,
        title_font_size=14
    )
    
    return fig

def create_enhanced_2d_plot(data=None, x_col='dimension_1', y_col='dimension_2', 
                           user_type_col='user_type', size_col=None, 
                           title='Enhanced 2D Ideal Points Space'):
    """
    Create an enhanced 2D plot with optional sizing by another variable (e.g., influence, activity).
    """
    
    # Generate sample data if none provided
    if data is None:
        np.random.seed(42)
        n_users = 300
        n_main_users = 75
        
        users_dim1 = np.random.normal(0, 1.2, n_users)
        users_dim2 = np.random.normal(0, 1.0, n_users)
        main_users_dim1 = np.random.normal(0.4, 0.8, n_main_users)
        main_users_dim2 = np.random.normal(-0.3, 0.6, n_main_users)
        
        # Add influence/activity measure
        users_influence = np.random.exponential(2, n_users)
        main_users_influence = np.random.exponential(5, n_main_users)
        
        data = pd.DataFrame({
            x_col: np.concatenate([users_dim1, main_users_dim1]),
            y_col: np.concatenate([users_dim2, main_users_dim2]),
            user_type_col: ['User'] * n_users + ['Main User'] * n_main_users,
            'influence': np.concatenate([users_influence, main_users_influence]),
            'user_id': range(n_users + n_main_users)
        })
        
        if size_col is None:
            size_col = 'influence'
    
    # Create enhanced scatter plot
    fig = px.scatter(
        data, 
        x=x_col, 
        y=y_col,
        color=user_type_col,
        size=size_col if size_col in data.columns else None,
        symbol=user_type_col,
        title=title,
        color_discrete_map={
            'User': '#2E86AB',
            'Main User': '#A23B72'
        },
        symbol_map={
            'User': 'circle',
            'Main User': 'diamond'
        }
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        width=900,
        height=700,
        font=dict(family="Arial", size=12),
        xaxis_title="First Dimension (Economic)",
        yaxis_title="Second Dimension (Social)"
    )
    
    return fig

def create_density_overlay_plot(data=None, x_col='dimension_1', y_col='dimension_2', 
                               user_type_col='user_type', title='2D Ideal Points with Density'):
    """
    Create a 2D plot with density contours overlaid on scatter points.
    """
    
    # Generate sample data if none provided
    if data is None:
        np.random.seed(42)
        n_users = 300
        n_main_users = 75
        
        users_dim1 = np.random.normal(0, 1.2, n_users)
        users_dim2 = np.random.normal(0, 1.0, n_users)
        main_users_dim1 = np.random.normal(0.4, 0.8, n_main_users)
        main_users_dim2 = np.random.normal(-0.3, 0.6, n_main_users)
        
        data = pd.DataFrame({
            x_col: np.concatenate([users_dim1, main_users_dim1]),
            y_col: np.concatenate([users_dim2, main_users_dim2]),
            user_type_col: ['User'] * n_users + ['Main User'] * n_main_users,
            'user_id': range(n_users + n_main_users)
        })
    
    # Create subplots to combine scatter and density
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=1, cols=1)
    
    # Add scatter points for each user type
    colors = {'User': '#2E86AB', 'Main User': '#A23B72'}
    symbols = {'User': 'circle', 'Main User': 'diamond'}
    
    for user_type in data[user_type_col].unique():
        subset = data[data[user_type_col] == user_type]
        
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            name=user_type,
            marker=dict(
                symbol=symbols.get(user_type, 'circle'),
                size=8,
                color=colors.get(user_type, '#2E86AB'),
                opacity=0.6,
                line=dict(width=1, color='white')
            )
        ))
        
        # Add density contour for each user type
        fig.add_trace(go.Histogram2dContour(
            x=subset[x_col],
            y=subset[y_col],
            name=f'{user_type} Density',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors.get(user_type, '#2E86AB')]],
            showscale=False,
            opacity=0.3,
            line=dict(width=1, color=colors.get(user_type, '#2E86AB'))
        ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="First Dimension (Economic)",
        yaxis_title="Second Dimension (Social)",
        template='plotly_white',
        width=900,
        height=700,
        font=dict(family="Arial", size=12)
    )
    
    return fig

def create_political_compass_style(data=None, x_col='dimension_1', y_col='dimension_2', 
                                  user_type_col='user_type', 
                                  title='Political Compass Style Ideal Points'):
    """
    Create a political compass style visualization with colored quadrants.
    """
    
    # Generate sample data if none provided
    if data is None:
        np.random.seed(42)
        n_users = 300
        n_main_users = 75
        
        users_dim1 = np.random.normal(0, 1.2, n_users)
        users_dim2 = np.random.normal(0, 1.0, n_users)
        main_users_dim1 = np.random.normal(0.4, 0.8, n_main_users)
        main_users_dim2 = np.random.normal(-0.3, 0.6, n_main_users)
        
        data = pd.DataFrame({
            x_col: np.concatenate([users_dim1, main_users_dim1]),
            y_col: np.concatenate([users_dim2, main_users_dim2]),
            user_type_col: ['User'] * n_users + ['Main User'] * n_main_users
        })
    
    fig = go.Figure()
    
    # Add colored quadrant backgrounds
    x_range = [data[x_col].min() - 0.5, data[x_col].max() + 0.5]
    y_range = [data[y_col].min() - 0.5, data[y_col].max() + 0.5]
    
    # Quadrant colors (very subtle)
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_range[1], y1=y_range[1],
                  fillcolor="lightblue", opacity=0.1, line=dict(width=0))
    fig.add_shape(type="rect", x0=x_range[0], y0=0, x1=0, y1=y_range[1],
                  fillcolor="lightcoral", opacity=0.1, line=dict(width=0))
    fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=0, y1=0,
                  fillcolor="lightgreen", opacity=0.1, line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=y_range[0], x1=x_range[1], y1=0,
                  fillcolor="lightyellow", opacity=0.1, line=dict(width=0))
    
    # Add scatter points
    colors = {'User': '#2E86AB', 'Main User': '#A23B72'}
    symbols = {'User': 'circle', 'Main User': 'diamond'}
    
    for user_type in data[user_type_col].unique():
        subset = data[data[user_type_col] == user_type]
        
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            name=user_type,
            marker=dict(
                symbol=symbols.get(user_type, 'circle'),
                size=10 if user_type == 'Main User' else 8,
                color=colors.get(user_type, '#2E86AB'),
                opacity=0.8,
                line=dict(width=1.5, color='white')
            )
        ))
    
    # Add axis lines
    fig.add_hline(y=0, line_color="black", line_width=2, opacity=0.8)
    fig.add_vline(x=0, line_color="black", line_width=2, opacity=0.8)
    
    # Add quadrant labels
    fig.add_annotation(x=x_range[1]*0.8, y=y_range[1]*0.9, 
                      text="Liberal<br/>Progressive", 
                      showarrow=False, font=dict(size=12, color="darkblue"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="darkblue")
    fig.add_annotation(x=x_range[0]*0.8, y=y_range[1]*0.9, 
                      text="Conservative<br/>Progressive", 
                      showarrow=False, font=dict(size=12, color="darkred"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="darkred")
    fig.add_annotation(x=x_range[0]*0.8, y=y_range[0]*0.9, 
                      text="Conservative<br/>Traditional", 
                      showarrow=False, font=dict(size=12, color="darkgreen"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="darkgreen")
    fig.add_annotation(x=x_range[1]*0.8, y=y_range[0]*0.9, 
                      text="Liberal<br/>Traditional", 
                      showarrow=False, font=dict(size=12, color="darkorange"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="darkorange")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Economic Dimension (Left ← → Right)",
        yaxis_title="Social Dimension (Traditional ← → Progressive)",
        template='plotly_white',
        width=900,
        height=700,
        font=dict(family="Arial", size=12),
        xaxis=dict(range=x_range, zeroline=False),
        yaxis=dict(range=y_range, zeroline=False)
    )
    
    return fig

# Example usage and demonstration
if __name__ == "__main__":
    print("Generating 2D ideal points visualizations...")
    
    # Example 1: Basic 2D ideal points plot
    fig_2d = create_2d_ideal_points_plot(
        title="2D Ideal Points Space: Users vs Main Users"
    )
    fig_2d.show()
    
    # Example 2: Enhanced plot with sizing
    fig_enhanced = create_enhanced_2d_plot(
        title="Enhanced 2D Ideal Points (Size = Influence)"
    )
    fig_enhanced.show()
    
    # Example 3: Density overlay
    fig_density = create_density_overlay_plot(
        title="2D Ideal Points with Density Contours"
    )
    fig_density.show()
    
    # Example 4: Political compass style
    fig_compass = create_political_compass_style(
        title="Political Compass Style Visualization"
    )
    fig_compass.show()
    
    # Example with custom data
    print("\nTo use with your own data:")
    print("df = pd.read_csv('your_data.csv')")
    print("fig = create_2d_ideal_points_plot(")
    print("    data=df,")
    print("    x_col='economic_dimension',")
    print("    y_col='social_dimension',")
    print("    user_type_col='actor_type',")
    print("    title='Your Study Title',")
    print("    x_label='Economic Ideology (Liberal-Conservative)',")
    print("    y_label='Social Ideology (Traditional-Progressive)'")
    print(")")
    print("fig.show()")
    print("# fig.write_html('ideal_points.html')")
    print("# fig.write_image('ideal_points.png')  # Requires kaleido package")
    
    # Sample custom data
    custom_data = pd.DataFrame({
        'economic_pos': np.random.normal(0, 1, 200),
        'social_pos': np.random.normal(0, 1, 200),
        'actor_type': np.random.choice(['Citizen', 'Elite'], 200, p=[0.8, 0.2]),
        'influence_score': np.random.exponential(1, 200)
    })
    
    print("\nExample with custom data:")
    fig_custom = create_2d_ideal_points_plot(
        data=custom_data,
        x_col='economic_pos',
        y_col='social_pos',
        user_type_col='actor_type',
        title='Custom Political Space Analysis',
        x_label='Economic Position (Left-Right)',
        y_label='Social Position (Conservative-Liberal)'
    )
    fig_custom.show()