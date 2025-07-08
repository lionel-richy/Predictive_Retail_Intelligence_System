import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .model-comparison-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .risk-high { 
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
    }
    .risk-medium { 
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .risk-low { 
        border-left-color: #28a745;
        background: linear-gradient(135deg, #e6f7e6 0%, #ccf2cc 100%);
    }
    
    .model-tab {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .model-tab:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .comparison-metrics {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .comparison-metric {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .footer {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data with different models
@st.cache_data
def load_data():
    """Loads and processes analysis data from different models"""
    
    # GPT-4o Mini data
    gpt4o_mini_data = [
        {
            'title': 'Is Early Summer Becoming a Bigger Retail Sales Opportunity?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-03',
            'category': 'market_expansion',
            'sentiment': 'Positive',
            'competitive_impact': 7,
            'supply_chain_impact': 5,
            'market_opportunity': 8,
            'strategic_importance': 7,
            'model': 'GPT-4o Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Early summer retail sales opportunity discussion focusing on strategic adaptations.',
            'recommendation_priority': 'High',
            'confidence_score': 0.85
        },
        {
            'title': 'Should Trader Joe s Open Stores Next To Each Other?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'market_expansion',
            'sentiment': 'Neutral',
            'competitive_impact': 6,
            'supply_chain_impact': 4,
            'market_opportunity': 5,
            'strategic_importance': 6,
            'model': 'GPT-4o Mini',
            'entities': {'brands': ['Trader Joe s'], 'regions': [], 'products': []},
            'summary': 'Strategic analysis of Trader Joe s adjacent store opening strategy.',
            'recommendation_priority': 'Medium',
            'confidence_score': 0.78
        },
        {
            'title': 'How Can Retailers Best Attract Value-Seeking Consumers?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'consumer_behavior',
            'sentiment': 'Positive',
            'competitive_impact': 8,
            'supply_chain_impact': 5,
            'market_opportunity': 7,
            'strategic_importance': 8,
            'model': 'GPT-4o Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Strategies for attracting value-seeking consumers beyond price competition.',
            'recommendation_priority': 'High',
            'confidence_score': 0.92
        },
        {
            'title': 'Customers are ditching Shein and Temu. Can Amazon win them over?',
            'source': 'retaildive.com',
            'publish_date': '2025-07-01',
            'category': 'competitive_intelligence',
            'sentiment': 'Positive',
            'competitive_impact': 9,
            'supply_chain_impact': 6,
            'market_opportunity': 8,
            'strategic_importance': 9,
            'model': 'GPT-4o Mini',
            'entities': {'brands': ['Amazon', 'Shein', 'Temu'], 'ieno':None, 'products': ['fast fashion']},
            'summary': 'Amazon positioning to capture market share from declining fast-fashion platforms.',
            'recommendation_priority': 'High',
            'confidence_score': 0.89
        },
        {
            'title': '8 retail trends to watch in 2025',
            'source': 'retaildive.com',
            'publish_date': '2025-01-07',
            'category': 'technology_retail',
            'sentiment': 'Positive',
            'competitive_impact': 8,
            'supply_chain_impact': 5,
            'market_opportunity': 7,
            'strategic_importance': 8,
            'model': 'GPT-4o Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Key retail trends for 2025 including AI and sustainability initiatives.',
            'recommendation_priority': 'High',
            'confidence_score': 0.87
        }
    ]
    
    # GPT-4.1 Mini data
    gpt41_mini_data = [
        {
            'title': 'Is Early Summer Becoming a Bigger Retail Sales Opportunity?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-03',
            'category': 'pricing_strategy',
            'sentiment': 'Positive',
            'competitive_impact': 6,
            'supply_chain_impact': 2,
            'market_opportunity': 7,
            'strategic_importance': 6,
            'model': 'GPT-4.1 Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Early summer emerging as valuable sales period with growth potential.',
            'recommendation_priority': 'Medium',
            'confidence_score': 0.82
        },
        {
            'title': 'Should Trader Joe s Open Stores Next To Each Other?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'competitive_intelligence',
            'sentiment': 'Neutral',
            'competitive_impact': 5,
            'supply_chain_impact': 1,
            'market_opportunity': 4,
            'strategic_importance': 4,
            'model': 'GPT-4.1 Mini',
            'entities': {'brands': ['Trader Joe s'], 'regions': [], 'products': []},
            'summary': 'Strategic debate on Trader Joe s clustering stores with mixed outcomes.',
            'recommendation_priority': 'Low',
            'confidence_score': 0.71
        },
        {
            'title': 'How Can Retailers Best Attract Value-Seeking Consumers?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'consumer_behavior',
            'sentiment': 'Positive',
            'competitive_impact': 8,
            'supply_chain_impact': 2,
            'market_opportunity': 7,
            'strategic_importance': 8,
            'model': 'GPT-4.1 Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Strategies for engaging value-conscious consumers beyond pricing.',
            'recommendation_priority': 'High',
            'confidence_score': 0.90
        },
        {
            'title': 'Customers are ditching Shein and Temu. Can Amazon win them over?',
            'source': 'retaildive.com',
            'publish_date': '2025-07-01',
            'category': 'competitive_intelligence',
            'sentiment': 'Positive',
            'competitive_impact': 9,
            'supply_chain_impact': 7,
            'market_opportunity': 8,
            'strategic_importance': 9,
            'model': 'GPT-4.1 Mini',
            'entities': {'brands': ['Amazon', 'Shein', 'Temu'], 'regions': [], 'products': ['fast fashion']},
            'summary': 'Amazon positioned to benefit from Shein/Temu customer migration.',
            'recommendation_priority': 'High',
            'confidence_score': 0.91
        },
        {
            'title': '8 retail trends to watch in 2025',
            'source': 'retaildive.com',
            'publish_date': '2025-01-07',
            'category': 'technology_retail',
            'sentiment': 'Positive',
            'competitive_impact': 8,
            'supply_chain_impact': 6,
            'market_opportunity': 8,
            'strategic_importance': 9,
            'model': 'GPT-4.1 Mini',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Eight key retail trends highlighting technology and sustainability.',
            'recommendation_priority': 'High',
            'confidence_score': 0.88
        }
    ]
    
    # Llama 4 data
    llama4_data = [
        {
            'title': 'Is Early Summer Becoming a Bigger Retail Sales Opportunity?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-03',
            'category': 'consumer_trends',
            'sentiment': 'Positive',
            'competitive_impact': 7,
            'supply_chain_impact': 5,
            'market_opportunity': 8,
            'strategic_importance': 7,
            'model': 'Llama 4',
            'entities': {'brands': [], 'regions': [], 'products': ['summer-related products']},
            'summary': 'Changing consumer behaviors make early summer a significant retail sales opportunity.',
            'recommendation_priority': 'Medium',
            'confidence_score': 0.84
        },
        {
            'title': 'Should Trader Joe s Open Stores Next To Each Other?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'market_expansion',
            'sentiment': 'Neutral',
            'competitive_impact': 8,
            'supply_chain_impact': 3,
            'market_opportunity': 7,
            'strategic_importance': 9,
            'model': 'Llama 4',
            'entities': {'brands': ['Trader Joe s'], 'regions': [], 'products': []},
            'summary': 'Debate on Trader Joe s adjacent stores, weighing market share gains against sales cannibalization.',
            'recommendation_priority': 'High',
            'confidence_score': 0.80
        },
        {
            'title': 'How Can Retailers Best Attract Value-Seeking Consumers?',
            'source': 'retailwire.com',
            'publish_date': '2025-07-01',
            'category': 'consumer_trends',
            'sentiment': 'Positive',
            'competitive_impact': 9,
            'supply_chain_impact': 6,
            'market_opportunity': 9,
            'strategic_importance': 9,
            'model': 'Llama 4',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Retail strategies to attract value-seeking consumers through experience and loyalty programs.',
            'recommendation_priority': 'High',
            'confidence_score': 0.91
        },
        {
            'title': 'Customers are ditching Shein and Temu. Can Amazon win them over?',
            'source': 'retaildive.com',
            'publish_date': '2025-07-01',
            'category': 'competitive_intelligence',
            'sentiment': 'Neutral',
            'competitive_impact': 9,
            'supply_chain_impact': 7,
            'market_opportunity': 8,
            'strategic_importance': 9,
            'model': 'Llama 4',
            'entities': {'brands': ['Amazon', 'Shein', 'Temu'], 'regions': [], 'products': ['fast fashion']},
            'summary': 'Amazon’s opportunity to capture Shein and Temu customers amid tariff impacts.',
            'recommendation_priority': 'High',
            'confidence_score': 0.88
        },
        {
            'title': '8 retail trends to watch in 2025',
            'source': 'retaildive.com',
            'publish_date': '2025-01-07',
            'category': 'technology_retail',
            'sentiment': 'Positive',
            'competitive_impact': 9,
            'supply_chain_impact': 6,
            'market_opportunity': 9,
            'strategic_importance': 9,
            'model': 'Llama 4',
            'entities': {'brands': [], 'regions': [], 'products': []},
            'summary': 'Retail trends for 2025, focusing on omnichannel, sustainability, and AI integration.',
            'recommendation_priority': 'High',
            'confidence_score': 0.89
        }
    ]
    
    # Combine all data
    all_data = gpt4o_mini_data + gpt41_mini_data + llama4_data
    
    return pd.DataFrame(all_data)

# Function to create metrics per model
def create_model_metrics(df, model_name):
    """Creates model-specific metrics"""
    model_df = df[df['model'] == model_name]
    
    if len(model_df) == 0:
        return 0, 0, 0, 0, 0
    
    total_articles = len(model_df)
    avg_competitive_impact = model_df['competitive_impact'].mean()
    high_priority_articles = len(model_df[model_df['strategic_importance'] >= 7])
    positive_sentiment_pct = (model_df['sentiment'] == 'Positive').mean() * 100
    avg_confidence = model_df['confidence_score'].mean()
    
    return total_articles, avg_competitive_impact, high_priority_articles, positive_sentiment_pct, avg_confidence

# Function to create comparison chart
def create_comparison_chart(df):
    """Creates a comparison chart between models"""
    
    # Prepare data for comparison
    comparison_data = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        comparison_data.append({
            'Model': model,
            'Avg_Competitive_Impact': model_df['competitive_impact'].mean(),
            'Avg_Supply_Chain_Impact': model_df['supply_chain_impact'].mean(),
            'Avg_Market_Opportunity': model_df['market_opportunity'].mean(),
            'Avg_Strategic_Importance': model_df['strategic_importance'].mean(),
            'Avg_Confidence': model_df['confidence_score'].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create radar chart
    fig = go.Figure()
    
    categories = ['Avg_Competitive_Impact', 'Avg_Supply_Chain_Impact', 
                 'Avg_Market_Opportunity', 'Avg_Strategic_Importance', 'Avg_Confidence']
    
    for _, row in comparison_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=row['Model'],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Model Performance Comparison"
    )
    
    return fig

# Main interface
def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏪 Predictive Retail Intelligence System</h1>
        <h3>Multi-Model Competitive Analysis</h3>
        <p>Comparison: GPT-4o Mini vs GPT-4.1 Mini vs Llama 4</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df rift= load_data()
    
    # Sidebar for filters
    st.sidebar.header("🔍 Analysis Filters")
    
    # Model selection
    models = ['All'] + list(df['model'].unique())
    selected_model = st.sidebar.selectbox("AI Model", models)
    
    # Category filter
    categories = ['All'] + list(df['category'].unique())
    selected_category = st.sidebar.selectbox("Category", categories)
    
    # Sentiment filter
    sentiments = ['All'] + list(df['sentiment'].unique())
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Display mode
    display_mode = st.sidebar.radio(
        "Display Mode",
        ["Global View", "Model Comparison", "Detailed Analysis"]
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_model != 'All':
        filtered_df = filtered_df[filtered_df['model'] == selected_model]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    
    # Display based on selected mode
    if display_mode == "Global View":
        show_global_view(filtered_df)
    elif display_mode == "Model Comparison":
        show_model_comparison(df)
    else:
        show_detailed_analysis(filtered_df)

def show_global_view(df):
    """Displays the global view"""
    st.header("📊 Global Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyzed Articles", len(df))
    
    with col2:
        avg_impact = df['competitive_impact'].mean()
        st.metric("Avg Competitive Impact", f"{avg_impact:.1f}/10")
    
    with col3:
        high_priority = len(df[df['strategic_importance'] >= 7])
        st.metric("High Priority Articles", high_priority)
    
    with col4:
        positive_pct = (df['sentiment'] == 'Positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cat = px.pie(
            df, 
            names='category', 
            title="Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
 muodist.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        fig_model = px.bar(
            df.groupby('model').size().reset_index(name='count'),
            x='model',
            y='count',
            title="Articles by Model",
            color='model'
        )
        st.plotly_chart(fig_model, use_container_width=True)

def show_model_comparison(df):
    """Displays model comparison"""
    st.header("🔬 Model Comparison")
    
    # Comparison metrics
    st.markdown("""
    <div class="model-comparison-card">
        <h3>📈 Comparative Performance Analysis</h3>
        <p>Detailed comparison between GPT-4o Mini, GPT-4.1 Mini, and Llama 4</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Radar comparison chart
    fig_comparison = create_comparison_chart(df)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed metrics by model
    st.subheader("📊 Detailed Metrics by Model")
    
    models = df['model'].unique()
    cols = st.columns(len(models))
    
    for i, model in enumerate(models):
        with cols[i]:
            st.markdown(f"""
            <div class="model-tab">
                <h4>{model}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            total, avg_comp, high_prio, pos_sent, avg_conf = create_model_metrics(df, model)
            
            st.metric("Articles", total)
            st.metric("Avg Impact", f"{avg_comp:.1f}")
            st.metric("High Priority", high_prio)
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
    
    # Detailed comparison table
    st.subheader("📋 Article-by-Article Comparison")
    
    # Group by title for comparison
    comparison_data = []
    titles = df['title'].unique()
    
    for title in titles:
        article_versions = df[df['title'] == title]
        if len(article_versions) > 1:
            row = {'Title': title[:50] + '...'}
            for model in df['model'].unique():
                model_data = article_versions[article_versions['model'] == model]
                if len(model_data) > 0:
                    row[f'{model} - Impact'] = model_data['competitive_impact'].iloc[0]
                    row[f'{model} - Priority'] = model_data['recommendation_priority'].iloc[0]
                else:
                    row[f'{model} - Impact'] = 'N/A'
                    row[f'{model} - Priority'] = 'N/A'
            # Calculate impact differences between models
            impact_values = [row[f'{model} - Impact'] for model in df['model'].unique() if row[f'{model} - Impact'] != 'N/A']
            if len(impact_values) >= 2:
                row['Max Impact Difference'] = max([abs(impact_values[i] - impact_values[j]) 
                                                   for i in range(len(impact_values)) 
                                                   for j in range(i+1, len(impact_values))])
            else:
                row['Max Impact Difference'] = 'N/A'
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Comparison insights
    st.subheader("🧠 Comparison Insights")
    
    insights = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        other_models = [m for m in df['model'].unique() if m != model]
        for other_model in other_models:
            other_data = df[df['model'] == other_model]
            avg_impact_diff = model_data['competitive_impact'].mean() - other_data['competitive_impact'].mean()
            if abs(avg_impact_diff) > 0.5:
                model_higher = model if avg_impact_diff > 0 else other_model
                insights.append({
                    'type': 'comparison',
                    'title': 'Significant Impact Difference',
                    'content': f"{model_higher} rates competitive impact {abs(avg_impact_diff):.1f} points higher on average than {model if model_higher == other_model else other_model}.",
                    'priority': 'medium'
                })
            avg_conf_diff = model_data['confidence_score'].mean() - other_data['confidence_score'].mean()
            if abs(avg_conf_diff) > 0.05:
                model_higher = model if avg_conf_diff > 0 else other_model
                insights.append({
                    'type': 'confidence',
                    'title': 'Confidence Difference',
                    'content': f"{model_higher} shows a confidence level {abs(avg_conf_diff):.2f} points higher than {model if model_higher == other_model else other_model}.",
                    'priority': 'low'
                })
    
    for insight in insights:
        priority_class = f"risk-{insight['priority']}"
        st.markdown(f"""
        <div class="insight-card {priority_class}">
            <h4>{insight['title']}</h4>
            <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_detailed_analysis(df):
    """Displays detailed analysis"""
    st.header("🔍 Detailed Analysis")
    
    # Advanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation matrix
        numeric_cols = ['competitive_impact', 'supply_chain_impact', 'market_opportunity', 'strategic_importance']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Metrics",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Sentiment analysis by model
        sentiment_model = df.groupby(['model', 'sentiment']).size().reset_index(name='count')
        
        fig_sentiment = px.bar(
            sentiment_model,
            x='model',
            y='count',
            color='sentiment',
            title="Sentiment Distribution by Model",
            barmode='group'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Confidence analysis
    st.subheader("🎯 Confidence Analysis")
    
    fig_conf = px.box(
        df,
        x='model',
        y='confidence_score',
        title="Confidence Score Distribution by Model",
        color='model'
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed data table
    st.subheader("📋 Complete Data")
    
    columns_to_show = ['title', 'model', 'source', 'publish_date', 'category', 'sentiment', 
                      'competitive_impact', 'market_opportunity', 'strategic_importance', 
                      'recommendation_priority', 'confidence_score']
    
    display_df = df[columns_to_show].copy()
    display_df['publish_date'] = pd.to_datetime(display_df['publish_date']).dt.strftime('%m/%d/%Y')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'title': st.column_config.TextColumn('Title', width='large'),
            'model': st.column_config.TextColumn('Model', width='medium'),
            'source': st.column_config.TextColumn('Source', width='medium'),
            'publish_date': st.column_config.TextColumn('Date', width='small'),
            'category': st.column_config.TextColumn('Category', width='medium'),
            'sentiment': st.column_config.TextColumn('Sentiment', width='small'),
            'competitive_impact': st.column_config.ProgressColumn('Competitive Impact', min_value=0, max_value=10),
            'market_opportunity': st.column_config.ProgressColumn('Market Opportunity', min_value=0, max_value=10),
            'strategic_importance': st.column_config.ProgressColumn('Strategic Importance', min_value=0, max_value=10),
            'recommendation_priority': st.column_config.TextColumn('Priority', width='small'),
            'confidence_score': st.column_config.ProgressColumn('Confidence', min_value=0, max_value=1)
        }
    )

# Footer
st.markdown("""
<div class="footer">
    <h3>🤖 Predictive Retail Intelligence System</h3>
    <p>AI-Powered Multi-Model Analysis</p>
    <p>📊 Comparison: GPT-4o Mini vs GPT-4.1 Mini vs Llama 4</p>
    <p>⚡ Real-time Update - Last Analysis: {}</p>
</div>
""".format(datetime.now().strftime('%m/%d/%Y %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()