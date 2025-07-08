import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re

# Configuration de la page
st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .risk-high { border-left-color: #dc3545; }
    .risk-medium { border-left-color: #ffc107; }
    .risk-low { border-left-color: #28a745; }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger et traiter les données
@st.cache_data
def load_data():
    """Charge et traite les données d'analyse"""
    # Simulation de données basée sur ton output
    sample_data = {
        'articles': [
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
                'entities': {'brands': [], 'regions': [], 'products': []},
                'summary': 'Early summer retail sales opportunity discussion focusing on strategic adaptations.'
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
                'entities': {'brands': ['Trader Joe s'], 'regions': [], 'products': []},
                'summary': 'Strategic analysis of Trader Joe s adjacent store opening strategy.'
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
                'entities': {'brands': [], 'regions': [], 'products': []},
                'summary': 'Strategies for attracting value-seeking consumers beyond price competition.'
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
                'entities': {'brands': ['Amazon', 'Shein', 'Temu'], 'regions': [], 'products': ['fast fashion']},
                'summary': 'Amazon positioning to capture market share from declining fast-fashion platforms.'
            }
        ]
    }
    
    return pd.DataFrame(sample_data['articles'])

# Fonction pour créer des métriques
def create_metrics(df):
    """Crée des métriques clés pour le dashboard"""
    total_articles = len(df)
    avg_competitive_impact = df['competitive_impact'].mean()
    high_priority_articles = len(df[df['strategic_importance'] >= 7])
    positive_sentiment_pct = (df['sentiment'] == 'Positive').mean() * 100
    
    return total_articles, avg_competitive_impact, high_priority_articles, positive_sentiment_pct

# Interface principale
def main():
    st.title("🏪 Système d'Intelligence Retail Prédictive")
    st.markdown("### Analyse automatisée de la veille concurrentielle retail/FMCG")
    
    # Chargement des données
    df = load_data()
    
    # Sidebar pour les filtres
    st.sidebar.header("🔍 Filtres d'Analyse")
    
    # Filtre par catégorie
    categories = ['Toutes'] + list(df['category'].unique())
    selected_category = st.sidebar.selectbox("Catégorie", categories)
    
    # Filtre par sentiment
    sentiments = ['Tous'] + list(df['sentiment'].unique())
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Filtre par période
    date_range = st.sidebar.date_input(
        "Période d'analyse",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    # Application des filtres
    filtered_df = df.copy()
    if selected_category != 'Toutes':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_sentiment != 'Tous':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    
    # Métriques principales
    st.header("📊 Métriques Clés")
    total_articles, avg_competitive_impact, high_priority_articles, positive_sentiment_pct = create_metrics(filtered_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles Analysés", total_articles)
    
    with col2:
        st.metric("Impact Concurrentiel Moyen", f"{avg_competitive_impact:.1f}/10")
    
    with col3:
        st.metric("Articles Haute Priorité", high_priority_articles)
    
    with col4:
        st.metric("Sentiment Positif", f"{positive_sentiment_pct:.1f}%")
    
    # Graphiques d'analyse
    st.header("📈 Analyse des Tendances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition par catégorie
        fig_cat = px.pie(
            filtered_df, 
            names='category', 
            title="Répartition par Catégorie",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Impact vs Opportunité
        fig_impact = px.scatter(
            filtered_df,
            x='competitive_impact',
            y='market_opportunity',
            size='strategic_importance',
            color='sentiment',
            hover_data=['title'],
            title="Impact Concurrentiel vs Opportunité Marché"
        )
        st.plotly_chart(fig_impact, use_container_width=True)
    
    # Matrice des risques
    st.header("⚠️ Matrice des Risques et Opportunités")
    
    # Création de la matrice
    risk_matrix = go.Figure()
    
    for idx, row in filtered_df.iterrows():
        risk_matrix.add_trace(go.Scatter(
            x=[row['supply_chain_impact']],
            y=[row['competitive_impact']],
            mode='markers+text',
            text=[row['title'][:30] + '...'],
            textposition='top center',
            marker=dict(
                size=row['strategic_importance'] * 3,
                color=row['market_opportunity'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Opportunité Marché")
            ),
            name=row['category']
        ))
    
    risk_matrix.update_layout(
        title="Matrice Risques Supply Chain vs Impact Concurrentiel",
        xaxis_title="Impact Supply Chain",
        yaxis_title="Impact Concurrentiel",
        showlegend=False
    )
    
    st.plotly_chart(risk_matrix, use_container_width=True)
    
    # Insights automatiques
    st.header("🧠 Insights Automatiques")
    
    # Génération d'insights basés sur l'analyse
    insights = []
    
    # Insight sur les tendances
    if len(filtered_df) > 0:
        top_category = filtered_df['category'].value_counts().index[0]
        insights.append({
            'type': 'trend',
            'title': 'Tendance Dominante',
            'content': f"La catégorie '{top_category}' représente la majorité des articles analysés, suggérant une focus stratégique sur ce domaine.",
            'priority': 'medium'
        })
    
    # Insight sur les risques
    high_risk_articles = filtered_df[filtered_df['supply_chain_impact'] >= 7]
    if len(high_risk_articles) > 0:
        insights.append({
            'type': 'risk',
            'title': 'Alerte Risque Supply Chain',
            'content': f"{len(high_risk_articles)} article(s) signalent des risques élevés pour la supply chain. Action immédiate recommandée.",
            'priority': 'high'
        })
    
    # Insight sur les opportunités
    high_opportunity = filtered_df[filtered_df['market_opportunity'] >= 8]
    if len(high_opportunity) > 0:
        insights.append({
            'type': 'opportunity',
            'title': 'Opportunité Marché Détectée',
            'content': f"{len(high_opportunity)} article(s) révèlent des opportunités de marché significatives à exploiter rapidement.",
            'priority': 'high'
        })
    
    # Affichage des insights
    for insight in insights:
        priority_class = f"risk-{insight['priority']}"
        st.markdown(f"""
        <div class="insight-card {priority_class}">
            <h4>{insight['title']}</h4>
            <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tableau détaillé des articles
    st.header("📋 Analyse Détaillée des Articles")
    
    # Configuration des colonnes à afficher
    columns_to_show = ['title', 'source', 'publish_date', 'category', 'sentiment', 
                      'competitive_impact', 'market_opportunity', 'strategic_importance']
    
    # Mise en forme du tableau
    display_df = filtered_df[columns_to_show].copy()
    display_df['publish_date'] = pd.to_datetime(display_df['publish_date']).dt.strftime('%d/%m/%Y')
    
    # Affichage avec formatage conditionnel
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'title': st.column_config.TextColumn('Titre', width='large'),
            'source': st.column_config.TextColumn('Source', width='medium'),
            'publish_date': st.column_config.TextColumn('Date', width='small'),
            'category': st.column_config.TextColumn('Catégorie', width='medium'),
            'sentiment': st.column_config.TextColumn('Sentiment', width='small'),
            'competitive_impact': st.column_config.ProgressColumn('Impact Concurrentiel', min_value=0, max_value=10),
            'market_opportunity': st.column_config.ProgressColumn('Opportunité Marché', min_value=0, max_value=10),
            'strategic_importance': st.column_config.ProgressColumn('Importance Stratégique', min_value=0, max_value=10)
        }
    )
    
    # Footer avec informations système
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Système d'Intelligence Retail Prédictive - Alimenté par LLMs Multi-Modèles<br>
        📊 Analyse automatisée de la veille concurrentielle retail/FMCG<br>
        ⚡ Mise à jour en temps réel - Dernière analyse: {}
        </p>
    </div>
    """.format(datetime.now().strftime('%d/%m/%Y %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()