# app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Optional: For real API call
# from openai import OpenAI

# Initialisation
st.set_page_config(page_title="Retail Intelligence App", layout="wide")
load_dotenv()

# Title
st.title("📊 Système d'Intelligence Retail Prédictive")
st.markdown("Analyse de veille concurrentielle, supply chain et tendances dans le secteur FMCG/Retail.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload ton fichier CSV contenant les articles analysés", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display raw data
    with st.expander("🧾 Aperçu brut des données"):
        st.dataframe(df.head(10), use_container_width=True)

    # Sélection d'article
    article_titles = df["title"].tolist()
    selected_title = st.selectbox("🗂 Choisis un article pour voir l'analyse détaillée", article_titles)

    article = df[df["title"] == selected_title].iloc[0]

    st.markdown("---")
    st.subheader(f"📰 {article['title']}")
    st.markdown(f"**URL**: [Lien vers l'article]({article['url']})")
    st.markdown(f"**Date de publication**: `{article['publish_date']}`")

    # Analyse structurée
    st.markdown("### 🧠 Analyse Générative")
    st.markdown(f"**Résumé**: {article.get('summary', 'Non disponible')}")
    st.markdown(f"**Catégories**: {article.get('category', 'Non classé')}")

    with st.expander("🔍 Extraction d'entités"):
        st.markdown(f"- **Entreprises**: {article.get('brands', 'N/A')}")
        st.markdown(f"- **Produits**: {article.get('products', 'N/A')}")
        st.markdown(f"- **Régions**: {article.get('regions', 'N/A')}")
        st.markdown(f"- **Personnalités**: {article.get('figures', 'N/A')}")
        st.markdown(f"- **Données financières**: {article.get('financial_data', 'N/A')}")

    with st.expander("📉 Analyse des risques et tendances"):
        st.markdown(f"**Sentiment Marché**: {article.get('market_sentiment', 'N/A')}")
        st.markdown(f"**Sentiment Concurrence**: {article.get('competitive_sentiment', 'N/A')}")
        st.markdown(f"**Sentiment Supply Chain**: {article.get('supply_chain_sentiment', 'N/A')}")
        st.markdown(f"**Tendance Émergente**: {article.get('trend', 'N/A')}")

    with st.expander("📊 Scoring d’Impact Business"):
        st.write(f"**Impact Concurrentiel**: {article.get('competitive_impact', 'N/A')}/10")
        st.write(f"**Impact Supply Chain**: {article.get('supply_chain_impact', 'N/A')}/10")
        st.write(f"**Opportunité Marché**: {article.get('market_opportunity', 'N/A')}/10")
        st.write(f"**Importance Stratégique**: {article.get('strategic_importance', 'N/A')}/10")

    # Rapport téléchargeable
    if st.button("📥 Télécharger le rapport au format texte"):
        newsletter = f"""
        Weekly Retail Intelligence Report

        TITLE: {article['title']}
        URL: {article['url']}
        SUMMARY: {article.get('summary', '')}
        CATEGORY: {article.get('category', '')}
        BRANDS: {article.get('brands', '')}
        SENTIMENT MARKET: {article.get('market_sentiment', '')}
        COMPETITIVE IMPACT: {article.get('competitive_impact', '')}/10

        Strategic Recommendation: {article.get('recommendation_priority', 'N/A')}
        """

        st.download_button("📄 Télécharger Newsletter", newsletter, file_name="retail_report.txt")

    st.success("✅ Analyse terminée. Tu peux passer à un autre article ou exporter.")

else:
    st.info("📌 Merci d'importer un fichier CSV pour commencer l'analyse.")

# Footer
st.markdown("---")
st.caption("Application développée par Lionel • Pour le poste Dev AI chez Markant 🇩🇪")
