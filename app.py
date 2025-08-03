#!/usr/bin/env python3
"""
Advanced Crypto Portfolio Dashboard with Streamlit
==================================================

Interactive dashboard for cryptocurrency portfolio analysis with:
- Portfolio optimization & sentiment analysis
- Real-time performance tracking
- Advanced rebalancing strategies
- AI-powered insights with Gemini
- Risk management tools
- Market analysis & predictions

Usage:
    streamlit run crypto_dashboard.py

Requirements:
    pip install streamlit plotly pandas numpy google-generativeai python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class CryptoDashboard:
    """Advanced Crypto Portfolio Dashboard with AI Integration"""
    
    def __init__(self):
        """Initialize dashboard with data loading"""
        self.data_dir = './crypto_analysis_results'
        self.setup_gemini()
        self.load_all_data()
        
    def setup_gemini(self):
        """Setup Gemini AI for portfolio insights"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                self.gemini_available = True
            else:
                self.gemini_available = False
                st.warning("🤖 Add GEMINI_API_KEY to .env for AI insights!")
        except Exception as e:
            self.gemini_available = False
            st.warning(f"🤖 Gemini AI unavailable: {str(e)[:50]}...")
    
    def load_all_data(self):
        """Load all analysis data from files"""
        try:
            # Load portfolio weights
            weights_file = os.path.join(self.data_dir, 'portfolio_weights.csv')
            if os.path.exists(weights_file):
                self.portfolio_weights = pd.read_csv(weights_file)
                self.portfolio_weights['weight_pct'] = self.portfolio_weights['weight'] * 100
            else:
                self.portfolio_weights = pd.DataFrame()
            
            # Load backtest results  
            backtest_file = os.path.join(self.data_dir, 'backtest_results.csv')
            if os.path.exists(backtest_file):
                self.backtest_data = pd.read_csv(backtest_file)
                self.backtest_data['date'] = pd.to_datetime(self.backtest_data['date'])
            else:
                self.backtest_data = pd.DataFrame()
            
            # Load combined data
            combined_file = os.path.join(self.data_dir, 'combined_data.csv')
            if os.path.exists(combined_file):
                self.combined_data = pd.read_csv(combined_file)
                self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
            else:
                self.combined_data = pd.DataFrame()
            
            # Load news data if available
            news_file = os.path.join(self.data_dir, 'news_data.json')
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    self.news_data = json.load(f)
            else:
                self.news_data = []
            
            # Load sentiment data if available
            sentiment_file = os.path.join(self.data_dir, 'sentiment_data.json')
            if os.path.exists(sentiment_file):
                with open(sentiment_file, 'r') as f:
                    self.sentiment_data = json.load(f)
            else:
                self.sentiment_data = []
                
            self.data_loaded = True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            self.data_loaded = False
    
    def get_portfolio_summary(self) -> Dict:
        """Get key portfolio metrics"""
        if self.backtest_data.empty:
            return {}
        
        latest = self.backtest_data.iloc[-1]
        first = self.backtest_data.iloc[0]
        
        total_return = (latest['portfolio_value'] - 1) * 100
        btc_return = (latest['btc_value'] - 1) * 100
        
        returns = self.backtest_data['portfolio_return'].dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        portfolio_values = self.backtest_data['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min() * 100
        
        return {
            'total_return': total_return,
            'btc_return': btc_return,
            'outperformance': total_return - btc_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown),
            'current_value': latest['portfolio_value'],
            'days_analyzed': len(self.backtest_data)
        }
    
    def create_performance_chart(self) -> go.Figure:
        """Create interactive performance chart"""
        if self.backtest_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=self.backtest_data['date'],
            y=self.backtest_data['portfolio_value'],
            mode='lines',
            name='Smart Portfolio',
            line=dict(color='#00D4AA', width=3),
            hovertemplate='<b>Smart Portfolio</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # BTC benchmark
        fig.add_trace(go.Scatter(
            x=self.backtest_data['date'],
            y=self.backtest_data['btc_value'],
            mode='lines',
            name='BTC Benchmark',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            hovertemplate='<b>BTC Benchmark</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Equal weight benchmark
        if 'equal_weight_value' in self.backtest_data.columns:
            fig.add_trace(go.Scatter(
                x=self.backtest_data['date'],
                y=self.backtest_data['equal_weight_value'],
                mode='lines',
                name='Equal Weight',
                line=dict(color='#4ECDC4', width=2, dash='dot'),
                hovertemplate='<b>Equal Weight</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:.3f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '📈 Portfolio Performance Over Time',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        return fig
    
    def create_allocation_chart(self) -> go.Figure:
        """Create portfolio allocation pie chart"""
        if self.portfolio_weights.empty:
            return go.Figure()
        
        # Get top 10 for cleaner visualization
        top_holdings = self.portfolio_weights.nlargest(10, 'weight')
        others_weight = self.portfolio_weights.iloc[10:]['weight'].sum() if len(self.portfolio_weights) > 10 else 0
        
        symbols = top_holdings['symbol'].tolist()
        weights = top_holdings['weight_pct'].tolist()
        
        if others_weight > 0:
            symbols.append('Others')
            weights.append(others_weight * 100)
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=weights,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#FFFFFF', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Allocation: %{percent}<br>' +
                         'Value: %{value:.1f}%<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '🥧 Current Portfolio Allocation',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def create_sentiment_chart(self) -> go.Figure:
        """Create sentiment analysis chart"""
        if self.combined_data.empty:
            return go.Figure()
        
        # Aggregate sentiment by date
        daily_sentiment = self.combined_data.groupby('date')['sentiment_score'].mean().reset_index()
        
        fig = go.Figure()
        
        # Sentiment line
        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['sentiment_score'],
            mode='lines+markers',
            name='Daily Sentiment',
            line=dict(color='#9B59B6', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Sentiment Score</b><br>' +
                         'Date: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add sentiment zones
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Neutral", annotation_position="bottom right")
        fig.add_hline(y=0.1, line_dash="dot", line_color="green", 
                     annotation_text="Positive", annotation_position="top right")
        fig.add_hline(y=-0.1, line_dash="dot", line_color="red", 
                     annotation_text="Negative", annotation_position="bottom right")
        
        fig.update_layout(
            title={
                'text': '🧠 Market Sentiment Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_risk_metrics_chart(self) -> go.Figure:
        """Create risk metrics visualization"""
        if self.backtest_data.empty:
            return go.Figure()
        
        # Calculate rolling metrics
        returns = self.backtest_data['portfolio_return'].rolling(7).std() * np.sqrt(252) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling 7-Day Volatility (%)', 'Daily Returns Distribution'),
            vertical_spacing=0.15
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=self.backtest_data['date'],
                y=returns,
                mode='lines',
                name='7-Day Volatility',
                line=dict(color='#E74C3C', width=2)
            ),
            row=1, col=1
        )
        
        # Returns histogram
        portfolio_returns = self.backtest_data['portfolio_return'] * 100
        fig.add_trace(
            go.Histogram(
                x=portfolio_returns,
                nbinsx=20,
                name='Return Distribution',
                marker_color='#3498DB',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title={
                'text': '⚠️ Risk Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap of portfolio assets"""
        if self.combined_data.empty:
            return go.Figure()
        
        # Pivot data to get returns by symbol
        returns_pivot = self.combined_data.pivot(index='date', columns='symbol', values='daily_return')
        
        if returns_pivot.shape[1] < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = returns_pivot.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '🔗 Asset Correlation Matrix',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def get_latest_sentiment(self) -> Dict:
        """Get latest sentiment analysis"""
        if self.combined_data.empty:
            return {'mean': 0, 'std': 0, 'count': 0}
        
        latest_date = self.combined_data['date'].max()
        latest_sentiment = self.combined_data[self.combined_data['date'] == latest_date]['sentiment_score']
        
        return {
            'mean': latest_sentiment.mean(),
            'std': latest_sentiment.std(),
            'count': len(latest_sentiment),
            'date': latest_date.strftime('%Y-%m-%d')
        }
    
    def get_rebalance_recommendations(self) -> List[Dict]:
        """Generate rebalancing recommendations"""
        if self.portfolio_weights.empty:
            return []
        
        recommendations = []
        
        # Check for overweight positions (>25%)
        overweight = self.portfolio_weights[self.portfolio_weights['weight'] > 0.25]
        for _, asset in overweight.iterrows():
            recommendations.append({
                'type': 'REDUCE',
                'symbol': asset['symbol'],
                'current_weight': asset['weight_pct'],
                'suggested_weight': 25.0,
                'reason': 'Position exceeds maximum allocation limit'
            })
        
        # Check for underperforming assets (negative Sharpe)
        underperforming = self.portfolio_weights[self.portfolio_weights['sharpe_ratio'] < 0]
        for _, asset in underperforming.iterrows():
            if asset['weight_pct'] > 5:  # Only flag if significant allocation
                recommendations.append({
                    'type': 'REVIEW',
                    'symbol': asset['symbol'],
                    'current_weight': asset['weight_pct'],
                    'sharpe_ratio': asset['sharpe_ratio'],
                    'reason': 'Negative risk-adjusted returns'
                })
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_news_headlines(self, limit: int = 5) -> List[Dict]:
        """Get recent news headlines"""
        if not self.news_data:
            return []
        
        # Sort by date and get most recent
        sorted_news = sorted(self.news_data, key=lambda x: x.get('date', ''), reverse=True)
        
        headlines = []
        for article in sorted_news[:limit]:
            headlines.append({
                'title': article.get('title', 'No title').title(),
                'date': article.get('date', 'Unknown'),
                'source': article.get('source', 'Unknown'),
                'sentiment': 'Positive' if 'positive' in article.get('title', '').lower() else 
                           'Negative' if 'negative' in article.get('title', '').lower() else 'Neutral'
            })
        
        return headlines
    
    def ask_gemini(self, question: str) -> str:
        """Get AI insights using Gemini"""
        if not self.gemini_available:
            return "🤖 Gemini AI is not available. Please add GEMINI_API_KEY to your .env file."
        
        try:
            # Prepare context from portfolio data
            portfolio_summary = self.get_portfolio_summary()
            sentiment_summary = self.get_latest_sentiment()
            
            context = f"""
            Portfolio Analysis Context:
            - Total Return: {portfolio_summary.get('total_return', 0):.1f}%
            - Outperformance vs BTC: {portfolio_summary.get('outperformance', 0):.1f}%
            - Sharpe Ratio: {portfolio_summary.get('sharpe_ratio', 0):.2f}
            - Max Drawdown: {portfolio_summary.get('max_drawdown', 0):.1f}%
            - Current Sentiment: {sentiment_summary.get('mean', 0):.3f}
            - Analysis Period: {portfolio_summary.get('days_analyzed', 0)} days
            
            Top Holdings:
            {self.portfolio_weights.head(5)[['symbol', 'weight_pct', 'sharpe_ratio']].to_string(index=False) if not self.portfolio_weights.empty else 'No data available'}
            
            User Question: {question}
            
            Please provide a helpful, analytical response about the portfolio or market sentiment.
            """
            
            response = self.gemini_model.generate_content(context)
            return response.text
            
        except Exception as e:
            return f"🤖 Sorry, I couldn't process your question: {str(e)[:100]}..."

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Crypto Portfolio Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    :root {
        --primary-color: #6c5ce7;
        --secondary-color: #a29bfe;
        --accent-color: #00cec9;
        --dark-color: #2d3436;
        --light-color: #f5f6fa;
        --success-color: #00b894;
        --warning-color: #fdcb6e;
        --danger-color: #d63031;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: var(--light-color);
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
        border-left: 4px solid var(--accent-color);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: var(--dark-color) !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    
    .status-good { color: var(--success-color) !important; }
    .status-bad { color: var(--danger-color) !important; }
    .status-neutral { color: var(--warning-color) !important; }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1.5rem;
    }
    
    .sidebar-title {
        color: var(--primary-color);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
        color: #666;
        font-size: 0.9rem;
        text-align: center;
    }
    
    .stButton>button {
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3);
    }
    
    .stSelectbox, .stTextArea, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    .stExpander {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .stExpander .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--dark-color);
        padding: 1rem 1.5rem;
    }
    
    .stAlert {
        border-radius: 12px;
    }
    
    .plotly-container {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        background: white;
        padding: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = CryptoDashboard()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Crypto Portfolio Dashboard</h1>
        <p>AI-powered portfolio optimization & market sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data is loaded
    if not dashboard.data_loaded:
        st.error("❌ **Data not found!** Please run the crypto analysis system first.")
        st.info("💡 Run: `python crypto_system.py` to generate analysis data")
        st.stop()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">📊 Dashboard Navigation</h2>', unsafe_allow_html=True)
        page = st.selectbox(
            "Choose Analysis:",
            ["📈 Portfolio Overview", "🎯 Performance Analysis", "🧠 Sentiment Analysis", 
             "⚠️ Risk Management", "🔄 Rebalancing", "🤖 AI Insights", "📰 Market News"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sidebar-title">📈 Portfolio Quick Stats</h3>', unsafe_allow_html=True)
        
        portfolio_summary = dashboard.get_portfolio_summary()
        if portfolio_summary:
            st.metric("Portfolio Value", f"${portfolio_summary.get('current_value', 1):.3f}")
            st.metric("Days Analyzed", f"{portfolio_summary.get('days_analyzed', 0)}")
            
            # Performance gauge
            total_return = portfolio_summary.get('total_return', 0)
            if total_return > 10:
                st.success(f"🚀 Strong Performance: +{total_return:.1f}%")
            elif total_return > 0:
                st.info(f"📈 Positive Returns: +{total_return:.1f}%")
            else:
                st.warning(f"📉 Negative Returns: {total_return:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sidebar-title">🛠️ System Info</h3>', unsafe_allow_html=True)
        st.info("💾 Data from: ./crypto_analysis_results/")
        st.info("🔄 Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        # Refresh data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            dashboard.load_all_data()
            st.success("Data refreshed!")
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sidebar-title">💾 Download Data</h3>', unsafe_allow_html=True)
        
        if not dashboard.portfolio_weights.empty:
            csv_data = dashboard.portfolio_weights.to_csv(index=False)
            st.download_button(
                label="📊 Download Portfolio Weights",
                data=csv_data,
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if not dashboard.backtest_data.empty:
            csv_data = dashboard.backtest_data.to_csv(index=False)
            st.download_button(
                label="📈 Download Backtest Results",
                data=csv_data,
                file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        st.markdown("**📊 Crypto Portfolio Dashboard**")
        st.markdown("*Built with Hub Agencies consultancy Murang'a, Kenya*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on selected page
    if page == "📈 Portfolio Overview":
        st.header("📈 Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = portfolio_summary.get('total_return', 0)
            st.metric(
                "Total Return",
                f"{total_return:.1f}%",
                delta=f"vs BTC: {portfolio_summary.get('outperformance', 0):+.1f}%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio_summary.get('sharpe_ratio', 0):.2f}",
                delta="Risk-adjusted return"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{portfolio_summary.get('max_drawdown', 0):.1f}%",
                delta="Maximum loss period"
            )
        
        with col4:
            sentiment = dashboard.get_latest_sentiment()
            sentiment_color = "status-good" if sentiment['mean'] > 0.1 else "status-bad" if sentiment['mean'] < -0.1 else "status-neutral"
            st.metric(
                "Market Sentiment",
                f"{sentiment['mean']:.3f}",
                delta=f"Date: {sentiment.get('date', 'N/A')}"
            )
        
        # Performance chart
        with st.container():
            st.plotly_chart(dashboard.create_performance_chart(), use_container_width=True)
        
        # Portfolio allocation
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.plotly_chart(dashboard.create_allocation_chart(), use_container_width=True)
        
        with col2:
            st.subheader("🏆 Top Holdings")
            if not dashboard.portfolio_weights.empty:
                top_holdings = dashboard.portfolio_weights.head(10)[['symbol', 'weight_pct', 'sharpe_ratio']]
                top_holdings.columns = ['Symbol', 'Weight (%)', 'Sharpe Ratio']
                st.dataframe(
                    top_holdings.style.background_gradient(cmap='Blues'),
                    use_container_width=True,
                    height=500
                )
    
    elif page == "🎯 Performance Analysis":
        st.header("🎯 Detailed Performance Analysis")
        
        # Performance metrics table
        if portfolio_summary:
            metrics_data = {
                'Metric': ['Total Return', 'BTC Return', 'Outperformance', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                'Value': [
                    f"{portfolio_summary['total_return']:.1f}%",
                    f"{portfolio_summary['btc_return']:.1f}%", 
                    f"{portfolio_summary['outperformance']:+.1f}%",
                    f"{portfolio_summary['volatility']:.1f}%",
                    f"{portfolio_summary['sharpe_ratio']:.2f}",
                    f"{portfolio_summary['max_drawdown']:.1f}%"
                ]
            }
            
            st.subheader("📊 Performance Summary")
            st.dataframe(
                pd.DataFrame(metrics_data).style.highlight_max(axis=0, color='#d4edda'),
                use_container_width=True
            )
        
        # Performance chart
        with st.container():
            st.plotly_chart(dashboard.create_performance_chart(), use_container_width=True)
        
        # Risk analysis
        with st.container():
            st.plotly_chart(dashboard.create_risk_metrics_chart(), use_container_width=True)
        
        # Correlation analysis
        with st.container():
            st.plotly_chart(dashboard.create_correlation_heatmap(), use_container_width=True)
    
    elif page == "🧠 Sentiment Analysis":
        st.header("🧠 Market Sentiment Analysis")
        
        # Latest sentiment
        sentiment = dashboard.get_latest_sentiment()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Sentiment", f"{sentiment['mean']:.3f}")
        with col2:
            st.metric("Sentiment Volatility", f"{sentiment['std']:.3f}")
        with col3:
            st.metric("Data Points", f"{sentiment['count']}")
        
        # Sentiment chart
        with st.container():
            st.plotly_chart(dashboard.create_sentiment_chart(), use_container_width=True)
        
        # Sentiment interpretation
        mean_sentiment = sentiment['mean']
        if mean_sentiment > 0.1:
            st.success("🟢 **Positive Market Sentiment** - Market shows optimistic outlook")
        elif mean_sentiment < -0.1:
            st.error("🔴 **Negative Market Sentiment** - Market shows pessimistic outlook")
        else:
            st.info("🟡 **Neutral Market Sentiment** - Market sentiment is balanced")
    
    elif page == "⚠️ Risk Management":
        st.header("⚠️ Risk Management Dashboard")
        
        if portfolio_summary:
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volatility = portfolio_summary['volatility']
                vol_status = "🟢 Low" if volatility < 30 else "🟡 Medium" if volatility < 60 else "🔴 High"
                st.metric("Portfolio Volatility", f"{volatility:.1f}%", delta=vol_status)
            
            with col2:
                sharpe = portfolio_summary['sharpe_ratio']
                sharpe_status = "🟢 Excellent" if sharpe > 2 else "🟡 Good" if sharpe > 1 else "🔴 Poor"
                st.metric("Risk-Adjusted Return", f"{sharpe:.2f}", delta=sharpe_status)
            
            with col3:
                drawdown = portfolio_summary['max_drawdown']
                dd_status = "🟢 Low" if drawdown < 10 else "🟡 Medium" if drawdown < 20 else "🔴 High"
                st.metric("Maximum Drawdown", f"{drawdown:.1f}%", delta=dd_status)
        
        # Risk analysis charts
        with st.container():
            st.plotly_chart(dashboard.create_risk_metrics_chart(), use_container_width=True)
        
        # Portfolio concentration analysis
        if not dashboard.portfolio_weights.empty:
            st.subheader("🎯 Portfolio Concentration Analysis")
            
            max_weight = dashboard.portfolio_weights['weight_pct'].max()
            top_3_weight = dashboard.portfolio_weights.head(3)['weight_pct'].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Largest Position", f"{max_weight:.1f}%")
            with col2:
                st.metric("Top 3 Concentration", f"{top_3_weight:.1f}%")
            
            if max_weight > 25:
                st.warning("⚠️ Portfolio has high concentration risk - consider rebalancing")
            elif top_3_weight > 60:
                st.warning("⚠️ Top 3 positions represent high portfolio concentration")
            else:
                st.success("✅ Portfolio shows good diversification")
    
    elif page == "🔄 Rebalancing":
        st.header("🔄 Portfolio Rebalancing")
        
        # Rebalance date
        rebalance_date = datetime.now().strftime("%Y-%m-%d")
        st.info(f"**Rebalance Date:** {rebalance_date}")
        
        # Current allocation
        st.subheader("📊 Current Allocation")
        if not dashboard.portfolio_weights.empty:
            st.dataframe(
                dashboard.portfolio_weights[['symbol', 'weight_pct', 'sharpe_ratio', 'avg_return']].round(3),
                use_container_width=True,
                height=400
            )
        
        # Rebalancing recommendations
        recommendations = dashboard.get_rebalance_recommendations()
        
        if recommendations:
            st.subheader("💡 Rebalancing Recommendations")
            
            for rec in recommendations:
                if rec['type'] == 'REDUCE':
                    st.warning(f"🔻 **{rec['symbol']}**: Reduce from {rec['current_weight']:.1f}% to {rec['suggested_weight']:.1f}% - {rec['reason']}")
                elif rec['type'] == 'REVIEW':
                    st.info(f"🔍 **{rec['symbol']}**: Review allocation ({rec['current_weight']:.1f}%) - {rec['reason']}")
        else:
            st.success("✅ Portfolio allocation looks well-balanced!")
        
        # Rebalancing simulator
        st.subheader("🎮 Rebalancing Simulator")
        st.info("💡 Coming soon: Interactive rebalancing simulator")
    
    elif page == "🤖 AI Insights":
        st.header("🤖 AI-Powered Portfolio Insights")
        
        # AI question interface
        st.subheader("Ask Gemini: Portfolio & Sentiment Explanation")
        
        # Sample questions
        sample_questions = [
            "Why did my portfolio rebalance?",
            "What's driving today's sentiment?",
            "How risky is my current allocation?",
            "Should I rebalance based on recent performance?",
            "What are the key factors affecting my portfolio returns?",
            "How does my portfolio compare to holding just Bitcoin?",
            "What's the outlook for my top holdings?",
            "Are there any concerning trends in my portfolio?"
        ]
        
        # Question input
        selected_question = st.selectbox(
            "Choose a sample question or type your own:",
            [""] + sample_questions
        )
        
        user_question = st.text_area(
            "Ask a question about your portfolio or today's sentiment:",
            value=selected_question,
            placeholder="Example: Why did my portfolio rebalance?",
            height=100
        )
        
        if st.button("🚀 Ask Gemini", type="primary"):
            if user_question.strip():
                with st.spinner("🤖 Gemini is analyzing your portfolio..."):
                    response = dashboard.ask_gemini(user_question)
                    st.markdown("### 🤖 Gemini's Analysis:")
                    st.markdown(response)
            else:
                st.warning("Please enter a question first!")
        
        # Pre-generated insights
        st.subheader("💡 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Portfolio Performance Summary", use_container_width=True):
                with st.spinner("Generating insights..."):
                    insight = dashboard.ask_gemini("Provide a comprehensive summary of my portfolio's performance, highlighting key strengths and areas for improvement.")
                    st.markdown(insight)
        
        with col2:
            if st.button("🧠 Sentiment Impact Analysis", use_container_width=True):
                with st.spinner("Analyzing sentiment impact..."):
                    insight = dashboard.ask_gemini("How is current market sentiment affecting my portfolio? What should I watch for?")
                    st.markdown(insight)
        
        with col1:
            if st.button("⚖️ Risk Assessment", use_container_width=True):
                with st.spinner("Assessing risks..."):
                    insight = dashboard.ask_gemini("What are the main risks in my current portfolio allocation? How can I mitigate them?")
                    st.markdown(insight)
        
        with col2:
            if st.button("🔮 Future Outlook", use_container_width=True):
                with st.spinner("Analyzing outlook..."):
                    insight = dashboard.ask_gemini("Based on my portfolio composition and current market sentiment, what's the outlook for the next few weeks?")
                    st.markdown(insight)
    
    elif page == "📰 Market News":
        st.header("📰 Recent Market News & Analysis")
        
        # Latest sentiment summary
        sentiment = dashboard.get_latest_sentiment()
        st.subheader("📊 Latest Daily Sentiment")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Sentiment", f"{sentiment['mean']:.3f}")
        with col2:
            sentiment_label = "Positive" if sentiment['mean'] > 0.1 else "Negative" if sentiment['mean'] < -0.1 else "Neutral"
            st.metric("Market Mood", sentiment_label)
        with col3:
            st.metric("Analysis Date", sentiment.get('date', 'N/A'))
        
        # News headlines
        st.subheader("📰 Recent News Headlines")
        news_headlines = dashboard.get_news_headlines(10)
        
        if news_headlines:
            for i, headline in enumerate(news_headlines, 1):
                with st.expander(f"📄 {headline['title'][:100]}{'...' if len(headline['title']) > 100 else ''}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Date:** {headline['date']}")
                    with col2:
                        st.write(f"**Source:** {headline['source']}")
                    with col3:
                        sentiment_color = "🟢" if headline['sentiment'] == 'Positive' else "🔴" if headline['sentiment'] == 'Negative' else "🟡"
                        st.write(f"**Sentiment:** {sentiment_color} {headline['sentiment']}")
        else:
            st.info("📰 No recent news data available. Run the analysis system to fetch latest news.")
        
        # Market sentiment trends
        if not dashboard.combined_data.empty:
            st.subheader("📈 Sentiment Trends")
            with st.container():
                st.plotly_chart(dashboard.create_sentiment_chart(), use_container_width=True)
        
        # News impact analysis
        st.subheader("📊 News Impact on Portfolio")
        if st.button("🔍 Analyze News Impact", type="primary"):
            with st.spinner("Analyzing news impact..."):
                analysis = dashboard.ask_gemini("How are recent news headlines affecting my portfolio performance? What key themes should I be aware of?")
                st.markdown("### 📊 News Impact Analysis:")
                st.markdown(analysis)

if __name__ == "__main__":
    main()