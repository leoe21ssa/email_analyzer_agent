import streamlit as st
import logging
import re
import hashlib
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
from src.processor import processEmailData
from src.agent import (
    initializeGeminiAgent, 
    analyzeEmailBatch, 
    chatWithEmailExpert, 
    analyzeSingleEmailForImprovement
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "analysis_cache.json"

def calculateDataHash(email_data, email_ids=None):
    """
    Calculate a hash of the email data to detect changes.
    Uses subject, content, metrics, and email IDs to create a unique identifier.
    
    Args:
        email_data: DataFrame with email data
        email_ids: Optional list of email IDs being analyzed (from EMAIL_IDS env var)
    """
    # Create a string representation of key data
    data_string = ""
    
    # Include email IDs in hash if provided (ensures cache is specific to ID list)
    if email_ids:
        # Sort IDs to ensure consistent hash regardless of order
        sorted_ids = sorted([str(id) for id in email_ids])
        data_string += f"IDS:{','.join(sorted_ids)}|"
    else:
        # If no IDs specified, include all IDs from the data to detect changes
        if 'id' in email_data.columns:
            sorted_ids = sorted([str(id) for id in email_data['id'].unique()])
            data_string += f"IDS:{','.join(sorted_ids)}|"
    
    # Include email content and metrics
    for _, row in email_data.iterrows():
        data_string += f"{row.get('id', '')}{row.get('subject', '')}{row.get('plaintext', '')}{row.get('message_body', '')}"
        data_string += f"{row.get('mcsent', 0)}{row.get('mcopened', 0)}{row.get('mcclicked', 0)}{row.get('mcunsub', 0)}"
    
    # Calculate hash
    return hashlib.md5(data_string.encode()).hexdigest()

def loadCachedAnalysis():
    """Load cached analysis if it exists and is valid."""
    if not CACHE_FILE.exists():
        return None
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load cache: {str(e)}")
        return None

def saveAnalysisToCache(analysis_result, email_context, data_hash, email_data):
    """Save analysis results to cache."""
    try:
        cache_data = {
            'analysis_result': analysis_result,
            'email_context': email_context,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            'email_count': len(email_data),
            'avg_open_rate': float(email_data['openRate'].mean()) if 'openRate' in email_data.columns else 0,
            'avg_click_rate': float(email_data['clickRate'].mean()) if 'clickRate' in email_data.columns else 0,
        }
        
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis cached successfully. Hash: {data_hash[:8]}...")
    except Exception as e:
        logger.error(f"Failed to save cache: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Email Marketing Expert Agent",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'email_context' not in st.session_state:
    st.session_state.email_context = None
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None
if 'analysis_loaded' not in st.session_state:
    st.session_state.analysis_loaded = False
if 'analysis_initialized' not in st.session_state:
    st.session_state.analysis_initialized = False

def runCompleteAnalysis(force_refresh=False, show_spinners=True):
    """
    Run complete email analysis automatically.
    Uses cache if available and data hasn't changed.
    
    Args:
        force_refresh: Force a new analysis even if cache exists
        show_spinners: Whether to show spinner messages (set to False to avoid UI interference)
    """
    # Step 1: Load email data
    spinner_context = st.spinner("Loading emails from database...") if show_spinners else st.empty()
    with spinner_context:
        try:
            email_data = processEmailData()
            if email_data.empty:
                if show_spinners:
                st.error("No email data found in database")
                return False
        except Exception as e:
            if show_spinners:
            st.error(f"Failed to load email data: {str(e)}")
            return False
    
    # Step 2: Get email IDs from environment (for cache hash)
    emailIdsStr = os.getenv("EMAIL_IDS", "").strip()
    email_ids = None
    if emailIdsStr:
        email_ids = [id.strip() for id in emailIdsStr.split(',') if id.strip()]
    
    # Step 3: Calculate data hash (including IDs to detect changes in ID list)
    current_hash = calculateDataHash(email_data, email_ids)
    
    # Step 4: Check cache if not forcing refresh
    if not force_refresh:
        cached_data = loadCachedAnalysis()
        if cached_data and cached_data.get('data_hash') == current_hash:
            logger.info("Using cached analysis (data unchanged)")
            st.session_state.analysis_results = cached_data['analysis_result']
            st.session_state.email_context = cached_data['email_context']
            st.session_state.data_hash = current_hash
            return True
    
    # Step 5: Initialize agent if needed
    if st.session_state.gemini_model is None:
        spinner_context = st.spinner("Initializing AI agent...") if show_spinners else st.empty()
        with spinner_context:
            try:
                st.session_state.gemini_model = initializeGeminiAgent()
            except Exception as e:
                if show_spinners:
                st.error(f"Failed to initialize agent: {str(e)}")
                return False
    
    # Step 6: Run analysis (data changed or cache doesn't exist)
    spinner_context = st.spinner("Analyzing emails with AI... This may take a few minutes.") if show_spinners else st.empty()
    with spinner_context:
        try:
            analysis = analyzeEmailBatch(
                email_data, 
                st.session_state.gemini_model, 
                batchSize=3
            )
            st.session_state.analysis_results = analysis
            
            # Create context summary for chat
            topEmails = email_data.nlargest(3, 'effectivenessScore')
            summary = f"""
Email Performance Summary:
- Total emails analyzed: {len(email_data)}
- Average open rate: {email_data['openRate'].mean():.2f}%
- Average click rate: {email_data['clickRate'].mean():.2f}%
- Average unsubscribe rate: {email_data['unsubRate'].mean():.2f}%
- Top performing email subject: {topEmails.iloc[0]['subject'] if len(topEmails) > 0 else 'N/A'}
"""
            st.session_state.email_context = summary
            st.session_state.data_hash = current_hash
            
            # Save to cache
            saveAnalysisToCache(analysis, summary, current_hash, email_data)
            
            return True
        except Exception as e:
            errorStr = str(e)
            
            # Detect daily limit
            if "429" in errorStr or "ResourceExhausted" in errorStr:
                if "GenerateRequestsPerDay" in errorStr or "free_tier_requests" in errorStr or "limit: 20" in errorStr:
                    if show_spinners:
                    st.error("""
                    ⚠️ **Daily Limit Reached**
                    
                    You have reached the daily limit of 20 requests on the free tier of Gemini API.
                    
                    **You must wait until tomorrow** for the limit to reset automatically.
                    
                    The limit resets daily at 00:00 UTC.
                    """)
                    return False
                else:
                    # Rate limit (per minute) - can retry
                    match = re.search(r'retry in ([\d.]+)s', errorStr, re.IGNORECASE)
                    if match:
                        waitTime = float(match.group(1))
                        if show_spinners:
                        st.warning(f"⚠️ Rate limit exceeded. Please wait {waitTime:.0f} seconds and try again.")
                    else:
                        if show_spinners:
                        st.error(f"Quota error: {str(e)}")
            else:
                if show_spinners:
                st.error(f"Analysis failed: {str(e)}")
            return False

# Main UI
st.title("📧 Email Marketing Expert Agent")
st.markdown("---")

# Three main modes
tab1, tab2, tab3 = st.tabs(["📊 Analysis Mode", "💬 Interactive Chat", "📈 Metrics Dashboard"])

# TAB 1: Analysis Mode
with tab1:
    st.header("📊 Complete Email Analysis")
    
    # Load cached analysis if available (only once, silently)
    # Don't run analysis automatically - let user trigger it manually
    if not st.session_state.get('cache_loaded', False):
        st.session_state.cache_loaded = True
        cached_data = loadCachedAnalysis()
        if cached_data:
            # Load cached data silently
            st.session_state.analysis_results = cached_data.get('analysis_result')
            st.session_state.email_context = cached_data.get('email_context')
            st.session_state.data_hash = cached_data.get('data_hash')
    
    # Show cache status
    cached_data = loadCachedAnalysis()
    if cached_data and st.session_state.data_hash == cached_data.get('data_hash'):
        cache_time = cached_data.get('timestamp', 'Unknown')
        try:
            cache_datetime = datetime.fromisoformat(cache_time)
            formatted_time = cache_datetime.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = cache_time
        st.info(f"ℹ️ **Using cached analysis** from {formatted_time}. Data unchanged since last analysis.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Click the button below to analyze all emails from your database.")
    with col2:
        if st.button("🔄 Force Refresh", help="Force a new analysis even if data hasn't changed"):
            if runCompleteAnalysis(force_refresh=True):
                st.success("✅ Analysis refreshed successfully!")
                st.rerun()
    
    if st.button("🚀 Run Complete Analysis", type="primary", key="run_analysis"):
        if runCompleteAnalysis():
            st.success("✅ Analysis completed successfully!")
            st.balloons()
            st.rerun()
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### 📄 Analysis Results")
        st.markdown(st.session_state.analysis_results)
        
        st.info("💡 **Tip**: Switch to 'Interactive Chat' tab to ask questions about this analysis or get recommendations for specific emails.")

# TAB 2: Interactive Chat Mode
with tab2:
    st.header("💬 Interactive Chat with Email Expert")
    
    if st.session_state.analysis_results is None:
        st.info("ℹ️ **Note**: Run the analysis first in the 'Analysis Mode' tab to get better context-aware responses.")
    else:
        st.success("✅ Analysis context loaded. The expert can reference your email performance data.")
    
    st.markdown("---")
    
    # Section: Analyze a specific email
    with st.expander("📝 Analyze a Specific Email", expanded=False):
        st.markdown("Paste an email below to get specific improvement recommendations:")
        
        email_subject = st.text_input("Subject Line:", placeholder="Enter email subject line...", key="email_subject")
        email_content = st.text_area(
            "Email Content:", 
            placeholder="Paste your email content here...",
            height=200,
            key="email_content"
        )
        
        if st.button("🔍 Analyze This Email", type="primary", key="analyze_single"):
            if not email_content:
                st.warning("⚠️ Please paste email content first.")
            else:
                # Auto-initialize agent if not already initialized
                if st.session_state.gemini_model is None:
                    with st.spinner("Initializing AI agent..."):
                        try:
                            st.session_state.gemini_model = initializeGeminiAgent()
                        except Exception as e:
                            st.error(f"Failed to initialize agent: {str(e)}")
                            st.stop()
                
                # Proceed with analysis
                with st.spinner("Analyzing email and generating recommendations..."):
                    try:
                        recommendations = analyzeSingleEmailForImprovement(
                            st.session_state.gemini_model,
                            email_content,
                            email_subject if email_subject else None,
                            None,  # emailMetrics
                            st.session_state.analysis_results if st.session_state.analysis_results else None  # Pass batch analysis context
                        )
                        
                        # Add to conversation history
                        if 'conversation_history' not in st.session_state:
                            st.session_state.conversation_history = []
                        user_msg = f"Please analyze this email:\n\nSubject: {email_subject if email_subject else 'N/A'}\n\nContent:\n{email_content}"
                        st.session_state.conversation_history.append({
                            'role': 'user',
                            'content': user_msg
                        })
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': recommendations
                        })
                        
                        st.success("✅ Analysis complete! Check the chat below.")
                        st.rerun()
                    except Exception as e:
                        errorStr = str(e)
                        if "429" in errorStr or "ResourceExhausted" in errorStr:
                            if "GenerateRequestsPerDay" in errorStr or "free_tier_requests" in errorStr or "limit: 20" in errorStr:
                                st.error("""
                                ⚠️ **Daily Limit Reached**
                                
                                You have reached the daily limit of 20 requests on the free tier.
                                
                                **You must wait until tomorrow** to continue using the service.
                                """)
                            else:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Chat section
    st.markdown("### 💬 Chat with Expert")
    st.markdown("Ask questions about email marketing, get advice, or discuss your email performance.")
    
    # Display conversation history
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        for msg in st.session_state.conversation_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                with st.chat_message("user"):
                    # Truncate very long messages for display
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                        with st.expander("View full message"):
                            st.write(content)
                    else:
                        st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    
    # Chat input
    user_question = st.chat_input("Ask the email marketing expert...")
    
    if user_question:
        if st.session_state.gemini_model is None:
            st.warning("⚠️ Please run the analysis first in 'Analysis Mode' to initialize the agent.")
        else:
            # Add user message to history
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get expert response
            with st.chat_message("assistant"):
                with st.spinner("Expert is thinking..."):
                    try:
                        conversation_history = st.session_state.get('conversation_history', [])
                        email_context = st.session_state.get('email_context', None)
                        
                        response = chatWithEmailExpert(
                            st.session_state.gemini_model,
                            user_question,
                            conversation_history[:-1] if conversation_history else [],  # Exclude current message
                            email_context
                        )
                        st.markdown(response)
                        
                        # Add assistant response to history
                        if 'conversation_history' not in st.session_state:
                            st.session_state.conversation_history = []
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                    except Exception as e:
                        errorStr = str(e)
                        if "429" in errorStr or "ResourceExhausted" in errorStr:
                            if "GenerateRequestsPerDay" in errorStr or "free_tier_requests" in errorStr or "limit: 20" in errorStr:
                                error_msg = """
                                ⚠️ **Daily Limit Reached**
                                
                                You have reached the daily limit of 20 requests on the free tier.
                                
                                **You must wait until tomorrow** to continue using the service.
                                """
                            else:
                                error_msg = f"Error: {str(e)}"
                        else:
                            error_msg = f"Error: {str(e)}"
                        
                        st.error(error_msg)
                        if 'conversation_history' not in st.session_state:
                            st.session_state.conversation_history = []
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': error_msg
                        })
    
    # Sidebar actions
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            if 'conversation_history' in st.session_state:
                st.session_state.conversation_history = []
            st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear Cache", help="Delete cached analysis to force refresh"):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
                st.success("Cache cleared!")
                st.rerun()
            else:
                st.info("No cache file found.")

# TAB 3: Metrics Dashboard
with tab3:
    st.header("📈 Metrics Dashboard")
    st.markdown("Analyze relationships between email metrics and performance rates.")
    
    # Load email data
    @st.cache_data
    def loadEmailDataForDashboard():
        try:
            return processEmailData()
        except Exception as e:
            st.error(f"Failed to load email data: {str(e)}")
            return pd.DataFrame()
    
    email_data = loadEmailDataForDashboard()
    
    if email_data.empty:
        st.warning("No email data available. Please ensure emails are loaded in the database.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            rate_options = {
                'Open Rate': 'openRate',
                'Click Rate': 'clickRate',
                'Unsubscribe Rate': 'unsubRate',
                'Abuse Rate': 'abuseRate'
            }
            selected_rate_label = st.selectbox(
                "Select Rate (X-axis)",
                options=list(rate_options.keys()),
                key="rate_selector"
            )
            selected_rate = rate_options[selected_rate_label]
        
        with col2:
            metric_options = {
                'Subject - Words': 'subject_words',
                'Subject - Characters': 'subject_chars',
                'Subject - Emojis': 'subject_emojis',
                'Body - Words': 'body_words',
                'Body - Characters': 'body_chars',
                'Body - Emojis Total': 'body_emojis_total',
                'Body - Emojis Distinct': 'body_emojis_distinct',
                'Body - Images Total': 'body_images_total',
                'Body - Images With Link': 'body_images_with_link',
                'Body - CTAs Total': 'body_ctas_total',
                'Body - CTAs Distinct': 'body_ctas_distinct'
            }
            selected_metric_label = st.selectbox(
                "Select Metric (Y-axis)",
                options=list(metric_options.keys()),
                key="metric_selector"
            )
            selected_metric = metric_options[selected_metric_label]
        
        # Check if selected metric exists in data
        if selected_metric not in email_data.columns:
            st.error(f"Metric '{selected_metric}' not found in data. Please ensure metrics are calculated.")
        else:
            # Filter out NaN values
            plot_data = email_data[[selected_metric, selected_rate]].dropna()
            
            if len(plot_data) == 0:
                st.warning("No data available for selected metric and rate combination.")
            else:
                # Create scatter plot with rate on X-axis and metric on Y-axis
                fig = px.scatter(
                    plot_data,
                    x=selected_rate,
                    y=selected_metric,
                    title=f'{selected_metric_label} vs {selected_rate_label}',
                    labels={
                        selected_rate: selected_rate_label + ' (%)',
                        selected_metric: selected_metric_label
                    },
                    opacity=0.6,
                    trendline="ols"  # Add linear regression trendline
                )
                
                fig.update_layout(
                    xaxis_title=selected_rate_label + ' (%)',
                    yaxis_title=selected_metric_label,
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Display summary statistics
                st.markdown("### 📊 Summary Statistics")
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric(
                        "Total Emails",
                        len(plot_data)
                    )
                
                with col_stat2:
                    st.metric(
                        f"Avg {selected_rate_label}",
                        f"{plot_data[selected_rate].mean():.2f}%"
                    )
                
                with col_stat3:
                    st.metric(
                        f"Avg {selected_metric_label}",
                        f"{plot_data[selected_metric].mean():.1f}"
                    )
                
                with col_stat4:
                    correlation = plot_data[selected_metric].corr(plot_data[selected_rate])
                    st.metric(
                        "Correlation",
                        f"{correlation:.3f}"
                    )
                
                # Correlation interpretation
                st.markdown("---")
                st.markdown(f"**Correlation Coefficient:** {correlation:.3f}")
                
                if abs(correlation) < 0.1:
                    st.info("Very weak correlation")
                elif abs(correlation) < 0.3:
                    st.info("Weak correlation")
                elif abs(correlation) < 0.5:
                    st.info("Moderate correlation")
                elif abs(correlation) < 0.7:
                    st.info("Strong correlation")
                else:
                    st.info("Very strong correlation")
                
                # Display detailed statistics table
                st.markdown("### 📋 Detailed Statistics")
                
                stats_data = {
                    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    selected_rate_label: [
                        len(plot_data),
                        plot_data[selected_rate].mean(),  # Keep as numeric
                        plot_data[selected_rate].median(),
                        plot_data[selected_rate].std(),
                        plot_data[selected_rate].min(),
                        plot_data[selected_rate].max()
                    ],
                    selected_metric_label: [
                        len(plot_data),
                        plot_data[selected_metric].mean(),  # Keep as numeric
                        plot_data[selected_metric].median(),
                        plot_data[selected_metric].std(),
                        plot_data[selected_metric].min(),
                        plot_data[selected_metric].max()
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df, 
                    width='stretch', 
                    hide_index=True,
                    column_config={
                        selected_rate_label: st.column_config.NumberColumn(
                            selected_rate_label,
                            format="%.2f%%"
                        ),
                        selected_metric_label: st.column_config.NumberColumn(
                            selected_metric_label,
                            format="%.1f"
                        )
                    }
                )

if __name__ == "__main__":
    pass
