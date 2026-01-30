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
    
    if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary", key="run_analysis"):
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
        
        if st.button("🔍 Analyze This Email", use_container_width=True, type="primary", key="analyze_single"):
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
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if 'conversation_history' in st.session_state:
                st.session_state.conversation_history = []
            st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear Cache", use_container_width=True, help="Delete cached analysis to force refresh"):
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rate_options = {
                'Open Rate': 'openRate',
                'Click Rate': 'clickRate',
                'Unsubscribe Rate': 'unsubRate',
                'Abuse Rate': 'abuseRate'
            }
            selected_rate_label = st.selectbox(
                "Select Rate",
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
                "Select Metric",
                options=list(metric_options.keys()),
                key="metric_selector"
            )
            selected_metric = metric_options[selected_metric_label]
        
        with col3:
            grouping_options = ['Quartiles', 'Deciles']
            selected_grouping = st.selectbox(
                "Grouping Method",
                options=grouping_options,
                index=0,
                key="grouping_selector"
            )
        
        # Check if selected metric exists in data
        if selected_metric not in email_data.columns:
            st.error(f"Metric '{selected_metric}' not found in data. Please ensure metrics are calculated.")
        else:
            # Filter out NaN values
            plot_data = email_data[[selected_metric, selected_rate]].dropna()
            
            if len(plot_data) == 0:
                st.warning("No data available for selected metric and rate combination.")
            else:
                # Grouping logic
                def create_qcut_groups(data, q, labels=None, metric_name=None):
                    """Helper function to create quartile groups with fixed percentiles.
                    Always creates exactly 4 quartiles based on 25%, 50%, 75% percentiles.
                    Returns (grouped_series, bins_info_dict) where bins_info contains range information."""
                    try:
                        # For quartiles (q=4), always use fixed percentiles
                        if q == 4 and labels and len(labels) == 4:
                            min_val = float(data.min())
                            max_val = float(data.max())
                            
                            # If all values are the same, create 4 equal groups anyway
                            if min_val == max_val:
                                bins = [min_val, min_val, min_val, min_val, max_val]
                            else:
                                # Calculate percentiles: 25%, 50%, 75%
                                p25 = float(data.quantile(0.25))
                                p50 = float(data.quantile(0.50))
                                p75 = float(data.quantile(0.75))
                                
                                # Create bins: use percentiles, but ensure we have exactly 5 edges
                                # If percentiles are equal, use equal-width bins
                                unique_percentiles = sorted(list(set([min_val, p25, p50, p75, max_val])))
                                
                                if len(unique_percentiles) >= 5:
                                    # All percentiles are different - use them
                                    bins = [min_val, p25, p50, p75, max_val]
                                elif len(unique_percentiles) == 4:
                                    # One percentile is duplicate - fill with equal spacing
                                    bins = [min_val]
                                    if p25 > min_val:
                                        bins.append(p25)
                                    bins.append(p50)
                                    if p75 > p50:
                                        bins.append(p75)
                                    bins.append(max_val)
                                    # Fill to 5 if needed
                                    while len(bins) < 5:
                                        # Insert intermediate values
                                        for i in range(len(bins) - 1):
                                            mid = (bins[i] + bins[i+1]) / 2
                                            if mid not in bins:
                                                bins.insert(i+1, mid)
                                                break
                                elif len(unique_percentiles) == 3:
                                    # Two percentiles are duplicates - use equal spacing
                                    bins = [min_val, 
                                           min_val + (max_val - min_val) * 0.25,
                                           min_val + (max_val - min_val) * 0.5,
                                           min_val + (max_val - min_val) * 0.75,
                                           max_val]
                                else:
                                    # All percentiles are the same - use equal spacing
                                    bins = [min_val,
                                           min_val + (max_val - min_val) * 0.25,
                                           min_val + (max_val - min_val) * 0.5,
                                           min_val + (max_val - min_val) * 0.75,
                                           max_val]
                            
                            # Ensure bins are sorted and we have exactly 5 edges
                            bins = sorted(list(set(bins)))
                            while len(bins) < 5:
                                # Add intermediate values
                                for i in range(len(bins) - 1):
                                    mid = (bins[i] + bins[i+1]) / 2
                                    if abs(mid - bins[i]) > 1e-10 and abs(mid - bins[i+1]) > 1e-10:
                                        bins.insert(i+1, mid)
                                        break
                                bins = sorted(bins)
                            
                            # Take exactly 5 bins for 4 groups
                            bins = bins[:5]
                            
                            # Create bins_info dictionary with ranges
                            # Use proper interval notation: [a,b] for first interval (include_lowest=True), (a,b] for others
                            bins_info = {}
                            for i in range(4):
                                bin_min = bins[i]
                                bin_max = bins[i + 1]
                                
                                # Format the range in an intuitive way: "3.0 to 4.0 words"
                                # Use simple "to" instead of mathematical notation for better readability
                                if metric_name:
                                    if 'words' in metric_name:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f} words"
                                    elif 'chars' in metric_name:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f} chars"
                                    elif 'emojis' in metric_name:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f} emojis"
                                    elif 'images' in metric_name:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f} images"
                                    elif 'ctas' in metric_name:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f} CTAs"
                                    else:
                                        range_str = f"{bin_min:.1f} to {bin_max:.1f}"
                                else:
                                    range_str = f"{bin_min:.1f} to {bin_max:.1f}"
                                
                                if i < len(labels):
                                    bins_info[labels[i]] = range_str
                            
                            # Create descriptive labels with ranges (show only range, not Q1, Q2, etc.)
                            # This makes it more intuitive for non-statistical users
                            descriptive_labels = [bins_info[label] for label in labels]
                            
                            # Use pd.cut with fixed bins to always create 4 groups
                            result = pd.cut(data, bins=bins, labels=descriptive_labels, include_lowest=True, duplicates='drop')
                            
                            return result, bins_info
                        
                        # For other cases (deciles, etc.), use original qcut approach
                        else:
                            result, bins = pd.qcut(data, q=q, duplicates='drop', retbins=True)
                            actual_bins = len(bins) - 1
                            
                            # Create bins_info dictionary with ranges
                            bins_info = {}
                            for i in range(actual_bins):
                                bin_min = bins[i]
                                bin_max = bins[i + 1]
                                # Format the range based on metric type
                                if metric_name:
                                    if 'words' in metric_name:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f} words"
                                    elif 'chars' in metric_name:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f} chars"
                                    elif 'emojis' in metric_name:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f} emojis"
                                    elif 'images' in metric_name:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f} images"
                                    elif 'ctas' in metric_name:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f} CTAs"
                                    else:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f}"
                                else:
                                    range_str = f"{bin_min:.2f}-{bin_max:.2f}"
                                
                                if labels and i < len(labels):
                                    bins_info[labels[i]] = range_str
                                else:
                                    bins_info[f"Bin {i+1}"] = range_str
                            
                            if labels is None:
                                return result, bins_info
                            else:
                                if actual_bins == len(labels):
                                    descriptive_labels = [f"{label} ({bins_info[label]})" for label in labels[:actual_bins]]
                                    return pd.qcut(data, q=q, duplicates='drop', labels=descriptive_labels), bins_info
                                elif actual_bins > 0 and actual_bins < len(labels):
                                    adjusted_labels = labels[:actual_bins]
                                    descriptive_labels = [f"{label} ({bins_info[label]})" for label in adjusted_labels]
                                    return pd.qcut(data, q=q, duplicates='drop', labels=descriptive_labels), bins_info
                                else:
                                    return result, bins_info
                    except Exception as e:
                        # Fallback: create equal-width bins
                        try:
                            min_val = data.min()
                            max_val = data.max()
                            if max_val == min_val:
                                # All values are the same
                                return pd.Series([f"{labels[0] if labels else 'All'} ({min_val:.0f})"] * len(data), index=data.index), {}
                            
                            bins = [min_val, min_val + (max_val - min_val) * 0.25,
                                   min_val + (max_val - min_val) * 0.5,
                                   min_val + (max_val - min_val) * 0.75, max_val]
                            
                            if labels and len(labels) == 4:
                                bins_info = {}
                                for i in range(4):
                                    bin_min = bins[i]
                                    bin_max = bins[i + 1]
                                    if metric_name:
                                        if 'words' in metric_name:
                                            range_str = f"{bin_min:.1f}-{bin_max:.1f} words"
                                        elif 'chars' in metric_name:
                                            range_str = f"{bin_min:.1f}-{bin_max:.1f} chars"
                                        else:
                                            range_str = f"{bin_min:.1f}-{bin_max:.1f}"
                                    else:
                                        range_str = f"{bin_min:.1f}-{bin_max:.1f}"
                                    bins_info[labels[i]] = range_str
                                
                                descriptive_labels = [f"{label} ({bins_info[label]})" for label in labels]
                                result = pd.cut(data, bins=bins, labels=descriptive_labels, include_lowest=True)
                                return result, bins_info
                            else:
                                return pd.cut(data, bins=bins, include_lowest=True), {}
                        except:
                            return pd.Series(['All'] * len(data), index=data.index), {}
                
                bins_info = {}
                if selected_grouping == 'Quartiles':
                    plot_data['group'], bins_info = create_qcut_groups(
                        plot_data[selected_metric], 
                        q=4, 
                        labels=['Q1', 'Q2', 'Q3', 'Q4'],
                        metric_name=selected_metric
                    )
                elif selected_grouping == 'Deciles':
                    plot_data['group'], bins_info = create_qcut_groups(
                        plot_data[selected_metric], 
                        q=10,
                        metric_name=selected_metric
                    )
                    
                    if plot_data['group'].nunique() < 4:
                        st.info("Not enough data for deciles. Using quartiles instead.")
                        plot_data['group'], bins_info = create_qcut_groups(
                            plot_data[selected_metric], 
                            q=4, 
                            labels=['Q1', 'Q2', 'Q3', 'Q4'],
                            metric_name=selected_metric
                        )
                # Convert any Interval objects to strings for Plotly compatibility
                # This is needed for deciles and other groupings that don't use labels
                # Format with 2 decimal places for consistency
                def format_interval_to_string(interval_obj):
                    """Convert Interval object to formatted string with 2 decimal places."""
                    if hasattr(interval_obj, 'left') and hasattr(interval_obj, 'right'):
                        left = float(interval_obj.left)
                        right = float(interval_obj.right)
                        # Check if it's closed on left/right
                        left_bracket = '[' if interval_obj.closed == 'left' or interval_obj.closed == 'both' else '('
                        right_bracket = ']' if interval_obj.closed == 'right' or interval_obj.closed == 'both' else ')'
                        return f"{left_bracket}{left:.1f}, {right:.1f}{right_bracket}"
                    return str(interval_obj)
                
                try:
                    # Try to access the first value to check its type
                    if len(plot_data) > 0:
                        first_val = plot_data['group'].iloc[0]
                        # Check if it's an Interval object (has 'left' and 'right' attributes)
                        if hasattr(first_val, 'left') and hasattr(first_val, 'right'):
                            # Convert Interval objects to formatted string representation
                            plot_data['group'] = plot_data['group'].apply(format_interval_to_string)
                        else:
                            # Already strings, but ensure they're properly formatted
                            plot_data['group'] = plot_data['group'].astype(str)
                except (AttributeError, IndexError):
                    # If we can't check, try converting anyway (safe operation)
                    try:
                        plot_data['group'] = plot_data['group'].astype(str)
                    except:
                        pass
                
                # Calculate average rate per group
                grouped_stats = plot_data.groupby('group', observed=True).agg({
                    selected_rate: ['mean', 'median', 'min', 'max', 'count']
                }).round(2)
                grouped_stats.columns = ['Average Rate', 'Median Rate', 'Rate Min', 'Rate Max', 'Email Count']
                grouped_stats = grouped_stats.reset_index()
                
                # Ensure group column is string for Plotly
                grouped_stats['group'] = grouped_stats['group'].astype(str)
                
                # Create bar chart
                fig = px.bar(
                    grouped_stats,
                    x='group',
                    y='Average Rate',
                    title=f'{selected_rate_label} by {selected_metric_label} ({selected_grouping})',
                    labels={'group': selected_metric_label, 'Average Rate': selected_rate_label + ' (%)'},
                    text='Average Rate'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(
                    xaxis_title=selected_metric_label,
                    yaxis_title=selected_rate_label + ' (%)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics table with sorting capability
                st.markdown("### 📊 Statistics by Group")
                
                # Add sorting controls
                col_sort1, col_sort2 = st.columns(2)
                with col_sort1:
                    sort_column = st.selectbox(
                        "Sort by column:",
                        options=['group', 'Average Rate', 'Median Rate', 'Rate Min', 'Rate Max', 'Email Count'],
                        index=0,
                        key="sort_column_selector"
                    )
                with col_sort2:
                    sort_ascending = st.selectbox(
                        "Sort order:",
                        options=[True, False],
                        format_func=lambda x: "Ascending" if x else "Descending",
                        index=0,
                        key="sort_order_selector"
                    )
                
                # Sort the dataframe
                sorted_stats = grouped_stats.sort_values(by=sort_column, ascending=sort_ascending).reset_index(drop=True)
                
                # Display sorted table
                st.dataframe(sorted_stats, use_container_width=True)
                
                # Calculate correlation
                correlation = plot_data[selected_metric].corr(plot_data[selected_rate])
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

if __name__ == "__main__":
    pass
