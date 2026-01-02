import streamlit as st
import logging
import re
import hashlib
import json
import os
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

def calculateDataHash(email_data):
    """
    Calculate a hash of the email data to detect changes.
    Uses subject, content, and metrics to create a unique identifier.
    """
    # Create a string representation of key data
    data_string = ""
    for _, row in email_data.iterrows():
        data_string += f"{row.get('subject', '')}{row.get('plaintext', '')}{row.get('message_body', '')}"
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

def runCompleteAnalysis(force_refresh=False):
    """
    Run complete email analysis automatically.
    Uses cache if available and data hasn't changed.
    """
    # Step 1: Load email data
    with st.spinner("Loading emails from database..."):
        try:
            email_data = processEmailData()
            if email_data.empty:
                st.error("No email data found in database")
                return False
        except Exception as e:
            st.error(f"Failed to load email data: {str(e)}")
            return False
    
    # Step 2: Calculate data hash
    current_hash = calculateDataHash(email_data)
    
    # Step 3: Check cache if not forcing refresh
    if not force_refresh:
        cached_data = loadCachedAnalysis()
        if cached_data and cached_data.get('data_hash') == current_hash:
            logger.info("Using cached analysis (data unchanged)")
            st.session_state.analysis_results = cached_data['analysis_result']
            st.session_state.email_context = cached_data['email_context']
            st.session_state.data_hash = current_hash
            return True
    
    # Step 4: Initialize agent if needed
    if st.session_state.gemini_model is None:
        with st.spinner("Initializing AI agent..."):
            try:
                st.session_state.gemini_model = initializeGeminiAgent()
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                return False
    
    # Step 5: Run analysis (data changed or cache doesn't exist)
    with st.spinner("Analyzing emails with AI... This may take a few minutes."):
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
                        st.warning(f"⚠️ Rate limit exceeded. Please wait {waitTime:.0f} seconds and try again.")
                    else:
                        st.error(f"Quota error: {str(e)}")
            else:
                st.error(f"Analysis failed: {str(e)}")
            return False

# Main UI
st.title("📧 Email Marketing Expert Agent")
st.markdown("---")

# Auto-load analysis on first run
if not st.session_state.analysis_loaded:
    st.session_state.analysis_loaded = True
    runCompleteAnalysis()

# Two main modes
tab1, tab2 = st.tabs(["📊 Analysis Mode", "💬 Interactive Chat"])

# TAB 1: Analysis Mode
with tab1:
    st.header("📊 Complete Email Analysis")
    
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
            elif st.session_state.gemini_model is None:
                st.warning("⚠️ Please run the analysis first to initialize the agent.")
            else:
                with st.spinner("Analyzing email and generating recommendations..."):
                    try:
                        recommendations = analyzeSingleEmailForImprovement(
                            st.session_state.gemini_model,
                            email_content,
                            email_subject if email_subject else None
                        )
                        
                        # Add to conversation history
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
    if st.session_state.conversation_history:
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
                        response = chatWithEmailExpert(
                            st.session_state.gemini_model,
                            user_question,
                            st.session_state.conversation_history[:-1],  # Exclude current message
                            st.session_state.email_context
                        )
                        st.markdown(response)
                        
                        # Add assistant response to history
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
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': error_msg
                        })
    
    # Sidebar actions
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
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

if __name__ == "__main__":
    pass
