import streamlit as st
import logging
import re
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

def runCompleteAnalysis():
    """
    Run complete email analysis automatically.
    This function does everything: initializes agent, loads data, and analyzes.
    """
    # Step 1: Initialize agent if needed
    if st.session_state.gemini_model is None:
        with st.spinner("Initializing AI agent..."):
            try:
                st.session_state.gemini_model = initializeGeminiAgent()
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                return False
    
    # Step 2: Load email data
    with st.spinner("Loading emails from database..."):
        try:
            email_data = processEmailData()
            if email_data.empty:
                st.error("No email data found in database")
                return False
        except Exception as e:
            st.error(f"Failed to load email data: {str(e)}")
            return False
    
    # Step 3: Run analysis
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

# Two main modes
tab1, tab2 = st.tabs(["📊 Analysis Mode", "💬 Interactive Chat"])

# TAB 1: Analysis Mode
with tab1:
    st.header("📊 Complete Email Analysis")
    st.markdown("Click the button below to automatically analyze all emails from your database.")
    
    if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary", key="run_analysis"):
        if runCompleteAnalysis():
            st.success("✅ Analysis completed successfully!")
            st.balloons()
    
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
    
    # Quick question suggestions
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    col1, col2 = st.columns(2)
    
    def handleQuickQuestion(question):
        """Handle quick question button click - add question and generate response."""
        if st.session_state.gemini_model is None:
            st.warning("⚠️ Please run the analysis first in 'Analysis Mode' to initialize the agent.")
            return
        
        # Add user question to history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': question
        })
        
        # Generate expert response
        try:
            response = chatWithEmailExpert(
                st.session_state.gemini_model,
                question,
                st.session_state.conversation_history[:-1],  # Exclude current message
                st.session_state.email_context
            )
            
            # Add assistant response to history
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()
        except Exception as e:
            errorStr = str(e)
            if "429" in errorStr or "ResourceExhausted" in errorStr:
                if "GenerateRequestsPerDay" in errorStr or "free_tier_requests" in errorStr or "limit: 20" in errorStr:
                    st.error("""
                    ⚠️ **Daily Limit Reached**
                    
                    You have reached the daily limit of 20 requests.
                    
                    **You must wait until tomorrow** to continue.
                    """)
                else:
                    st.error(f"Error: {str(e)}")
            else:
                st.error(f"Error: {str(e)}")
    
    with col1:
        if st.button("How to improve open rates?", use_container_width=True, key="q1"):
            handleQuickQuestion("How can I improve my open rates?")
        
        if st.button("What makes a good subject line?", use_container_width=True, key="q2"):
            handleQuickQuestion("What makes a good subject line?")
    
    with col2:
        if st.button("How to reduce unsubscribes?", use_container_width=True, key="q3"):
            handleQuickQuestion("How to reduce unsubscribe rates?")
        
        if st.button("Best practices for CTAs?", use_container_width=True, key="q4"):
            handleQuickQuestion("Best practices for email CTAs?")
    
    # Sidebar actions
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

if __name__ == "__main__":
    pass
