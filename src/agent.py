import os
import logging
import time
import re
import google.generativeai as genai
import google.api_core.exceptions as gcp_exceptions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def initializeGeminiAgent():
    """
    Initialize Gemini API with API key from environment variables.
    Returns configured Gemini model for email analysis.
    """
    apiKey = os.getenv("GEMINI_API_KEY")
    
    if not apiKey:
        logger.error("GEMINI_API_KEY environment variable is not set")
        raise ValueError("GEMINI_API_KEY must be set in environment variables")
    
    try:
        genai.configure(api_key=apiKey)
        
        # Try to list available models to find a compatible one
        try:
            models = genai.list_models()
            availableModels = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            logger.info(f"Available models: {availableModels}")
            
            # Try preferred models in order (using full model names from the list)
            preferredModelNames = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro-latest', 'gemini-flash-latest']
            modelName = None
            
            for preferred in preferredModelNames:
                fullModelName = f'models/{preferred}'
                if fullModelName in availableModels:
                    modelName = preferred  # Use short name for GenerativeModel
                    break
            
            if not modelName:
                # Use the first available model (extract short name)
                if availableModels:
                    modelName = availableModels[0].split('/')[-1]
                    logger.warning(f"Using first available model: {modelName}")
                else:
                    raise ValueError("No available models found")
            
            model = genai.GenerativeModel(modelName)
            logger.info(f"Gemini agent initialized successfully with model: {modelName}")
        except Exception as listError:
            # Fallback: try common model names
            logger.warning(f"Could not list models, trying fallback: {str(listError)}")
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini agent initialized with gemini-1.5-flash")
            except:
                model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini agent initialized with gemini-pro")
        
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini agent: {str(e)}")
        raise

def analyzeEmailEffectiveness(emailData, model):
    """
    Analyze email effectiveness using Gemini AI.
    Analyzes email content and metrics to identify what makes emails more effective.
    
    Args:
        emailData: Dictionary or string containing email information and metrics
        model: Initialized Gemini model
    
    Returns:
        Analysis results from Gemini
    """
    try:
        prompt = f"""
You are an expert email marketing analyst. Analyze the following email data and identify what elements contribute to higher email effectiveness.

Email Data:
{emailData}

Based on the metrics provided (sent, opened, clicked, unsubscribed), analyze:
1. What subject lines perform better?
2. What content elements (in plaintext or message_body) drive more opens and clicks?
3. What patterns lead to fewer unsubscribes?
4. What specific elements make emails more effective overall?

Provide a detailed analysis with specific recommendations.
"""
        
        response = model.generate_content(prompt)
        logger.info("Email effectiveness analysis completed")
        return response.text
    except Exception as e:
        logger.error(f"Failed to analyze email effectiveness: {str(e)}")
        raise

def analyzeEmailBatch(emailDataFrame, model, batchSize=3):
    """
    Analyze all emails in batches to identify patterns and best practices.
    Processes all emails in small groups to stay within API limits.
    
    Args:
        emailDataFrame: pandas DataFrame with email data
        model: Initialized Gemini model
        batchSize: Number of emails to analyze per batch (default: 3)
    
    Returns:
        Comprehensive analysis of email patterns
    """
    try:
        # Calculate effectiveness metrics for each email
        emailDataFrame['openRate'] = (emailDataFrame['mcopened'] / emailDataFrame['mcsent'] * 100).fillna(0)
        emailDataFrame['clickRate'] = (emailDataFrame['mcclicked'] / emailDataFrame['mcsent'] * 100).fillna(0)
        emailDataFrame['unsubRate'] = (emailDataFrame['mcunsub'] / emailDataFrame['mcsent'] * 100).fillna(0)
        
        # Sort by effectiveness score for better analysis
        emailDataFrame['effectivenessScore'] = (
            emailDataFrame['openRate'] * 0.4 + 
            emailDataFrame['clickRate'] * 0.5 - 
            emailDataFrame['unsubRate'] * 0.1
        )
        emailDataFrame = emailDataFrame.sort_values('effectivenessScore', ascending=False)
        
        # Split emails into batches
        totalEmails = len(emailDataFrame)
        allAnalyses = []
        
        for i in range(0, totalEmails, batchSize):
            batch = emailDataFrame.iloc[i:i+batchSize]
            batchNum = (i // batchSize) + 1
            totalBatches = (totalEmails + batchSize - 1) // batchSize
            
            logger.info(f"Analyzing batch {batchNum}/{totalBatches} ({len(batch)} emails)")
            
            # Prepare data for this batch
            batchData = batch[['subject', 'plaintext', 'message_body', 'openRate', 'clickRate', 'unsubRate', 'effectivenessScore']].to_dict('records')
            
            prompt = f"""
You are an expert email marketing analyst. Analyze the following email batch to identify what makes emails effective.

EMAIL BATCH {batchNum} of {totalBatches}:
{batchData}

For this batch, analyze:
1. Subject line patterns and their impact on open rates
2. Content elements (plaintext/message_body) that drive clicks
3. Factors that affect unsubscribe rates
4. Specific strengths and weaknesses of these emails

Provide a concise analysis focusing on actionable insights.
"""
            
            # Retry logic for quota errors
            maxRetries = 3
            retryDelay = 20
            
            for attempt in range(maxRetries):
                try:
                    response = model.generate_content(prompt)
                    allAnalyses.append(f"\n--- BATCH {batchNum} ANALYSIS ---\n{response.text}\n")
                    break
                except gcp_exceptions.ResourceExhausted as e:
                    if attempt < maxRetries - 1:
                        errorStr = str(e)
                        if "retry in" in errorStr.lower():
                            try:
                                match = re.search(r'retry in ([\d.]+)s', errorStr, re.IGNORECASE)
                                if match:
                                    retryDelay = float(match.group(1)) + 2
                            except:
                                pass
                        
                        logger.warning(f"Quota exceeded. Waiting {retryDelay:.1f} seconds before retry {attempt + 1}/{maxRetries}")
                        time.sleep(retryDelay)
                        retryDelay *= 1.5
                    else:
                        logger.error(f"Failed after {maxRetries} retries for batch {batchNum}")
                        raise
                except Exception as e:
                    logger.error(f"Failed to analyze batch {batchNum}: {str(e)}")
                    raise
            
            # Small delay between batches to avoid hitting rate limits
            if i + batchSize < totalEmails:
                time.sleep(2)
        
        # Final comprehensive analysis combining all batches
        logger.info("Generating comprehensive analysis from all batches")
        
        finalPrompt = f"""
You are an expert email marketing analyst. Based on the following batch analyses, provide a comprehensive summary identifying:

BATCH ANALYSES:
{''.join(allAnalyses)}

Provide a final comprehensive analysis with:
1. Overall patterns across all emails
2. Subject line best practices identified
3. Content elements that drive engagement
4. Common mistakes to avoid
5. Actionable recommendations for improving email effectiveness

Provide a clear, actionable summary.
"""
        
        # Retry for final analysis
        maxRetries = 3
        retryDelay = 20
        
        for attempt in range(maxRetries):
            try:
                finalResponse = model.generate_content(finalPrompt)
                logger.info(f"Batch analysis completed for {totalEmails} emails")
                return finalResponse.text
            except gcp_exceptions.ResourceExhausted as e:
                if attempt < maxRetries - 1:
                    errorStr = str(e)
                    if "retry in" in errorStr.lower():
                        try:
                            match = re.search(r'retry in ([\d.]+)s', errorStr, re.IGNORECASE)
                            if match:
                                retryDelay = float(match.group(1)) + 2
                        except:
                            pass
                    
                    logger.warning(f"Quota exceeded. Waiting {retryDelay:.1f} seconds before retry {attempt + 1}/{maxRetries}")
                    time.sleep(retryDelay)
                    retryDelay *= 1.5
                else:
                    logger.error(f"Failed after {maxRetries} retries for final analysis")
                    raise
        
    except Exception as e:
        logger.error(f"Failed to analyze email batch: {str(e)}")
        raise

def getEmailMarketingExpertSystemPrompt(emailDataContext=None):
    """
    Get the expert system prompt for email marketing consultation.
    Includes comprehensive knowledge about email marketing best practices.
    
    Args:
        emailDataContext: Optional context from analyzed emails
    
    Returns:
        System prompt string
    """
    basePrompt = """You are an expert email marketing consultant with deep expertise in:

1. **Email Marketing Strategy & Best Practices:**
   - Subject line optimization (A/B testing, personalization, urgency, curiosity)
   - Open rate optimization (timing, sender reputation, preheader text)
   - Click-through rate (CTR) optimization (CTA placement, copy, design)
   - Conversion rate optimization (landing page alignment, offer clarity)
   - List segmentation and targeting strategies
   - Email deliverability and inbox placement
   - CAN-SPAM, GDPR, and email compliance

2. **Content Strategy:**
   - Email copywriting (storytelling, persuasion, clarity)
   - Content structure (scannability, hierarchy, white space)
   - Visual design principles (mobile responsiveness, accessibility)
   - Personalization and dynamic content
   - Email automation and drip campaigns
   - Lifecycle marketing (welcome series, re-engagement, win-back)

3. **Performance Analysis:**
   - Key metrics interpretation (open rates, CTR, conversion, revenue per email)
   - Benchmark comparisons by industry and list size
   - A/B testing methodology and statistical significance
   - Cohort analysis and subscriber behavior patterns
   - ROI calculation and attribution modeling

4. **Advanced Techniques:**
   - Behavioral triggers and transactional emails
   - Email and social media integration
   - Cross-channel marketing alignment
   - Advanced segmentation (RFM, predictive, behavioral)
   - Email marketing psychology and consumer behavior

5. **Technical Knowledge:**
   - Email service provider (ESP) capabilities
   - HTML/CSS for email design
   - Email testing tools and methodologies
   - Authentication (SPF, DKIM, DMARC)
   - List hygiene and data quality

**Your Communication Style:**
- Be practical, actionable, and data-driven
- Provide specific examples and templates when helpful
- Explain the "why" behind recommendations
- Consider the user's audience, industry, and goals
- Balance best practices with realistic expectations
- Be encouraging but honest about challenges

**When analyzing emails, focus on:**
- Actionable improvements (not just observations)
- Prioritized recommendations (what to fix first)
- Specific examples from the data provided
- Industry benchmarks and context
- Testing suggestions for validation

"""
    
    if emailDataContext:
        basePrompt += f"""

**Current Email Performance Context:**
{emailDataContext}

Use this context to provide specific, data-driven recommendations based on actual performance.
"""
    
    return basePrompt

def chatWithEmailExpert(model, userQuestion, conversationHistory=None, emailDataContext=None):
    """
    Interactive chat function for consulting with the email marketing expert.
    
    Args:
        model: Initialized Gemini model
        userQuestion: User's question or request
        conversationHistory: List of previous messages for context
        emailDataContext: Optional context from analyzed emails
    
    Returns:
        Expert response string
    """
    try:
        systemPrompt = getEmailMarketingExpertSystemPrompt(emailDataContext)
        
        # Build conversation context
        conversationText = ""
        if conversationHistory:
            for msg in conversationHistory[-10:]:  # Keep last 10 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    conversationText += f"User: {content}\n\n"
                elif role == 'assistant':
                    conversationText += f"Expert: {content}\n\n"
        
        # Construct the full prompt
        fullPrompt = f"""{systemPrompt}

**Conversation History:**
{conversationText if conversationText else "This is the start of the conversation."}

**Current User Question:**
{userQuestion}

**Your Response:**
Provide a helpful, expert response that addresses the user's question. Be specific, actionable, and reference the email performance context if relevant.
"""
        
        # Retry logic for quota errors
        maxRetries = 3
        retryDelay = 20
        
        for attempt in range(maxRetries):
            try:
                response = model.generate_content(fullPrompt)
                logger.info("Expert consultation response generated")
                return response.text
            except gcp_exceptions.ResourceExhausted as e:
                if attempt < maxRetries - 1:
                    errorStr = str(e)
                    if "retry in" in errorStr.lower():
                        try:
                            match = re.search(r'retry in ([\d.]+)s', errorStr, re.IGNORECASE)
                            if match:
                                retryDelay = float(match.group(1)) + 2
                        except:
                            pass
                    
                    logger.warning(f"Quota exceeded. Waiting {retryDelay:.1f} seconds before retry {attempt + 1}/{maxRetries}")
                    time.sleep(retryDelay)
                    retryDelay *= 1.5
                else:
                    logger.error(f"Failed after {maxRetries} retries")
                    raise
            except Exception as e:
                logger.error(f"Failed to generate expert response: {str(e)}")
                raise
        
    except Exception as e:
        logger.error(f"Failed in chatWithEmailExpert: {str(e)}")
        raise

def analyzeSingleEmailForImprovement(model, emailContent, emailSubject=None, emailMetrics=None):
    """
    Analyze a single email and provide specific improvement recommendations.
    
    Args:
        model: Initialized Gemini model
        emailContent: The email body/content to analyze
        emailSubject: Optional subject line
        emailMetrics: Optional dict with metrics (openRate, clickRate, etc.)
    
    Returns:
        Detailed improvement recommendations
    """
    try:
        subjectSection = ""
        if emailSubject:
            subjectSection = f"""
**Subject Line:**
{emailSubject}
"""
        
        metricsSection = ""
        if emailMetrics:
            metricsSection = f"""
**Current Performance Metrics:**
- Open Rate: {emailMetrics.get('openRate', 'N/A')}%
- Click Rate: {emailMetrics.get('clickRate', 'N/A')}%
- Unsubscribe Rate: {emailMetrics.get('unsubRate', 'N/A')}%
"""
        
        systemPrompt = getEmailMarketingExpertSystemPrompt()
        
        prompt = f"""{systemPrompt}

**Email to Analyze:**
{subjectSection}
**Email Content:**
{emailContent}
{metricsSection}

**Your Task:**
Analyze this email in detail and provide specific, actionable recommendations for improvement. Focus on:

1. **Subject Line Analysis** (if provided):
   - Strengths and weaknesses
   - Specific suggestions for improvement
   - Alternative subject line options

2. **Content Analysis**:
   - Structure and readability
   - Clarity of message and value proposition
   - CTA (Call-to-Action) effectiveness
   - Engagement elements
   - Areas that need improvement

3. **Specific Recommendations**:
   - Prioritized list of improvements (most important first)
   - Before/after examples where helpful
   - Best practices to apply
   - Testing suggestions

4. **Overall Assessment**:
   - What's working well
   - What needs immediate attention
   - Expected impact of recommended changes

Provide a comprehensive, actionable analysis that the email writer can immediately use to improve this email.
"""
        
        # Retry logic for quota errors
        maxRetries = 3
        retryDelay = 20
        
        for attempt in range(maxRetries):
            try:
                response = model.generate_content(prompt)
                logger.info("Single email analysis completed")
                return response.text
            except gcp_exceptions.ResourceExhausted as e:
                if attempt < maxRetries - 1:
                    errorStr = str(e)
                    if "retry in" in errorStr.lower():
                        try:
                            match = re.search(r'retry in ([\d.]+)s', errorStr, re.IGNORECASE)
                            if match:
                                retryDelay = float(match.group(1)) + 2
                        except:
                            pass
                    
                    logger.warning(f"Quota exceeded. Waiting {retryDelay:.1f} seconds before retry {attempt + 1}/{maxRetries}")
                    time.sleep(retryDelay)
                    retryDelay *= 1.5
                else:
                    logger.error(f"Failed after {maxRetries} retries")
                    raise
            except Exception as e:
                logger.error(f"Failed to analyze single email: {str(e)}")
                raise
        
    except Exception as e:
        logger.error(f"Failed in analyzeSingleEmailForImprovement: {str(e)}")
        raise

