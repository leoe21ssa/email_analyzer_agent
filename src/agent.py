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
    basePrompt = """You are an expert email marketing consultant. You analyze real email performance data and provide SPECIFIC, QUANTIFIABLE recommendations based on actual data patterns.

**CRITICAL: Always analyze the provided email data first and give SPECIFIC recommendations based on what actually works in the data, not generic advice.**

**Your Communication Style:**
- Be practical, actionable, and DATA-DRIVEN
- Provide SPECIFIC, QUANTIFIABLE recommendations (exact numbers)
- ALWAYS reference patterns from the actual email data provided
- Compare against top-performing emails in the dataset

**MANDATORY Analysis Requirements - You MUST provide SPECIFIC recommendations for:**

1. **Email Length:**
   - Exact word count recommendation (e.g., "150-200 words based on top performers")
   - Character count for subject line (e.g., "40-50 characters have 23% higher open rates")
   - Number of paragraphs/sections
   - Compare against top-performing emails in the dataset

2. **Emojis (Quantity and Type):**
   - YES or NO recommendation with data justification
   - If YES: Specify EXACT emojis to use (e.g., "Use 📧 in subject line, ✅ in body")
   - Exact count (e.g., "Use 1-2 emojis maximum, as emails with 3+ show 15% lower click rates")
   - Placement (subject line, body, CTA)
   - Compare emoji usage in high-performing vs low-performing emails

3. **Images:**
   - YES or NO recommendation with data justification
   - Exact number of images (e.g., "Use 2-3 images based on your top performers")
   - Image placement (header, body, footer)
   - Compare image usage patterns in successful emails

4. **Call-to-Actions (CTAs):**
   - Exact number of CTAs (e.g., "Use 2 CTAs: one at 25% scroll depth, one at 75%")
   - Specific placement locations (e.g., "First CTA after paragraph 2, second before closing")
   - CTA button text recommendations (exact wording)
   - Compare CTA patterns in high-converting emails

5. **Recommended Phrases:**
   - Specific phrases that work well in your dataset
   - Opening lines that drive engagement
   - Closing phrases that convert
   - Compare phrases from top-performing emails

6. **Tone:**
   - Recommended tone (formal, casual, friendly, professional, etc.)
   - Specific examples from top-performing emails
   - Tone consistency recommendations

7. **Text Length:**
   - Paragraph length recommendations
   - Sentence length recommendations
   - Line breaks and white space
   - Mobile readability optimization

**When analyzing emails, you MUST:**
- Reference specific emails from the dataset that performed well
- Provide exact numbers and metrics (not ranges unless data shows variance)
- Compare the analyzed email against top performers
- Identify specific patterns that correlate with high open/click rates
- Give before/after examples with exact specifications

"""
    
    if emailDataContext:
        basePrompt += f"""

**Current Email Performance Context:**
{emailDataContext}

**CRITICAL INSTRUCTIONS:**
- Analyze the patterns in this data FIRST before making recommendations
- Identify the top 10-20 performing emails by open rate, click rate, and overall effectiveness
- Extract SPECIFIC patterns from these top performers:
  * Average word count
  * Emoji usage patterns (which ones, how many, where)
  * Image count and placement
  * CTA count and placement
  * Subject line characteristics (length, style, emojis)
  * Phrases and tone that work best
- Compare any email being analyzed against these top performers
- Give recommendations that match the patterns of your BEST performing emails
- Always quantify: "emails with X characteristic have Y% higher performance"

Use this context to provide specific, data-driven recommendations based on actual performance patterns in YOUR dataset.
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
Analyze this email and provide SPECIFIC, QUANTIFIABLE recommendations based on the email data patterns.

**1. Email Length:**
   - Current: [word count] words, [character count] characters in subject
   - Recommended: [exact word count] words, [exact character count] characters in subject
   - Data evidence: "Top performers average [X] words with [Y]% higher open rates"

**2. Emojis (Quantity and Type):**
   - Current: [count] emojis in [locations]
   - Recommendation: [YES/NO with data justification]
   - If YES: Use exactly [number] emojis: [list specific emojis] in [specific locations]
   - Data evidence: "Emails with [X] emojis show [Y]% [higher/lower] [metric]"

**3. Images:**
   - Current: [count] images
   - Recommendation: [YES/NO with data justification]
   - If YES: Use exactly [number] images at [specific locations]
   - Data evidence: "Top emails average [X] images"

**4. Call-to-Actions (CTAs):**
   - Current: [count] CTAs at [locations]
   - Recommended: Use exactly [number] CTAs
   - Specific placements: [exact locations, e.g., "After paragraph 2, before closing"]
   - CTA text: [exact recommended text]
   - Data evidence: "Emails with [X] CTAs at [location] have [Y]% higher click rates"

**5. Recommended Phrases:**
   - Opening phrases that work: [specific examples from top performers]
   - Body phrases that engage: [specific examples]
   - Closing phrases that convert: [specific examples]
   - Phrases to avoid: [based on low performers]

**6. Tone:**
   - Current tone: [describe]
   - Recommended tone: [specific tone] with examples from top performers
   - Tone consistency: [recommendations]

**7. Text Length & Structure:**
   - Paragraph length: [recommended words per paragraph]
   - Sentence length: [recommended words per sentence]
   - Structure: [exact number] paragraphs, [exact number] sections
   - Line breaks: [specific recommendations]

**8. Prioritized Improvements:**
   - Priority 1: [Specific change with exact numbers] - Expected impact: [X]% improvement
   - Priority 2: [Specific change with exact numbers] - Expected impact: [X]% improvement
   - Priority 3: [Specific change with exact numbers] - Expected impact: [X]% improvement

Provide a concise, DATA-DRIVEN analysis with exact specifications. Every recommendation must be backed by patterns from the actual email dataset.
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

