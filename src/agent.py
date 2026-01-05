import os
import logging
import time
import re
import pandas as pd
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
You are an expert email marketing analyst. Analyze the following emails to identify what makes them effective.

EMAILS TO ANALYZE:
{batchData}

For these emails, analyze:
1. Subject line patterns and their impact on open rates
2. Content elements (plaintext/message_body) that drive clicks
3. Factors that affect unsubscribe rates
4. Specific strengths and weaknesses of these emails

Provide a concise analysis focusing on actionable insights. Do NOT mention "batch" or "batch number" - refer to emails naturally.
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
        
        # Helper functions to calculate metrics (for context, not strict statistics)
        def countWords(text):
            if pd.isna(text) or text == '':
                return 0
            return len(str(text).split())
        
        def countEmojis(text):
            if pd.isna(text) or text == '':
                return 0
            # Simple emoji detection
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            return len(emoji_pattern.findall(str(text)))
        
        def countCTAs(text):
            """
            Count actual CTAs more accurately.
            Looks for button patterns, href links, and CTA phrases, but counts each unique CTA area.
            """
            if pd.isna(text) or text == '':
                return 0
            text_str = str(text).lower()
            
            # Count button elements (more reliable indicator of actual CTAs)
            button_count = text_str.count('<button') + text_str.count('button>')
            
            # Count href links (but limit to reasonable number)
            href_count = text_str.count('href=')
            
            # Count explicit CTA phrases (but only once per phrase type to avoid overcounting)
            cta_phrases = ['click here', 'learn more', 'get started', 'sign up', 'join now', 
                          'register', 'download', 'shop now', 'buy now', 'subscribe']
            phrase_count = 0
            for phrase in cta_phrases:
                if phrase in text_str:
                    phrase_count += 1  # Count once per phrase type, not each occurrence
            
            # Take the maximum of these indicators, but cap at reasonable limit
            # Most emails have 1-3 CTAs, rarely more than 5
            estimated_ctas = max(button_count // 2, min(href_count, 5), phrase_count)
            return min(estimated_ctas, 5)  # Cap at 5 CTAs max
        
        def countImages(text):
            """
            Count images in email content by looking for img tags.
            """
            if pd.isna(text) or text == '':
                return 0
            text_str = str(text)
            
            # Count img tags (case insensitive)
            img_count = len(re.findall(r'<img[^>]*>', text_str, re.IGNORECASE))
            
            return img_count
        
        # Calculate basic metrics for context (calculate FIRST before selecting top/worst)
        # Use plaintext primarily (it's the actual readable text), fallback to message_body if plaintext is empty
        def getWordCount(row):
            plaintext_words = countWords(row.get('plaintext', ''))
            if plaintext_words > 0:
                return plaintext_words
            # Only use message_body if plaintext is empty
            return countWords(row.get('message_body', ''))
        
        emailDataFrame['word_count'] = emailDataFrame.apply(getWordCount, axis=1)
        emailDataFrame['subject_char_count'] = emailDataFrame['subject'].str.len()
        emailDataFrame['emoji_count'] = emailDataFrame['subject'].apply(countEmojis) + emailDataFrame['plaintext'].apply(countEmojis) + emailDataFrame['message_body'].apply(countEmojis)
        emailDataFrame['cta_count'] = emailDataFrame['plaintext'].apply(countCTAs) + emailDataFrame['message_body'].apply(countCTAs)
        emailDataFrame['image_count'] = emailDataFrame['plaintext'].apply(countImages) + emailDataFrame['message_body'].apply(countImages)
        
        # Cap CTA count at reasonable maximum (most emails have 1-3 CTAs, rarely more than 5)
        emailDataFrame['cta_count'] = emailDataFrame['cta_count'].clip(upper=5)
        
        # NOW get top and worst emails after calculating columns
        topEmails = emailDataFrame.nlargest(5, 'effectivenessScore')
        worstEmails = emailDataFrame.nsmallest(5, 'effectivenessScore')
        
        # Calculate statistics for top 10 and worst 10 performers
        top10 = emailDataFrame.nlargest(10, 'effectivenessScore')
        worst10 = emailDataFrame.nsmallest(10, 'effectivenessScore')
        
        # Calculate statistics for context
        topStats = {}
        worstStats = {}
        
        if len(top10) > 0:
            topStats = {
                'avg_open_rate': top10['openRate'].mean(),
                'avg_click_rate': top10['clickRate'].mean(),
                'avg_unsub_rate': top10['unsubRate'].mean(),
                'avg_word_count': top10['word_count'].mean(),
                'avg_subject_length': top10['subject_char_count'].mean(),
                'avg_emoji_count': top10['emoji_count'].mean(),
                'avg_cta_count': top10['cta_count'].mean(),
                'avg_image_count': top10['image_count'].mean(),
            }
        
        if len(worst10) > 0:
            worstStats = {
                'avg_open_rate': worst10['openRate'].mean(),
                'avg_click_rate': worst10['clickRate'].mean(),
                'avg_unsub_rate': worst10['unsubRate'].mean(),
                'avg_word_count': worst10['word_count'].mean(),
                'avg_subject_length': worst10['subject_char_count'].mean(),
                'avg_emoji_count': worst10['emoji_count'].mean(),
                'avg_cta_count': worst10['cta_count'].mean(),
                'avg_image_count': worst10['image_count'].mean(),
            }
        
        # Prepare simple examples for context
        topEmailExamples = []
        worstEmailExamples = []
        
        if len(topEmails) > 0:
            for idx, row in topEmails.head(3).iterrows():
                topEmailExamples.append({
                    'subject': str(row.get('subject', 'N/A'))[:100],
                    'openRate': f"{row.get('openRate', 0):.2f}%",
                    'clickRate': f"{row.get('clickRate', 0):.2f}%",
                    'wordCount': int(row.get('word_count', 0))
                })
        
        if len(worstEmails) > 0:
            for idx, row in worstEmails.head(3).iterrows():
                worstEmailExamples.append({
                    'subject': str(row.get('subject', 'N/A'))[:100],
                    'openRate': f"{row.get('openRate', 0):.2f}%",
                    'clickRate': f"{row.get('clickRate', 0):.2f}%",
                    'wordCount': int(row.get('word_count', 0))
                })
        
        finalPrompt = f"""You are an expert email marketing analyst. Based on the following analyses of actual emails, provide a structured analysis in TWO CLEAR SECTIONS.

EMAIL ANALYSES:
{''.join(allAnalyses)}

TOP 10 PERFORMING EMAILS STATISTICS:
- Average Open Rate: {topStats.get('avg_open_rate', 0):.2f}%
- Average Click Rate: {topStats.get('avg_click_rate', 0):.2f}%
- Average Unsubscribe Rate: {topStats.get('avg_unsub_rate', 0):.2f}%
- Average Word Count: {topStats.get('avg_word_count', 0):.0f} words
- Average Subject Length: {topStats.get('avg_subject_length', 0):.0f} characters
- Average Emoji Count: {topStats.get('avg_emoji_count', 0):.1f} emojis
- Average CTA Count: {topStats.get('avg_cta_count', 0):.1f} CTAs
- Average Image Count: {topStats.get('avg_image_count', 0):.1f} images

WORST 10 PERFORMING EMAILS STATISTICS:
- Average Open Rate: {worstStats.get('avg_open_rate', 0):.2f}%
- Average Click Rate: {worstStats.get('avg_click_rate', 0):.2f}%
- Average Unsubscribe Rate: {worstStats.get('avg_unsub_rate', 0):.2f}%
- Average Word Count: {worstStats.get('avg_word_count', 0):.0f} words
- Average Subject Length: {worstStats.get('avg_subject_length', 0):.0f} characters
- Average Emoji Count: {worstStats.get('avg_emoji_count', 0):.1f} emojis
- Average CTA Count: {worstStats.get('avg_cta_count', 0):.1f} CTAs
- Average Image Count: {worstStats.get('avg_image_count', 0):.1f} images

TOP PERFORMING EMAIL EXAMPLES:
{topEmailExamples}

WORST PERFORMING EMAIL EXAMPLES:
{worstEmailExamples}

**CRITICAL INSTRUCTIONS:**
You MUST provide your analysis in TWO CLEAR SECTIONS. Base ALL recommendations on the ACTUAL EMAIL ANALYSIS and STATISTICS above, NOT on generic best practices. 

**IMPORTANT: Do NOT mention "batch", "Batch 1", "Batch 2", or any batch numbers in your response. This is internal processing information that should not appear to the user. Instead, refer to emails naturally as "top performing emails", "some emails", "specific emails", "the analyzed emails", etc.**

**MANDATORY: You MUST include specific numbers and percentages in your analysis:**
- In PROS/CONTRAS: Include actual open rates, click rates, or other metrics when mentioning performance
- In Email Length: Provide EXACT word count (e.g., "250-350 words"), not vague descriptions
- Reference the statistics provided above to give weight to your analysis

---

## SECTION 1: ANALYSIS SUMMARY (Pros and Cons)

Provide a CONCISE summary (maximum 250 words) analyzing what you found in these emails. Structure it as:

**PROS (What Works Well):**
- List 3-5 specific strengths you found in the top performing emails
- Be specific: mention actual patterns, styles, or elements you observed
- **MUST include specific percentages or metrics** (e.g., "emails with X achieve Y% open rate" or "top performers average X% click rate")
- Reference the statistics provided above (e.g., "Top performers average {topStats.get('avg_open_rate', 0):.2f}% open rate")
- Do NOT mention batch numbers - refer to emails naturally

**CONS (What Needs Improvement):**
- List 3-5 specific weaknesses you found in the worst performing emails
- Be specific: mention actual problems, patterns, or missing elements you observed
- **MUST include specific percentages or metrics** (e.g., "emails with X have only Y% click rate" or "worst performers average X% open rate")
- Reference the statistics provided above (e.g., "Worst performers average only {worstStats.get('avg_click_rate', 0):.2f}% click rate")
- Do NOT mention batch numbers - refer to emails naturally

Keep it concrete and data-driven, using the statistics provided above to give weight to your observations.

---

## SECTION 2: SPECIFIC RECOMMENDATIONS

Based on your analysis of the actual emails and the statistics above, provide SPECIFIC, ACTIONABLE recommendations for each of the following. Be direct and specific. Do NOT mention batch numbers.

**Email Length:**
- Recommended word count: [MUST provide EXACT number or range like "250-350 words" based on top performers' average: {topStats.get('avg_word_count', 0):.0f} words. NOTE: If the average seems unusually high (>1000 words), use a reasonable range like "200-500 words" and explain that emails should be concise and scannable]
- Why: [Brief explanation with reference to the statistics - e.g., "Top performers average {topStats.get('avg_word_count', 0):.0f} words with {topStats.get('avg_open_rate', 0):.2f}% open rate"]

**Call-to-Actions (CTAs):**
- How many CTAs: [Specific number based on top performers' average: {topStats.get('avg_cta_count', 0):.1f} CTAs. NOTE: Most effective emails use 1-3 CTAs. If the average seems unusually high (>5), recommend 2-3 CTAs and explain that too many CTAs can create decision fatigue]
- Same CTA or different: [Should they use the same CTA text multiple times, or different CTAs? Explain based on what works in the analyzed emails]
- CTA placement: [Where should CTAs be placed? Be specific based on top performers]

**Images:**
- Should emails have images: [YES or NO, with brief explanation. Reference the statistics: top performers average {topStats.get('avg_image_count', 0):.1f} images]
- How many images: [Specific number if YES, based on what you observed. Reference the average: {topStats.get('avg_image_count', 0):.1f} images for top performers]
- Image placement: [Where should images be placed? Reference patterns from top performers if available]

**Tone:**
- Recommended tone: [Specific tone: formal, casual, friendly, professional, etc. based on top performers]
- Why: [Brief explanation with reference to performance metrics if relevant]

**Emojis:**
- Should use emojis: [YES or NO, with brief explanation]
- How many emojis: [If YES, give specific number or range based on top performers' average: {topStats.get('avg_emoji_count', 0):.1f} emojis]
- How to use them: [Where to place them: subject line, body, CTAs? Be specific]

**Subject Line:**
- Recommended length: [Specific character count or range based on top performers' average: {topStats.get('avg_subject_length', 0):.0f} characters]
- Style: [What style works best based on top performers]
- Emoji in subject: [YES or NO, with explanation]

Every recommendation MUST be based on what you actually observed in the analyzed emails and the statistics provided above, not generic advice. Always include specific numbers.
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

