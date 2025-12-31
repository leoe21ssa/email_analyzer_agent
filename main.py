import logging
import sys
import io
from src.database import getEmailMessages
from src.processor import processEmailData, getTopPerformingEmails, getWorstPerformingEmails, prepareEmailDataForAnalysis
from src.agent import initializeGeminiAgent, analyzeEmailBatch

# Configure stdout to handle Unicode characters (emojis, etc.)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate email analysis workflow.
    Extracts emails, processes data, and analyzes with Gemini AI.
    """
    try:
        logger.info("Starting email analyzer agent")
        
        # Step 1: Process email data from database
        logger.info("Step 1: Processing email data")
        processedEmails = processEmailData()
        
        if processedEmails.empty:
            logger.warning("No email data to analyze")
            return
        
        # Step 2: Initialize Gemini agent
        logger.info("Step 2: Initializing Gemini agent")
        geminiModel = initializeGeminiAgent()
        
        # Step 3: Analyze email batch with Gemini
        logger.info("Step 3: Analyzing emails with Gemini AI")
        analysisResults = analyzeEmailBatch(processedEmails, geminiModel, batchSize=3)
        
        # Step 4: Display results
        logger.info("Step 4: Analysis complete")
        print("\n" + "="*80)
        print("EMAIL EFFECTIVENESS ANALYSIS RESULTS")
        print("="*80)
        print(analysisResults)
        print("="*80 + "\n")
        
        logger.info("Email analysis workflow completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()

