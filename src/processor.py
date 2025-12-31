import pandas as pd
import logging
from src.database import getEmailMessages

logger = logging.getLogger(__name__)

def processEmailData():
    """
    Process email data from database.
    Extracts emails, calculates metrics, and prepares data for analysis.
    Returns processed DataFrame with calculated effectiveness metrics.
    """
    try:
        logger.info("Starting email data processing")
        df = getEmailMessages()
        
        if df.empty:
            logger.warning("No email data found in database")
            return df
        
        # Convert numeric columns to numeric type (in case they come as strings)
        numericColumns = ['mcsent', 'mcopened', 'mcclicked', 'mcunsub']
        for col in numericColumns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate effectiveness metrics
        df['openRate'] = (df['mcopened'] / df['mcsent'] * 100).fillna(0)
        df['clickRate'] = (df['mcclicked'] / df['mcsent'] * 100).fillna(0)
        df['unsubRate'] = (df['mcunsub'] / df['mcsent'] * 100).fillna(0)
        
        # Filter out emails with zero sends to avoid division issues
        df = df[df['mcsent'] > 0]
        
        # Add effectiveness score (weighted combination of metrics)
        # Higher open and click rates are good, lower unsub rate is good
        df['effectivenessScore'] = (
            df['openRate'] * 0.4 + 
            df['clickRate'] * 0.5 - 
            df['unsubRate'] * 0.1
        )
        
        logger.info(f"Processed {len(df)} emails with effectiveness metrics")
        return df
    except Exception as e:
        logger.error(f"Failed to process email data: {str(e)}")
        raise

def getTopPerformingEmails(df, topN=20):
    """
    Get top performing emails based on effectiveness score.
    
    Args:
        df: Processed email DataFrame
        topN: Number of top emails to return
    
    Returns:
        DataFrame with top performing emails
    """
    try:
        topEmails = df.nlargest(topN, 'effectivenessScore')
        logger.info(f"Retrieved top {len(topEmails)} performing emails")
        return topEmails
    except Exception as e:
        logger.error(f"Failed to get top performing emails: {str(e)}")
        raise

def getWorstPerformingEmails(df, worstN=20):
    """
    Get worst performing emails based on effectiveness score.
    
    Args:
        df: Processed email DataFrame
        worstN: Number of worst emails to return
    
    Returns:
        DataFrame with worst performing emails
    """
    try:
        worstEmails = df.nsmallest(worstN, 'effectivenessScore')
        logger.info(f"Retrieved worst {len(worstEmails)} performing emails")
        return worstEmails
    except Exception as e:
        logger.error(f"Failed to get worst performing emails: {str(e)}")
        raise

def prepareEmailDataForAnalysis(df):
    """
    Prepare email data in a format suitable for Gemini analysis.
    Selects relevant fields and formats them for the AI agent.
    
    Args:
        df: Processed email DataFrame
    
    Returns:
        List of dictionaries with formatted email data
    """
    try:
        # Select relevant columns for analysis
        analysisColumns = [
            'id', 'subject', 'plaintext', 'message_body', 
            'openRate', 'clickRate', 'unsubRate', 
            'mcsent', 'mcopened', 'mcclicked', 'mcunsub',
            'effectivenessScore'
        ]
        
        # Filter to only include columns that exist
        availableColumns = [col for col in analysisColumns if col in df.columns]
        formattedData = df[availableColumns].to_dict('records')
        
        logger.info(f"Prepared {len(formattedData)} emails for analysis")
        return formattedData
    except Exception as e:
        logger.error(f"Failed to prepare email data for analysis: {str(e)}")
        raise

