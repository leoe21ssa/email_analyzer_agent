import pandas as pd
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def countWords(text: Optional[str]) -> int:
    """Count words in text."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    return len(str(text).split())

def countCharacters(text: Optional[str]) -> int:
    """Count characters in text."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    return len(str(text))

def countEmojis(text: Optional[str]) -> int:
    """Count total emojis in text."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

def countDistinctEmojis(text: Optional[str]) -> int:
    """Count distinct emojis in text."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    emojis = emoji_pattern.findall(str(text))
    return len(set(emojis))

def countImages(text: Optional[str]) -> int:
    """Count total images in HTML text."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    text_str = str(text)
    # Count img tags (case insensitive)
    img_count = len(re.findall(r'<img[^>]*>', text_str, re.IGNORECASE))
    return img_count

def countImagesWithLink(text: Optional[str]) -> int:
    """Count images that are inside an <a href> tag (Option A)."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    text_str = str(text)
    
    # Find all <a> tags and check if they contain <img>
    # Pattern: <a[^>]*href[^>]*>.*?<img.*?>.*?</a>
    pattern = r'<a[^>]*href[^>]*>.*?<img[^>]*>.*?</a>'
    matches = re.findall(pattern, text_str, re.IGNORECASE | re.DOTALL)
    
    # Count images within each matched <a> tag
    count = 0
    for match in matches:
        # Count img tags within this <a> tag
        img_tags = len(re.findall(r'<img[^>]*>', match, re.IGNORECASE))
        count += img_tags
    
    return count

def countCTAs(text: Optional[str]) -> int:
    """Count total CTAs (Call-to-Actions) in text.
    Looks for button elements, href links, and CTA phrases."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    text_str = str(text).lower()
    
    # Count button elements
    button_count = text_str.count('<button') + text_str.count('</button>')
    
    # Count href links
    href_count = len(re.findall(r'href\s*=\s*["\']([^"\']+)["\']', text_str, re.IGNORECASE))
    
    # Count explicit CTA phrases (count once per phrase type)
    cta_phrases = ['click here', 'learn more', 'get started', 'sign up', 'join now', 
                  'register', 'download', 'shop now', 'buy now', 'subscribe',
                  'haz clic', 'aprende más', 'comienza', 'regístrate', 'únete',
                  'descargar', 'comprar ahora', 'suscríbete']
    phrase_count = 0
    for phrase in cta_phrases:
        if phrase in text_str:
            phrase_count += 1
    
    # Take the maximum of these indicators, cap at reasonable limit
    estimated_ctas = max(button_count // 2, min(href_count, 10), phrase_count)
    return min(estimated_ctas, 10)  # Cap at 10 CTAs max

def countDistinctCTAs(text: Optional[str]) -> int:
    """Count distinct CTAs by extracting unique URLs."""
    if pd.isna(text) or text == '' or text is None:
        return 0
    text_str = str(text)
    
    # Extract all href URLs
    urls = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', text_str, re.IGNORECASE)
    
    # Also extract URLs from button onclick or data attributes
    button_urls = re.findall(r'(?:onclick|data-url|data-href)\s*=\s*["\']([^"\']+)["\']', text_str, re.IGNORECASE)
    urls.extend(button_urls)
    
    # Get distinct URLs (normalize by removing query params and fragments for comparison)
    distinct_urls = set()
    for url in urls:
        # Remove query params and fragments for comparison
        clean_url = url.split('?')[0].split('#')[0]
        if clean_url:  # Only count non-empty URLs
            distinct_urls.add(clean_url.lower())
    
    return len(distinct_urls)

def calculateSubjectMetrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all subject-related metrics."""
    df['subject_words'] = df['subject'].apply(countWords)
    df['subject_chars'] = df['subject'].apply(countCharacters)
    df['subject_emojis'] = df['subject'].apply(countEmojis)
    return df

def calculateBodyMetrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all body-related metrics from plaintext.
    Note: For text metrics (words, chars, emojis) we use plaintext.
    For HTML metrics (images, CTAs) we use message_body since they require HTML tags."""
    # Text metrics from plaintext (user creates plaintext)
    df['body_words'] = df['plaintext'].apply(countWords)
    df['body_chars'] = df['plaintext'].apply(countCharacters)
    df['body_emojis_total'] = df['plaintext'].apply(countEmojis)
    df['body_emojis_distinct'] = df['plaintext'].apply(countDistinctEmojis)
    
    # HTML metrics from message_body (images and CTAs require HTML)
    df['body_images_total'] = df['message_body'].apply(countImages)
    df['body_images_with_link'] = df['message_body'].apply(countImagesWithLink)
    df['body_ctas_total'] = df['message_body'].apply(countCTAs)
    df['body_ctas_distinct'] = df['message_body'].apply(countDistinctCTAs)
    
    return df

def calculateAllMetrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all quantifiable metrics for emails."""
    df = calculateSubjectMetrics(df)
    df = calculateBodyMetrics(df)
    return df

