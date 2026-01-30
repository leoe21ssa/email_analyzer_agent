import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
Base = declarative_base()

class Message(Base):
    """
    SQLAlchemy model representing the messages table.
    Maps to the existing messages table in the database.
    """
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True)
    dlm = Column(DateTime)
    date = Column(DateTime)
    alias = Column(String)
    mcsent = Column(BigInteger)
    mcunsub = Column(BigInteger)
    mcabuse = Column(BigInteger)
    subject = Column(String)
    mcopened = Column(BigInteger)
    mcclicked = Column(BigInteger)
    plaintext = Column(Text)
    message_body = Column(Text)
    old_resource = Column(String)

def getDatabaseEngine():
    """
    Create database engine using environment variables.
    Returns SQLAlchemy engine for database operations.
    """
    dbHost = os.getenv("DB_HOST")
    dbPort = os.getenv("DB_PORT")
    dbName = os.getenv("DB_NAME")
    dbUser = os.getenv("DB_USER")
    dbPassword = os.getenv("DB_PASSWORD")
    
    if not all([dbHost, dbPort, dbName, dbUser, dbPassword]):
        logger.error("Missing required database environment variables")
        raise ValueError("All database environment variables must be set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
    
    # Construct database URL (assuming PostgreSQL, adjust if needed)
    dbUrl = f"postgresql://{dbUser}:{dbPassword}@{dbHost}:{dbPort}/{dbName}"
    
    try:
        engine = create_engine(dbUrl)
        logger.info(f"Database engine created for {dbHost}:{dbPort}/{dbName}")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise

def getDatabaseSession():
    """
    Create database session using SQLAlchemy ORM.
    Returns a session object for database operations.
    """
    engine = getDatabaseEngine()
    Session = sessionmaker(bind=engine)
    session = Session()
    return session, engine

def getEmailMessages():
    """
    Extract email messages from the messages table using SQLAlchemy ORM.
    Filters by specific email IDs.
    Returns a pandas DataFrame with email data and metrics.
    """
    session, engine = getDatabaseSession()
    
    # Get email IDs from environment variable
    emailIdsStr = os.getenv("EMAIL_IDS", "").strip()
    if emailIdsStr:
        # Split by comma, strip whitespace, and filter out empty strings
        targetIds = [id.strip() for id in emailIdsStr.split(',') if id.strip()]
        logger.info(f"Filtering emails by IDs from environment: {targetIds}")
    else:
        targetIds = None
        logger.info("No EMAIL_IDS specified in environment - analyzing all emails")

    try:
        if targetIds:
            # Use SQLAlchemy ORM to query messages with ID filter
            messages = session.query(Message).filter(Message.id.in_(targetIds)).all()
            logger.info(f"Filtered query: Found {len(messages)} emails matching IDs")
        else:
            # Get all emails if no IDs specified
            messages = session.query(Message).all()
            logger.info(f"Unfiltered query: Found {len(messages)} total emails")
        
        # Convert to list of dictionaries
        messagesData = [
            {
                'id': msg.id,
                'dlm': msg.dlm,
                'date': msg.date,
                'alias': msg.alias,
                'mcsent': msg.mcsent,
                'mcunsub': msg.mcunsub,
                'mcabuse': msg.mcabuse,
                'subject': msg.subject,
                'mcopened': msg.mcopened,
                'mcclicked': msg.mcclicked,
                'plaintext': msg.plaintext,
                'message_body': msg.message_body,
                'old_resource': msg.old_resource
            }
            for msg in messages
        ]
        
        df = pd.DataFrame(messagesData)

        if targetIds:
            logger.info(f"Successfully extracted {len(df)} email messages from database (filtered by IDs: {targetIds})")
        else:
            logger.info(f"Successfully extracted {len(df)} email messages from database (all emails)")
        
        return df
    except Exception as e:
        logger.error(f"Failed to extract email messages: {str(e)}")
        raise
    finally:
        session.close()
        engine.dispose()