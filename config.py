"""
Configuration settings for the YouTube to Blog converter.
"""

import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    groq_model: str = 'llama-3.1-8b-instant'
    max_tokens: int = 4000
    temperature: float = 0.7

@dataclass
class ProcessingConfig:
    """Configuration for content processing."""
    max_transcript_length: int = 10000
    max_search_results: int = 3
    chunk_size_seconds: int = 60
    keyword_extraction_limit: int = 500

@dataclass
class OutputConfig:
    """Configuration for output files."""
    markdown_filename: str = "generated_blog.md"
    html_filename: str = "generated_blog.html"
    output_directory: str = "output"

@dataclass
class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        # API Keys
        self.groq_api_key: Optional[str] = os.getenv('GROQ_API_KEY')
        self.tavily_api_key: Optional[str] = os.getenv('TAVILY_API_KEY')
        
        # Model configuration
        self.model = ModelConfig()
        
        # Processing configuration
        self.processing = ProcessingConfig()
        
        # Output configuration
        self.output = OutputConfig()
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration values."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
    
    def get_env_vars(self) -> dict:
        """Get environment variables for LangChain."""
        return {
            'GROQ_API_KEY': self.groq_api_key,
            'TAVILY_API_KEY': self.tavily_api_key
        }
