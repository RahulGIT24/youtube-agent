# YouTube to Blog Converter

A sophisticated multi-agent system that converts YouTube videos into SEO-optimized blog posts using AI-powered transcript extraction, web research, and content generation.

## 🚀 Features

- **Multi-Agent Workflow**: Intelligent agent coordination for optimal blog generation
- **Transcript Extraction**: Automatic YouTube video transcript extraction
- **Web Research**: AI-powered keyword extraction and web search for content enrichment
- **SEO Optimization**: Professional blog writing with SEO best practices
- **Multiple Output Formats**: Generate both Markdown and HTML files
- **Error Handling**: Robust error handling and validation
- **CLI Interface**: Easy-to-use command-line interface
- **Configurable**: Flexible configuration system

## 🏗️ Architecture

The system uses a multi-agent workflow with the following components:

1. **Supervisor Agent**: Orchestrates the workflow and assigns tasks
2. **Transcriptor Agent**: Extracts transcripts from YouTube videos
3. **Analyzer Agent**: Performs web research and keyword analysis
4. **Writer Agent**: Creates SEO-optimized blog content

## 📋 Requirements

- Python 3.12+
- GROQ API Key
- Tavily API Key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ytblog
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
```

Or create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🚀 Usage

### Command Line Interface (Recommended)

```bash
# Basic usage
python cli.py https://www.youtube.com/watch?v=VIDEO_ID

# With custom task
python cli.py https://www.youtube.com/watch?v=VIDEO_ID --task "Create a technical blog post about AI"

# Verbose logging
python cli.py https://www.youtube.com/watch?v=VIDEO_ID --verbose

# Custom output directory
python cli.py https://www.youtube.com/watch?v=VIDEO_ID --output-dir ./my_blogs

# Dry run (validate without processing)
python cli.py https://www.youtube.com/watch?v=VIDEO_ID --dry-run
```

### Programmatic Usage

```python
from main import BlogWorkflow

# Initialize workflow
workflow = BlogWorkflow()

# Process video
response = workflow.process_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    task="Create a blog from this video"
)

# Get blog content
blog_content = response.get('final_blog', '')
print(blog_content)
```

## 📁 Project Structure

```
ytblog/
├── main.py          # Main application logic
├── config.py        # Configuration management
├── utils.py         # Utility functions
├── cli.py           # Command-line interface
├── pyproject.toml   # Project dependencies
└── README.md        # This file
```

## ⚙️ Configuration

The application is highly configurable through the `config.py` file:

### Model Configuration
- `groq_model`: AI model to use (default: 'llama-3.1-8b-instant')
- `max_tokens`: Maximum tokens for AI responses
- `temperature`: AI creativity level

### Processing Configuration
- `max_transcript_length`: Maximum transcript length for blog generation
- `max_search_results`: Number of web search results to include
- `chunk_size_seconds`: Video chunk size for transcript extraction
- `keyword_extraction_limit`: Character limit for keyword extraction

### Output Configuration
- `output_directory`: Directory for generated files
- `markdown_filename`: Default markdown filename
- `html_filename`: Default HTML filename

## 🔧 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black .
isort .
```

### Type Checking
```bash
mypy .
```

## 📝 Output

The application generates:

1. **Markdown File**: SEO-optimized blog post in Markdown format
2. **HTML File**: Rendered HTML version of the blog post

Both files include:
- Meta description
- SEO-friendly tags
- Structured headings
- Professional formatting
- Video embedding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Invalid YouTube URL**: Ensure the URL is a valid YouTube video URL
2. **API Key Errors**: Verify your GROQ and Tavily API keys are set correctly
3. **No Transcript**: Some videos may not have available transcripts
4. **Rate Limiting**: Respect API rate limits for both GROQ and Tavily

### Debug Mode

Enable verbose logging to debug issues:
```bash
python cli.py <url> --verbose
```

## 🔮 Future Enhancements

- [ ] Support for multiple video formats
- [ ] Custom blog templates
- [ ] Image generation for blog posts
- [ ] Social media integration
- [ ] Analytics and performance tracking
- [ ] Batch processing capabilities
