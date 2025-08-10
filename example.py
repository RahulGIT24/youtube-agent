"""
Example usage of the YouTube to Blog converter.

This script demonstrates how to use the BlogWorkflow class programmatically.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import BlogWorkflow
from config import AppConfig
from utils import validate_youtube_url, get_video_id_from_url

def example_basic_usage():
    """Example of basic usage."""
    print("🎬 YouTube to Blog Converter - Basic Example")
    print("=" * 50)
    
    # Example video URL
    video_url = "https://www.youtube.com/watch?v=37srVu0q5o0"
    
    # Validate URL
    if not validate_youtube_url(video_url):
        print(f"❌ Invalid YouTube URL: {video_url}")
        return
    
    try:
        # Initialize workflow
        print("🚀 Initializing workflow...")
        workflow = BlogWorkflow()
        
        # Process video
        print(f"📺 Processing video: {video_url}")
        response = workflow.process_video(
            url=video_url,
            task="Create a comprehensive blog post about this video content"
        )
        
        # Check for errors
        if response.get('error_message'):
            print(f"❌ Error: {response['error_message']}")
            return
        
        # Get blog content
        blog_content = response.get('final_blog', '')
        if blog_content:
            print(f"✅ Blog generated successfully!")
            print(f"📊 Blog length: {len(blog_content)} characters")
            print(f"📝 Preview: {blog_content[:200]}...")
            
            # Save to file
            output_file = "example_blog.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(blog_content)
            print(f"💾 Blog saved to: {output_file}")
        else:
            print("⚠️  No blog content generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_with_custom_config():
    """Example with custom configuration."""
    print("\n🎬 YouTube to Blog Converter - Custom Config Example")
    print("=" * 50)
    
    try:
        # Create custom configuration
        config = AppConfig()
        
        # Customize settings
        config.processing.max_search_results = 5
        config.processing.max_transcript_length = 8000
        config.output.output_directory = "custom_output"
        
        # Initialize workflow with custom config
        print("🚀 Initializing workflow with custom config...")
        workflow = BlogWorkflow()
        
        # Example video URL
        video_url = "https://www.youtube.com/watch?v=37srVu0q5o0"
        video_id = get_video_id_from_url(video_url)
        
        print(f"📺 Processing video ID: {video_id}")
        
        # Process with custom task
        response = workflow.process_video(
            url=video_url,
            task="Create a technical blog post with detailed analysis and insights"
        )
        
        if response.get('error_message'):
            print(f"❌ Error: {response['error_message']}")
            return
        
        # Export with custom settings
        from main import BlogExporter
        exporter = BlogExporter(workflow.config)
        output_files = exporter.export_blog(response.get('final_blog', ''), video_id)
        
        print("✅ Blog generated with custom configuration!")
        print(f"📄 Markdown: {output_files['markdown']}")
        print(f"🌐 HTML: {output_files['html']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_batch_processing():
    """Example of processing multiple videos."""
    print("\n🎬 YouTube to Blog Converter - Batch Processing Example")
    print("=" * 50)
    
    # Example video URLs
    video_urls = [
        "https://www.youtube.com/watch?v=37srVu0q5o0",
        # Add more URLs here
    ]
    
    try:
        workflow = BlogWorkflow()
        
        for i, url in enumerate(video_urls, 1):
            print(f"\n📺 Processing video {i}/{len(video_urls)}: {url}")
            
            if not validate_youtube_url(url):
                print(f"❌ Invalid URL, skipping: {url}")
                continue
            
            video_id = get_video_id_from_url(url)
            
            # Process video
            response = workflow.process_video(
                url=url,
                task=f"Create blog post for video {i}"
            )
            
            if response.get('error_message'):
                print(f"❌ Error processing {url}: {response['error_message']}")
                continue
            
            # Export blog
            from main import BlogExporter
            exporter = BlogExporter(workflow.config)
            output_files = exporter.export_blog(response.get('final_blog', ''), video_id)
            
            print(f"✅ Video {i} processed successfully!")
            print(f"📄 Output: {output_files['markdown']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all examples."""
    print("🎬 YouTube to Blog Converter Examples")
    print("=" * 50)
    
    # Check if API keys are set
    if not os.getenv('GROQ_API_KEY') or not os.getenv('TAVILY_API_KEY'):
        print("❌ Please set GROQ_API_KEY and TAVILY_API_KEY environment variables")
        print("Example:")
        print("export GROQ_API_KEY='your_groq_api_key'")
        print("export TAVILY_API_KEY='your_tavily_api_key'")
        return
    
    # Run examples
    example_basic_usage()
    example_with_custom_config()
    example_batch_processing()
    
    print("\n🎉 All examples completed!")

if __name__ == "__main__":
    main()
