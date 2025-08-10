"""
Command-line interface for the YouTube to Blog converter.
"""

import argparse
import sys
import logging

from main import BlogWorkflow
from config import AppConfig
from utils import validate_youtube_url, get_video_id_from_url

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert YouTube videos to SEO-optimized blog posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py https://www.youtube.com/watch?v=VIDEO_ID
  python cli.py https://www.youtube.com/watch?v=VIDEO_ID --task "Create a technical blog post"
  python cli.py https://www.youtube.com/watch?v=VIDEO_ID --verbose --output-dir ./blogs
        """
    )
    
    parser.add_argument(
        'url',
        help='YouTube video URL to convert to blog post'
    )
    
    parser.add_argument(
        '--task',
        default='Create a blog from this video',
        help='Task description for the AI agents (default: "Create a blog from this video")'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for generated files (default: output)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate URL and configuration without processing'
    )
    
    return parser.parse_args()

def validate_input(url: str) -> bool:
    """Validate input URL."""
    if not url:
        print("❌ Error: URL is required")
        return False
    
    if not validate_youtube_url(url):
        print(f"❌ Error: Invalid YouTube URL: {url}")
        print("Please provide a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)")
        return False
    
    return True

def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not validate_input(args.url):
        sys.exit(1)
    
    try:
        # Initialize configuration
        config = AppConfig()
        
        # Override output directory if specified
        if args.output_dir:
            config.output.output_directory = args.output_dir
        
        # Extract video ID
        video_id = get_video_id_from_url(args.url)
        if video_id:
            logger.info(f"Processing video ID: {video_id}")
        
        if args.dry_run:
            print("✅ Configuration and URL validation successful!")
            print(f"📺 Video URL: {args.url}")
            print(f"📝 Task: {args.task}")
            print(f"📁 Output directory: {config.output.output_directory}")
            return
        
        # Initialize workflow
        logger.info("Initializing blog generation workflow...")
        workflow = BlogWorkflow()
        
        # Process the video
        logger.info(f"Starting blog generation for: {args.url}")
        response = workflow.process_video(args.url, args.task)
        
        # Check for errors
        if response.get('error_message'):
            logger.error(f"Workflow completed with error: {response['error_message']}")
            print(f"❌ Error: {response['error_message']}")
            sys.exit(1)
        
        # Export blog content
        blog_content = response.get('final_blog', '')
        if blog_content:
            from main import BlogExporter
            exporter = BlogExporter(workflow.config)
            output_files = exporter.export_blog(blog_content, video_id)
            
            print("\n✅ Blog generation completed successfully!")
            print(f"📄 Markdown file: {output_files['markdown']}")
            print(f"🌐 HTML file: {output_files['html']}")
            print(f"📊 Blog length: {len(blog_content)} characters")
        else:
            logger.warning("No blog content generated")
            print("⚠️  Warning: No blog content was generated")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
