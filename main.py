"""
YouTube to Blog Converter

A multi-agent system that converts YouTube videos into SEO-optimized blog posts
using transcript extraction, web research, and AI-powered content generation.
"""

import os
import logging
from typing import Dict, Literal, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, MessagesState
import markdown

from config import AppConfig
from utils import validate_youtube_url, ensure_directory_exists, sanitize_filename, get_video_id_from_url, clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SupervisorState(MessagesState):
    """State management for the multi-agent workflow."""
    
    next_agent: str = ''
    transcript_data: str = ''
    analyzed_data: str = ''
    final_blog: str = ''
    task_complete: bool = False
    current_task: str = ''
    input_url: str = ''
    error_message: str = ''



class SupervisorAgent:
    """Manages the workflow and assigns tasks to appropriate agents."""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prompt = self._create_supervisor_prompt()
    
    def _create_supervisor_prompt(self) -> ChatPromptTemplate:
        """Create the supervisor prompt template."""
        return ChatPromptTemplate.from_messages([
            ('system', """
You are an AI supervisor managing a multi-agent system for creating blog posts from YouTube videos.

Available Agents:
1. transcriptor — extracts transcript from YouTube video
2. analyzer — searches the web for relevant topics and insights
3. writer — creates SEO-optimized blog post from transcript and research

Current State:
- Transcript available: {has_transcript}
- Analysis complete: {has_analyzed}
- Blog written: {has_blog}

TASK: {task}

RESPOND ONLY with one of the following (lowercase):
- transcriptor
- analyzer
- writer
- done
"""),
            ("human", "{task}")
        ])
    
    def decide_next_agent(self, state: SupervisorState) -> Dict:
        """Decide which agent should handle the next task."""
        messages = state.get('messages', [])
        task = messages[-1].content if messages else 'No Task'
        
        has_transcript = bool(state.get('transcript_data', ''))
        has_analyzed = bool(state.get('analyzed_data', ''))
        has_blog = bool(state.get('final_blog', ''))
        
        chain = self.prompt | self.llm
        
        try:
            decision = chain.invoke({
                'task': task,
                'has_transcript': has_transcript,
                'has_analyzed': has_analyzed,
                'has_blog': has_blog
            })
            
            decision_text = decision.content.strip().lower()
            logger.info(f"Supervisor decision: {decision_text}")
            
            if "done" in decision_text or has_blog:
                next_agent = 'end'
                supervisor_message = "Supervisor: All tasks completed successfully! 🎉"
            elif "transcriptor" in decision_text or not has_transcript:
                next_agent = 'transcriptor'
                supervisor_message = "Supervisor: Assigning Transcriptor to extract video transcript"
            elif "analyzer" in decision_text or not has_analyzed:
                next_agent = 'analyzer'
                supervisor_message = "Supervisor: Assigning Analyzer to research relevant topics"
            elif 'writer' in decision_text or not has_blog:
                next_agent = 'writer'
                supervisor_message = "Supervisor: Assigning Writer to create SEO-optimized blog"
            else:
                next_agent = 'end'
                supervisor_message = "Supervisor: Task appears to be complete"
            
            return {
                'messages': [AIMessage(content=supervisor_message)],
                'next_agent': next_agent,
                'current_task': task
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor decision: {e}")
            return {
                'messages': [AIMessage(content=f"Supervisor: Error occurred - {str(e)}")],
                'next_agent': 'end',
                'error_message': str(e)
            }

class TranscriptorAgent:
    """Extracts transcripts from YouTube videos."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def extract_transcript(self, state: SupervisorState) -> Dict:
        """Extract transcript from YouTube video."""
        url = state.get('input_url', '')
        
        if not validate_youtube_url(url):
            error_msg = f"Invalid YouTube URL: {url}"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Transcriptor: {error_msg}")],
                'next_agent': 'end'
            }
        
        try:
            logger.info(f"Extracting transcript from: {url}")
            
            loader = YoutubeLoader.from_youtube_url(
                url,
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=self.config.processing.chunk_size_seconds,
            )
            
            transcript_docs = loader.load()
            transcript_text = "\n\n".join([doc.page_content for doc in transcript_docs])
            
            # Clean the transcript
            transcript_text = clean_text(transcript_text)
            
            if not transcript_text.strip():
                error_msg = "No transcript found for this video"
                logger.warning(error_msg)
                return {
                    'error_message': error_msg,
                    'messages': [AIMessage(content=f"Transcriptor: {error_msg}")],
                    'next_agent': 'end'
                }
            
            logger.info(f"Successfully extracted transcript ({len(transcript_text)} characters)")
            
            return {
                'transcript_data': transcript_text,
                'messages': [AIMessage(content="Transcriptor: Successfully extracted video transcript")],
                'next_agent': 'supervisor',
            }
            
        except Exception as e:
            error_msg = f"Failed to extract transcript: {str(e)}"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Transcriptor: {error_msg}")],
                'next_agent': 'end'
            }

class AnalyzerAgent:
    """Searches the web for relevant topics and insights."""
    
    def __init__(self, llm: ChatGroq, config: AppConfig):
        self.llm = llm
        self.config = config
        self.search = TavilySearch()
    
    def analyze_content(self, state: SupervisorState) -> Dict:
        """Analyze transcript and search for relevant web content."""
        transcript = state.get('transcript_data', '')
        task = state.get("current_task", "")
        
        if not transcript:
            error_msg = "No transcript available for analysis"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Analyzer: {error_msg}")],
                'next_agent': 'end'
            }
        
        try:
            # Limit transcript length for keyword extraction
            if len(transcript) > self.config.processing.keyword_extraction_limit:
                transcript_for_keywords = transcript[:self.config.processing.keyword_extraction_limit]
            else:
                transcript_for_keywords = transcript
            
            # Extract keywords
            keyword_prompt = f"""
            Based on the transcript below, extract 3-5 relevant keywords or topics for web search.
            Return only the keywords in comma-separated format.
            
            Task: {task}
            Transcript: {transcript_for_keywords}
            """
            
            keyword_response = self.llm.invoke([HumanMessage(content=keyword_prompt)])
            keywords = keyword_response.content.strip()
            
            logger.info(f"Extracted keywords: {keywords}")
            
            # Perform web search
            search_results = self.search.invoke({
                "query": keywords,
                "max_results": self.config.processing.max_search_results
            })
            
            # Process search results
            results = search_results.get("results", [])
            if not results:
                logger.warning("No search results found")
                result_text = "No additional web content found for this topic."
            else:
                result_text = "\n\n".join([r["content"] for r in results])
            
            logger.info(f"Analysis complete with {len(results)} search results")
            
            return {
                "messages": [AIMessage(content="Analyzer: Web research and analysis completed")],
                "analyzed_data": result_text,
                "next_agent": "supervisor"
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Analyzer: {error_msg}")],
                'next_agent': 'end'
            }

class WriterAgent:
    """Creates SEO-optimized blog posts."""
    
    def __init__(self, llm: ChatGroq, config: AppConfig):
        self.llm = llm
        self.config = config
    
    def write_blog(self, state: SupervisorState) -> Dict:
        """Write SEO-optimized blog post."""
        transcript = state.get('transcript_data', '')
        analyzed_data = state.get('analyzed_data', '')
        task = state.get("current_task", "")
        
        if not transcript:
            error_msg = "No transcript available for blog writing"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Writer: {error_msg}")],
                'next_agent': 'end'
            }
        
        try:
            # Limit transcript length for blog generation
            if len(transcript) > self.config.processing.max_transcript_length:
                transcript = transcript[:self.config.processing.max_transcript_length]
            
            writer_prompt = f"""
You are a professional SEO blog writer. Create a high-quality, SEO-optimized blog post in Markdown format.

Guidelines:
1. Structure with compelling title, introduction, H2/H3 headings, and conclusion
2. Use relevant keywords naturally throughout
3. Include meta description (~150 characters) at the top
4. Add 3-5 SEO-friendly tags
5. Make it professional and human-like
6. Embed video URL with description at the end

Topic: {task}

Transcript: {transcript}

Web Insights: {analyzed_data}

Generate the complete blog post in Markdown:
"""
            
            blog_response = self.llm.invoke([HumanMessage(content=writer_prompt)])
            blog_content = blog_response.content
            
            if not blog_content.strip():
                error_msg = "Failed to generate blog content"
                logger.error(error_msg)
                return {
                    'error_message': error_msg,
                    'messages': [AIMessage(content=f"Writer: {error_msg}")],
                    'next_agent': 'end'
                }
            
            logger.info(f"Blog generated successfully ({len(blog_content)} characters)")
            
            return {
                'messages': [AIMessage(content="Writer: SEO-optimized blog post created successfully")],
                'next_agent': 'end',
                'final_blog': blog_content,
            }
            
        except Exception as e:
            error_msg = f"Blog writing failed: {str(e)}"
            logger.error(error_msg)
            return {
                'error_message': error_msg,
                'messages': [AIMessage(content=f"Writer: {error_msg}")],
                'next_agent': 'end'
            }

class BlogWorkflow:
    """Main workflow orchestrator."""
    
    def __init__(self):
        self.config = AppConfig()
        
        # Set environment variables for LangChain
        env_vars = self.config.get_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value
        
        self.llm = ChatGroq(model=self.config.model.groq_model)
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.llm)
        self.transcriptor = TranscriptorAgent(self.config)
        self.analyzer = AnalyzerAgent(self.llm, self.config)
        self.writer = WriterAgent(self.llm, self.config)
        
        # Build workflow graph
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the workflow graph."""
        workflow = StateGraph(SupervisorState)
        
        # Add nodes
        workflow.add_node('supervisor', self.supervisor.decide_next_agent)
        workflow.add_node('transcriptor', self.transcriptor.extract_transcript)
        workflow.add_node('analyzer', self.analyzer.analyze_content)
        workflow.add_node('writer', self.writer.write_blog)
        
        # Set entry point
        workflow.set_entry_point('supervisor')
        
        # Add conditional edges
        nodes = ["supervisor", "transcriptor", "analyzer", "writer"]
        for node in nodes:
            workflow.add_conditional_edges(
                node,
                self._router,
                {
                    "supervisor": "supervisor",
                    "transcriptor": "transcriptor",
                    "analyzer": "analyzer",
                    "writer": "writer",
                    END: END
                }
            )
        
        return workflow.compile()
    
    def _router(self, state: SupervisorState) -> Literal["supervisor", "transcriptor", "analyzer", "writer", "__end__"]:
        """Route to the next agent or end the workflow."""
        next_agent = state.get('next_agent', "supervisor")
        
        if next_agent == "end" or state.get("task_complete", False):
            return END
        
        return next_agent
    
    def process_video(self, url: str, task: str = "Create a blog from this video") -> Dict:
        """Process a YouTube video and generate a blog post."""
        logger.info(f"Starting blog generation for: {url}")
        
        try:
            response = self.graph.invoke({
                "input_url": url,
                "messages": [HumanMessage(content=task)],
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'error_message': str(e),
                'final_blog': '',
                'messages': [AIMessage(content=f"Workflow failed: {str(e)}")]
            }

class BlogExporter:
    """Handles exporting blog content to different formats."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def export_blog(self, blog_content: str, video_id: Optional[str] = None) -> Dict[str, str]:
        """Export blog content to Markdown and HTML formats."""
        if not blog_content.strip():
            raise ValueError("No blog content to export")
        
        # Ensure output directory exists
        ensure_directory_exists(self.config.output.output_directory)
        
        # Generate filename
        if video_id:
            base_filename = f"blog_{video_id}"
        else:
            base_filename = self.config.output.markdown_filename.replace('.md', '')
        
        base_filename = sanitize_filename(base_filename)
        
        output_files = {}
        
        try:
            # Export to Markdown
            md_path = Path(self.config.output.output_directory) / f"{base_filename}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(blog_content)
            output_files['markdown'] = str(md_path)
            logger.info(f"Markdown file saved: {md_path}")
            
            # Export to HTML
            html_content = markdown.markdown(blog_content)
            html_path = Path(self.config.output.output_directory) / f"{base_filename}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            output_files['html'] = str(html_path)
            logger.info(f"HTML file saved: {html_path}")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize workflow
        workflow = BlogWorkflow()
        
        # Example video URL
        video_url = "https://www.youtube.com/watch?v=37srVu0q5o0"
        task = "Let's create a blog from this video."
        
        # Extract video ID for better file naming
        video_id = get_video_id_from_url(video_url)
        
        # Process the video
        logger.info("Starting blog generation workflow...")
        response = workflow.process_video(video_url, task)
        
        # Check for errors
        if response.get('error_message'):
            logger.error(f"Workflow completed with error: {response['error_message']}")
            return
        
        # Export blog content
        blog_content = response.get('final_blog', '')
        if blog_content:
            exporter = BlogExporter(workflow.config)
            output_files = exporter.export_blog(blog_content, video_id)
            
            print("✅ Blog generation completed successfully!")
            print(f"📄 Markdown file: {output_files['markdown']}")
            print(f"🌐 HTML file: {output_files['html']}")
        else:
            logger.warning("No blog content generated")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()