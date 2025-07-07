from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langgraph.graph import StateGraph,END,MessagesState
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.prompts import ChatPromptTemplate
from typing import Dict,Literal
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')

llm=ChatGroq(model='llama-3.1-8b-instant')

## Supervisor State
class SupervisorState(MessagesState):
    next_agent:str=''
    transcript_data:str=''
    analyzed_data:str=''
    final_blog:str=''
    task_complete:bool=False
    current_task:str=''
    input_url:str=''

def create_supervisor_chain():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ('system', """
You are an AI supervisor managing a multi-agent system. Your job is to assign the next best agent based on the current progress.

Agents:
1. transcriptor — creates transcript from YouTube video.
2. Analyzer - Search the web for relevant topics to write about.
2. writer — writes blog from transcript and make it SEO optimized as well.

Current State:
- Transcript available: {has_transcript}
- Analyzed: {has_analyzed}
- Blog written: {has_blog}

TASK: {task}

RESPOND ONLY with one of the following (lowercase):
- transcriptor- seo
- analyzer
- writer
- done
""")
,
        ("human","{task}")
    ])

    return supervisor_prompt | llm

def supervisor_agent(state:SupervisorState)->Dict:
    messages = state['messages']
    task=messages[-1].content if messages else 'No Task'

    has_transcript = bool(state.get('transcript_data',''))
    has_analyzed = bool(state.get('analyzed_data',''))
    has_blog = bool(state.get('final_blog',''))

    chain = create_supervisor_chain()

    decision = chain.invoke({
        'task':task,
        'has_transcript':has_transcript,
        'has_analyzed':has_analyzed,
        'has_blog':has_blog
    })

    decision_text=decision.content.strip().lower()


    if "done" in decision_text or has_blog:
        next_agent='end'
        supervisor_message="Supervisor: All tasks done. Great Work Team"
    elif "transcriptor" in decision_text or not has_transcript:
        next_agent='transcriptor'
        supervisor_message="Supervisor: Assign Transcriptor to get the transcript of the video"
    elif "analyzer" in decision_text or not has_analyzed:
        next_agent='analyzer'
        supervisor_message="Supervisor: Assign Analyzer to searc relevant things about blog"
    elif 'writer' in decision_text or not has_blog:
        next_agent='writer'
        supervisor_message="Supervisor: Assign SEO to write and optimize the blog"
    else:
        next_agent='end'
        supervisor_message = "✅ Supervisor: Task seems complete."
    
    return {
        'messages':[AIMessage(content=supervisor_message)],
        'next_agent':next_agent,
        'current_task':task
    }

# transcriptor
def transcriptor_video(state:SupervisorState) -> Dict:
    url=state['input_url']
    
    loader = YoutubeLoader.from_youtube_url(
        url,
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=60,
    )

    agent_message = f"Transcriptor: I have completed transcription for the task"
    transcript=loader.load()
    transcript_text = "\n\n".join([doc.page_content for doc in transcript])
    return {
        'transcript_data':transcript_text,
        'messages':[AIMessage(content=agent_message)],
        'next_agent':'supervisor',
    }

# analyzer
def analyzer(state: SupervisorState) -> Dict:
    search =TavilySearch()
    transcript = state.get('transcript_data')
    if len(transcript) > 500:
        transcript = transcript[:500]  # Limit for summarization

    task = state.get("current_task", "")

    keyword_prompt = f"""
    You are a helpful AI. Based on the transcript below, extract 3-5 keywords or topics to search on the web for blog enrichment. Just return the keywords in comma-separated format. Make sure all the keywords returned are relevant to the blog topic.
    Task:
    {task}
    Transcript:
    {transcript}
    """
    keyword_response = llm.invoke([HumanMessage(content=keyword_prompt)])
    keywords = keyword_response.content.strip()

    print("🔍 Keywords for search:", keywords)

    # Step 2: Perform Tavily web search
    search_results = search.invoke({
        "query": keywords,
        "max_results": 2
    })

    # Step 3: Summarize search results
    result_text = "\n\n".join([r["content"] for r in search_results.get("results", [])])

    return {
        "messages": [AIMessage(content="Analyzer: Search and analysis complete.")],
        "analyzed_data": result_text,
        "next_agent": "supervisor"
    }

# writer
def write_blog(state:SupervisorState)->Dict:
    transcript = state.get('transcript_data')
    analyzed_data = state.get('analyzed_data')
    if(len(transcript) > 10000):
        transcript=transcript[:10000]
    task = state.get("current_task", "")

    writer_prompt = f"""
You are a professional SEO blog writer.

Your task is to create a **high-quality, SEO-optimized blog post** in **Markdown** format using both:
- A transcript from a YouTube video (raw content)
- Analyzed web search results with key facts, statistics, and external insights

### 📋 Guidelines:
1. **Structure** the blog clearly with:
   - A compelling **title**
   - An engaging **introduction**
   - **H2/H3 headings**, **bullet points**, and **conclusion**
2. Use **important keywords** naturally throughout the blog for SEO (based on the task and search results).
3. Add a short but rich **meta description** (~150 characters) at the top.
4. Include **3–5 SEO-friendly tags**.
5. DO NOT mention that this was generated from a transcript or web results.
6. Do not mention SEO here and there. Make it professional, enrich with data. It should feel like that it is written by a human.
7. At the end embed video url with a short description.
---

### 🧠 Topic:
{task}

---

### 🎙 Transcript:
{transcript}

---

### 🌐 Analyzed Web Insights:
{analyzed_data}

---

Now generate the full blog post in Markdown below:
"""

    blog_writer_agent_response = llm.invoke([HumanMessage(content=writer_prompt)])

    blog = blog_writer_agent_response.content
    agent_message = f"Writer: Blog is written. Here are the top insights\n{blog[:400]}..."

    return {
        'messages':[AIMessage(content=agent_message)],
        'next_agent':'end',
        'final_blog':blog,
    }

def router(state:SupervisorState)->Literal["supervisor", "transcriptor", "writer", "seo", "__end__"]:
    next_agent=state.get('next_agent',"supervisor")
    if next_agent == "end" or state.get("task_complete", False):
        return END
    return next_agent

workflow=StateGraph(SupervisorState)
workflow.add_node('supervisor',supervisor_agent)
workflow.add_node('transcriptor',transcriptor_video)
workflow.add_node('analyzer',analyzer)
workflow.add_node('writer',write_blog)

workflow.set_entry_point('supervisor')

nodes=["supervisor", "transcriptor", 'analyzer' ,"writer"]

for node in nodes:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "transcriptor": "transcriptor",
            'analyzer':'analyzer',
            "writer": "writer",
            END: END
        }
    )
graph=workflow.compile()

if __name__=='__main__':
    response = graph.invoke({
        "input_url": "https://www.youtube.com/watch?v=37srVu0q5o0",
        "messages": [HumanMessage(content="Let's create a blog from this video.")],
    })

    blog_content=response['final_blog']
    with open("generated_blog.md", "w", encoding="utf-8") as f:
        f.write(blog_content)
        f.close()
        print('Markdown File Saved Successfully')

    ### get in html as well

    import markdown
    html = markdown.markdown(blog_content)
    with open("generated_blog.html", "w", encoding="utf-8") as f:
        f.write(html)
        f.close()
        print('HTML File Saved Successfully')