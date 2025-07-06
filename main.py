from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langgraph.graph import StateGraph,END,MessagesState
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import List,Annotated,Dict,Any,Literal
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

llm=ChatGroq(model='llama-3.1-8b-instant')

## Supervisor State
class SupervisorState(MessagesState):
    next_agent:str=''
    transcript_data:str=''
    written_blog:str=''
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
2. writer — writes blog from transcript.
3. seo — optimizes the blog for SEO.

Current State:
- Transcript available: {has_transcript}
- Blog written: {has_blog}
- Blog optimized: {has_optimized_blog}

TASK: {task}

RESPOND ONLY with one of the following (lowercase):
- transcriptor
- writer
- seo
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
    has_blog = bool(state.get('written_blog',''))
    has_optimized_blog = bool(state.get('final_blog',''))

    chain = create_supervisor_chain()

    decision = chain.invoke({
        'task':task,
        'has_transcript':has_transcript,
        'has_blog':has_blog,
        'has_optimized_blog':has_optimized_blog
    })

    decision_text=decision.content.strip().lower()


    if "done" in decision_text or has_optimized_blog:
        next_agent='end'
        supervisor_message="Supervisor: All tasks done. Great Work Team"
    elif "transcriptor" in decision_text or not has_transcript:
        next_agent='transcriptor'
        supervisor_message="Supervisor: Assign Transcriptor to get the transcript of the video"
    elif "writer" in decision_text or not has_blog:
        next_agent='writer'
        supervisor_message="Supervisor: Assign Writer to write the blog"
    elif 'seo' in decision_text or not has_optimized_blog:
        next_agent='seo'
        supervisor_message="Supervisor: Assign SEO to optimize the blog"
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

# writer
def write_blog(state:SupervisorState)->Dict:
    transcript = state.get('transcript_data')
    if(len(transcript) > 10000):
        transcript=transcript[:10000]
    task = state.get("current_task", "")


    writer_prompt = f"""
You are a professional blog writer.

Analyze the transcript provided below and generate a high-quality blog post from it. Ensure:
- The blog is well-structured, engaging, and easy to read.
- Obvious mistakes (like grammar/spelling errors or unclear sentences) are corrected.
- The final output is in **Markdown** format with appropriate headings, subheadings, and bullet points where needed.
- Focus on extracting **actionable insights** relevant to the following topic: **{task}**

Here is the transcript:
---
{transcript}
---
"""
    blog_writer_agent_response = llm.invoke([HumanMessage(content=writer_prompt)])

    blog = blog_writer_agent_response.content
    print(blog)

    agent_message = f"Writer: Blog is written. Here are the top insights\n{blog[:400]}..."

    print(blog)

    return {
        'messages':[AIMessage(content=agent_message)],
        'next_agent':'supervisor',
        'written_blog':blog
    }

# SEO and Final Blog Agent
def final_blog_generator(state:SupervisorState)->Dict:
    blog = state.get('written_blog')
    task = state.get("current_task", "")

    final_blog_prompt = f'''
    As a professional SEO expert, I will now analyze the blog and make sure it is optimized for search engines. I will make sure that the blog is well-structured and has the right keywords. Dont gave tips in blog how to optimize it just optimize and return. I will also make sure that the blog is easy to read and has the right meta description. Providing task and blog below make sure I should not hamper the Markdown format

    Task: {task}
    Blog: {blog}
'''
    final_blog_agent_response = llm.invoke([HumanMessage(content=final_blog_prompt)])
    final_blog = final_blog_agent_response.content
    agent_message = f"SEO Expert: Final Blog is ready. Here are the top insights\n"

    return {
        "messages": [AIMessage(content=agent_message)],
        "final_blog": final_blog,
        "next_agent": "supervisor",
        "task_complete": True
    }

def router(state:SupervisorState)->Literal["supervisor", "transcriptor", "writer", "seo", "__end__"]:
    next_agent=state.get('next_agent',"supervisor")
    if next_agent == "end" or state.get("task_complete", False):
        return END
    return next_agent

workflow=StateGraph(SupervisorState)
workflow.add_node('supervisor',supervisor_agent)
workflow.add_node('transcriptor',transcriptor_video)
workflow.add_node('writer',write_blog)
workflow.add_node('seo',final_blog_generator)

workflow.set_entry_point('supervisor')

nodes=["supervisor", "transcriptor", "writer", "seo"]

for node in nodes:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "transcriptor": "transcriptor",
            "writer": "writer",
            "seo": "seo",
            END: END
        }
    )
graph=workflow.compile()

response = graph.invoke({
    "input_url": "https://www.youtube.com/watch?v=74SnvbQYgx8&t=17s",
    "messages": [HumanMessage(content="Let's create a blog from this video.")],
})

print(response['final_blog'])