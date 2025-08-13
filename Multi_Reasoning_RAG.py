from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Annotated,List,Literal,TypedDict,Sequence
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END,START
from dotenv import load_dotenv,find_dotenv
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import os
import pandas as pd



load_dotenv(find_dotenv())

GROQ_API_KEY=os.environ['GROQAPI_KEY']




embedding_function=HuggingFaceEmbeddings()

llm=ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)

df = pd.read_excel(r"D:\LangGraph\07_RAG\gym.xlsx")

str_df=df.to_string(index=False)
csv_doc=Document(
    page_content=str_df,
    metadata={
        "source":"gym.xlsx"
    }
)

docs=[]
pdf_docs=[]
pdf_files = [
    r"D:\LangGraph\07_RAG\one.pdf",
    r"D:\LangGraph\07_RAG\two.pdf",
    r"D:\LangGraph\07_RAG\three.pdf",
    r"D:\LangGraph\07_RAG\04.pdf",
    r"D:\LangGraph\07_RAG\five.pdf"
]

for doc in pdf_files:
    loader=PyPDFLoader(doc)
    pdf_docs.extend(loader.load())

docs = [csv_doc] + pdf_docs

db=Chroma.from_documents(docs,embedding_function)

retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":5})

template="""
 Answer the question based on the following context and the chat history
 .Especially take the lastest question into consideration
 chat history:{history}
 context:{context}
 Question:{question}
"""

prompt=ChatPromptTemplate.from_template(
    template
)

rag_chain=prompt |llm



class AgentState(TypedDict):
    messages:List[BaseMessage]
    documents:List[Document]
    ontopic:str
    rephrased_question:str
    proceed_to_answer:bool
    rephrase_count:int
    question:HumanMessage



class GradeQuestion(BaseModel):
    score:str=Field(
        description="Question is about the specified topic? if yes-> 'Yes' if not->'No'"

    )    


def question_rewriter(state:AgentState):
    state["documents"]=[]
    state["ontopic"]=""
    state["proceed_to_answer"]=False
    state['rephrase_count']=0
    state["rephrased_question"]=""

    if "messages" not in state or state['messages'] is None:
        state['messages']=[]
    if state['question'] not in state['messages']:
        state['messages'].append(state['question'])    

    if len(state['messages'])>1:
        conversation=state['messages'][:-1]
        current_question=state['question'].content
        messages=[
            SystemMessage(
                content="You are helpful assistant that rephrases the user's question to be a suitable question optimized for the retrieval "
            )
        ]
        messages.extend(conversation) 
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt=ChatPromptTemplate.from_messages(messages)
        prompt=rephrase_prompt.format()
        reponse=llm.invoke(prompt)

        better_response=reponse.content.strip()
        state["rephrased_question"]=better_response
    else:
        state['rephrased_question']=state["question"].content

    return state


def question_classifier(state:AgentState):
    system_message=SystemMessage(
        content="""you are a classifier that determines whether the user's question is about one of the follwowing topics:
           "Information related to gym exercises, diets,Health,Fitness plans"
           if the question is from one of the following topics respond with "Yes" otherwise respond with a "No"
           Example response format:
           {"score": "Yes"}
           or
           {"score": "No"}
         """
    )        
    human_message=HumanMessage(
        content=f"User's question : {state['rephrased_question']}"
    )

    grade_prompt=ChatPromptTemplate.from_messages([system_message,human_message])
    structured_llm=llm.with_structured_output(GradeQuestion)
    grade_llm=grade_prompt | structured_llm

    result=grade_llm.invoke({"question": state['rephrased_question']})
    state['ontopic']=result.score.strip()
    print(f"question_classifier: ontopic = {state['ontopic']}")

    return state


def on_topic_router(state:AgentState):
    ontopic=state.get("ontopic","").strip().lower()
    if ontopic == 'yes' :
        return "retrieve"
    else:
        return "off_topic_response"
    

def retrieve(state:AgentState):
    documents=retriever.invoke(state["rephrased_question"])
    state['documents']=documents
    # for i, doc in enumerate(documents):
    #     print(f"Doc {i+1}: {doc.page_content[:100]}...")
    return state


class GradeDocuments(BaseModel):
    score:str=Field(
        description="Document is relevant to the question? if yes -> 'Yes if not -> 'No'"
    )    


def retrieval_grader(state:AgentState):
    system_message=SystemMessage(
        content="""you are a grader assessing the relevance of the a retrieved documnet to the user question

        You must respond with a JSON object containing a "score" field.
        The score should be "Yes" if the document contains information relevant to the user's question,
        or "No" if it does not contain relevant information.
        
        {"score": "Yes"}
           or
        {"score": "No"}
        """

    )
    structured_llm=llm.with_structured_output(GradeDocuments)
    relevant_docs=[]
    for doc in state['documents']:
        human_message=HumanMessage(
            content=f"User's question:{state['rephrased_question']}\nRetived Document:\n{doc.page_content}"
        )
        grade_prompt=ChatPromptTemplate.from_messages([system_message,human_message])
        grade_llm=grade_prompt|structured_llm
        result=grade_llm.invoke({"question": state['rephrased_question'], "document": doc.page_content[:1000]})
        if result.score.strip().lower()=='yes':
            relevant_docs.append(doc)

    state['documents']=relevant_docs
    state['proceed_to_answer'] = len(relevant_docs) > 0

    return state


def refine_question(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        return state

    question_to_refine = state['rephrased_question']
    system_message = SystemMessage(
        content="You are a helpful assistant that slightly refines the user's question to improve the retrieval results."
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n Provide a slightly refined question."
    )

    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()

    state['rephrased_question'] = refined_question
    state['rephrase_count'] = rephrase_count + 1  
    return state



def proceed_router(state:AgentState):
    rephrase_count=state.get('rephrase_count',0)
    if state.get("proceed_to_answer",False):
        return "generate_answer"
    elif rephrase_count>=2:
        return "cannot_answer"
    else:
        return "refine_question"


def generate_answer(state:AgentState):
    if "messages" not in state or state['messages'] is None:
        raise ValueError("State must contain some values")
    
    history=state['messages']
    documents=state["documents"]
    rephrased_question=state['rephrased_question']
    context_text = "\n\n".join(doc.page_content for doc in documents)
    response=rag_chain.invoke(
        {
            "history":history,"context":context_text,"question":rephrased_question
        }
    )

    generation=response.content.strip()

    state['messages'].append(AIMessage(content=generation))

    return state

def cannot_answer(state:AgentState):
    if "messages" not in state or state['messages'] is None:
        state['messages']=[]

    state['messages'].append(
        AIMessage(
            content="I am sorry but i cannot find any information related to your question"
        )
    )    

    return state



def off_topic_response(state:AgentState):
    if "messages" not in state or state['messages'] is None:
        state['messages']=[]
    state['messages'].append(AIMessage(content="Sorry, i cannot answer this question"))
    return state


checkpointer=MemorySaver()

graph=StateGraph(AgentState)

graph.add_node("question_rewriter",question_rewriter)
graph.add_node("question_classifier",question_classifier)
graph.add_node("off_topic_response",off_topic_response)
graph.add_node("retrieve",retrieve)
graph.add_node("retrieval_grader",retrieval_grader)
graph.add_node("generate_answer",generate_answer)
graph.add_node("refine_question",refine_question)
graph.add_node("cannot_answer",cannot_answer)

graph.add_edge("question_rewriter","question_classifier")
graph.add_edge("retrieve","retrieval_grader")

graph.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve":"retrieve",
        "off_topic_response":"off_topic_response"
    }
)
graph.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "generate_answer":"generate_answer",
        "cannot_answer":"cannot_answer",
        "refine_question":"refine_question"
    }
)

graph.add_edge("refine_question","retrieve")
graph.add_edge("generate_answer",END)
graph.add_edge("cannot_answer",END)
graph.add_edge("off_topic_response",END)
graph.set_entry_point("question_rewriter")

app=graph.compile(checkpointer=checkpointer)

reponse=app.invoke({
"question":HumanMessage(
        content="Benefits of Running"
    )
},config={"configurable":{"thread_id":2}})

print("ON-TOPIC RESPONSE")
print(reponse['messages'][-1].content)

response=app.invoke({
"question":HumanMessage(
        content="Benefits of Keeping Clothes in Summer?"
    )
},config={"configurable":{"thread_id":2}})

print("Now OFF-TOPIC QUESTION")
print(response['messages'][-1].content)
