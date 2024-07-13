#Reference
#Sources
#   Youtube
#       1) LangGraph: Agent Exectuor (https://www.youtube.com/watch?v=9dXp5q3OFdQ&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&index=2)


#Set environment variables
import os
from dotenv import load_dotenv
_=load_dotenv()
os.environ["OPENAI_API_KEY"]=os.environ["OPENAI_API_KEY"]#os.environ["LMSTUDIO_API_KEY"]
os.environ["TAVILY_API_KEY"]=os.environ["TAVILY_API_KEY"]



#create langchain agent
from langchain import hub ;
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults



tools=[TavilySearchResults(max_results=1)]

#Get the prompt to use- you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

#choose the llm that will drive the agent
llm=ChatOpenAI(model=os.environ["OPENAI_MODEL_NAME"],
            #    base_url=os.environ["LMSTUDIO_BASE_URL"],
               streaming=True)
#Construct The OpenAI Function agent
agent_runnable=create_openai_functions_agent(llm,tools,prompt)

from typing import TypedDict,Annotated,List,Union
from langchain_core.agents import AgentAction,AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    #The input string
    input:str
    #The list of previous messages in the conversation
    chat_history:list[BaseMessage]
    #The outcome of a given call to the agent
    #Needs "None" as a valid type, since this is what 
    agent_outcome:Union[AgentAction,AgentFinish,None]
    #List of actions and corresponding observations
    # Here we annotate this with 'operations.add" to indicate that operations to 
    # this state should be ADDED To the existing values ( not overwrite it )
    intermediate_steps:Annotated[list[tuple[AgentAction,str]],operator.add]

from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor

# this is a helper class we have have that is useful for running tools
# it takes in an agent action and calls that tool and returns the result.
tool_executer=ToolExecutor(tools)

#Define the agent
def run_agent(data):
    agent_outcome=agent_runnable.invoke(data)
    return {"agent_outcome":agent_outcome}
#Define the function to execute tools
def execute_tools(data):
    #Get the most recent agent_outcome - this is the key added in the 'agent' above.
    agent_action=data['agent_outcome']
    output=tool_executer.invoke(agent_action)
    return {"intermediate_steps":[(agent_action,str(output))]}

#Define the logic that will be used to determine which onditional edge to go down.
def should_continue(data):
    #If the agent outcome is an AgentFinish, then we return 'exit' string
    # this will be used when setting up the graph to define the flow.
    if isinstance(data['agent_outcome'],AgentFinish):
        return "end"
    #Otherwise, an AgentAction is returned
    #Here we return 'continue" string
    # this will be used when setting up the graph to define the flow
    else:
         return "continue"
     
#Define the graph.
from langgraph.graph import END,StateGraph

#Define a new graph.
workflow=StateGraph(AgentState)

#define the two nodes we will cycle b/w 
workflow.add_node("agent",run_agent)
workflow.add_node("action",execute_tools)

#set the entrypoint as 'agent'
# this means that this node is the first one called
workflow.set_entry_point("agent")


#we now add a conditional edge
workflow.add_conditional_edges(
    #First, we define the start node. we use 'agent'
    # this means these are the edges taken after the 'agent' node is called.
    "agent",
    #next, we pass in the function that will determine which node is called next.
    should_continue,
    #Finally we pass in a mapping.
    #The keys are strings, and the the graph should finish.
    # what will happen is we will call 'should_continue" and then the output of that
    #END is the a special node marking that the graph should finish.
    # what will happen is we will call 'should_continue" and then the output of that
    # will be matched against the keys in this mapping.
    #Based on which one it matches, the node will then be called.
    {
        #If 'tools' then we call the tool node.
        "continue":"action",
        #otherwise we finish.
        "end":END
    }
)
#we now add a normal edge from 'tools' to 'agent',
#this means that after 'tools' is called, 'agent' node is called next
workflow.add_edge('action','agent')

#Finally, we compile it!
# this complies it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

app=workflow.compile()

inputs={"input":"What is the weather in Hyderabad india","chat_history":[]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")