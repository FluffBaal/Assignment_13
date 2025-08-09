#!/usr/bin/env python3
"""
Simple LangGraph + MCP Integration Demo
This script demonstrates how to use your MCP server with LangGraph
"""

import asyncio
import os
import sys
from typing import TypedDict, Annotated, Sequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

# Define the state for our graph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    nasa_data: str
    news_data: str
    final_report: str

async def create_mcp_agent():
    """Create an agent with MCP tools"""
    # Initialize MCP client
    client = MultiServerMCPClient({
        "my_server": {
            "command": sys.executable,
            "args": ["server.py"],
            "transport": "stdio"
        }
    })
    
    # Get tools from the MCP server
    tools = await client.get_tools()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools, tools, client

async def get_nasa_data(state: State):
    """Node to fetch NASA data"""
    print("🔭 Fetching NASA data...")
    
    llm_with_tools, tools, client = await create_mcp_agent()
    
    # Use the astronomy picture tool
    messages = state["messages"]
    messages.append(HumanMessage(content="Get today's astronomy picture of the day"))
    
    response = await llm_with_tools.ainvoke(messages)
    
    # Execute the tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    result = await tool.ainvoke(tool_args)
                    state["nasa_data"] = str(result)
                    break
    
    # Client cleanup not needed with new API
    return state

async def get_news_data(state: State):
    """Node to fetch space news"""
    print("📰 Fetching space news...")
    
    llm_with_tools, tools, client = await create_mcp_agent()
    
    messages = [HumanMessage(content="Search for recent news about space and astronomy")]
    response = await llm_with_tools.ainvoke(messages)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            for tool in tools:
                if tool.name == tool_name:
                    result = await tool.ainvoke(tool_args)
                    state["news_data"] = str(result)
                    break
    
    # Client cleanup not needed with new API
    return state

async def create_report(state: State):
    """Node to create final report"""
    print("📝 Creating final report...")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""
    Based on the following data, create a brief space update report:
    
    NASA Data: {state.get('nasa_data', 'No NASA data available')}
    
    News Data: {state.get('news_data', 'No news data available')}
    
    Create a concise, engaging summary of today's space highlights.
    """
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    state["final_report"] = response.content
    
    return state

def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("get_nasa", get_nasa_data)
    workflow.add_node("get_news", get_news_data)
    workflow.add_node("create_report", create_report)
    
    # Set entry point
    workflow.set_entry_point("get_nasa")
    
    # Add edges
    workflow.add_edge("get_nasa", "get_news")
    workflow.add_edge("get_news", "create_report")
    workflow.add_edge("create_report", END)
    
    return workflow.compile()

async def test_simple_tool_call():
    """Test a simple tool call directly"""
    print("\n🧪 Testing direct MCP tool call...")
    
    client = MultiServerMCPClient({
        "test_server": {
            "command": sys.executable,
            "args": ["server.py"],
            "transport": "stdio"
        }
    })
    
    tools = await client.get_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Test the astronomy picture tool
    for tool in tools:
        if tool.name == "get_astronomy_picture":
            result = await tool.ainvoke({})
            print(f"\n🌌 Astronomy Picture Result:\n{result}")
            break
    
    # No need to call __aexit__ directly

async def main():
    """Main function"""
    print("=" * 60)
    print("🚀 LangGraph + MCP Integration Demo")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("\nTo run this demo, you need to:")
        print("1. Add OPENAI_API_KEY to your .env file")
        print("2. Or run the simple tool test without OpenAI:")
        print("   python simple_langgraph_mcp.py --test-only")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
            await test_simple_tool_call()
        return
    
    # Test simple tool call first
    await test_simple_tool_call()
    
    # Run the full workflow
    print("\n" + "=" * 60)
    print("🔄 Running LangGraph Workflow")
    print("=" * 60)
    
    workflow = create_workflow()
    
    initial_state = {
        "messages": [HumanMessage(content="Get space updates")],
        "nasa_data": "",
        "news_data": "",
        "final_report": ""
    }
    
    try:
        result = await workflow.ainvoke(initial_state)
        
        print("\n" + "=" * 60)
        print("📊 Final Report:")
        print("=" * 60)
        print(result["final_report"])
        
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        print("\nTip: Make sure your MCP server is working correctly")
        print("You can test it with: python server.py")

if __name__ == "__main__":
    asyncio.run(main())