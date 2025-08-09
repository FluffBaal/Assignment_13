import asyncio
import os
from typing import TypedDict, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import sys

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    task: str
    result: str

class NASAExplorerAgent:
    def __init__(self):
        self.client = None
        self.tools = None
        self.agent = None
        
    async def initialize(self):
        """Initialize the MCP client and tools"""
        print("🚀 Initializing NASA Explorer Agent...")
        
        # Configure the MCP client to connect to your server
        self.client = MultiServerMCPClient({
            "nasa_server": {
                "command": sys.executable,
                "args": ["server.py"],
                "transport": "stdio"
            }
        })
        
        # Get tools from the MCP server
        self.tools = await self.client.get_tools()
        print(f"✅ Loaded {len(self.tools)} tools from MCP server")
        
        # Create a LangChain agent with the MCP tools
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.agent = create_react_agent(llm, self.tools)
        print("✅ Agent initialized successfully!")
        
    async def explore_nasa(self, query: str):
        """Process a query about NASA data"""
        try:
            response = await self.agent.ainvoke({
                "messages": [HumanMessage(content=query)]
            })
            return response["messages"][-1].content
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)

class SpaceNewsWorkflow:
    """A LangGraph workflow that combines NASA data with news"""
    
    def __init__(self):
        self.client = None
        self.tools = None
        self.llm = None
        
    async def initialize(self):
        """Initialize the workflow"""
        print("🌌 Initializing Space News Workflow...")
        
        self.client = MultiServerMCPClient({
            "space_server": {
                "command": sys.executable,
                "args": ["server.py"],
                "transport": "stdio"
            }
        })
        
        self.tools = await self.client.get_tools()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        print("✅ Workflow initialized!")
        
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("get_astronomy_picture", self.get_astronomy_picture)
        workflow.add_node("search_space_news", self.search_space_news)
        workflow.add_node("analyze_results", self.analyze_results)
        
        # Define edges
        workflow.set_entry_point("get_astronomy_picture")
        workflow.add_edge("get_astronomy_picture", "search_space_news")
        workflow.add_edge("search_space_news", "analyze_results")
        workflow.add_edge("analyze_results", END)
        
        return workflow.compile()
    
    async def get_astronomy_picture(self, state: AgentState):
        """Get today's astronomy picture"""
        agent = create_react_agent(self.llm, self.tools)
        response = await agent.ainvoke({
            "messages": [HumanMessage(content="Get today's astronomy picture of the day")]
        })
        
        state["messages"] = response["messages"]
        return state
    
    async def search_space_news(self, state: AgentState):
        """Search for related space news"""
        agent = create_react_agent(self.llm, self.tools)
        response = await agent.ainvoke({
            "messages": [HumanMessage(content="Search for recent space and astronomy news")]
        })
        
        state["messages"].extend(response["messages"])
        return state
    
    async def analyze_results(self, state: AgentState):
        """Analyze and summarize the results"""
        summary_prompt = """Based on the astronomy picture and news gathered, 
        provide a brief summary of today's space highlights."""
        
        response = await self.llm.ainvoke([
            *state["messages"],
            HumanMessage(content=summary_prompt)
        ])
        
        state["result"] = response.content
        return state
    
    async def run(self, task: str = "Explore today's space content"):
        """Run the workflow"""
        initial_state = {
            "messages": [],
            "task": task,
            "result": ""
        }
        
        result = await self.workflow.ainvoke(initial_state)
        return result["result"]
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)

async def main():
    """Main function to demonstrate the LangGraph MCP integration"""
    
    print("=" * 60)
    print("🚀 NASA MCP + LangGraph Demo")
    print("=" * 60)
    
    # Demo 1: Simple NASA Explorer Agent
    print("\n📡 Demo 1: NASA Explorer Agent")
    print("-" * 40)
    
    explorer = NASAExplorerAgent()
    await explorer.initialize()
    
    # Example queries
    queries = [
        "What's today's astronomy picture about?",
        "Show me recent Mars rover photos",
        "Search for news about space exploration"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        result = await explorer.explore_nasa(query)
        print(f"📊 Result: {result[:500]}...")  # Truncate for display
    
    await explorer.cleanup()
    
    # Demo 2: Space News Workflow
    print("\n\n🌌 Demo 2: Space News Workflow")
    print("-" * 40)
    
    workflow = SpaceNewsWorkflow()
    await workflow.initialize()
    
    print("\n🔄 Running workflow...")
    result = await workflow.run()
    print(f"\n📰 Workflow Result:\n{result}")
    
    await workflow.cleanup()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        exit(1)
    
    # Run the async main function
    asyncio.run(main())