#!/usr/bin/env python3
"""
MCP Client - Answer any question using all available MCP tools
Usage: uv run python mcp_client.py "Your question here"
"""

import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os
import re
from datetime import datetime
from pathlib import Path

load_dotenv()

def extract_images_from_text(text: str):
    """Extract image URLs from text"""
    images = []
    
    # Pattern for NASA image URLs
    nasa_patterns = [
        r'🖼️ Image URL: (https?://[^\s]+)',
        r'🔭 HD Image: (https?://[^\s]+)',
        r'🔗 Image: (https?://[^\s]+\.(?:jpg|jpeg|png|gif))',
        r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif))'
    ]
    
    for pattern in nasa_patterns:
        matches = re.findall(pattern, text)
        images.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    
    return unique_images

def save_to_markdown(question: str, answer: str, tool_outputs: list):
    """Save the question, answer, and any images to a markdown file"""
    
    # Create outputs directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_question = re.sub(r'[^\w\s-]', '', question)[:50].strip().replace(' ', '_')
    filename = f"{timestamp}_{safe_question}.md"
    filepath = output_dir / filename
    
    # Build markdown content
    md_content = f"# MCP Query Result\n\n"
    md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"## Question\n\n{question}\n\n"
    md_content += f"## Answer\n\n{answer}\n\n"
    
    # Extract all images from tool outputs
    all_images = []
    for output in tool_outputs:
        images = extract_images_from_text(output)
        all_images.extend(images)
    
    # Add images section if any were found
    if all_images:
        md_content += "## Images\n\n"
        for i, img_url in enumerate(all_images, 1):
            # Determine image caption based on URL
            caption = "Image"
            if "apod" in img_url or "astronomy" in img_url:
                caption = "Astronomy Picture"
            elif "mars" in img_url:
                caption = "Mars Rover Photo"
            
            md_content += f"### {caption} {i}\n\n"
            md_content += f"![{caption} {i}]({img_url})\n\n"
            md_content += f"[Direct Link]({img_url})\n\n"
    
    # Add raw tool outputs in details section
    if tool_outputs:
        md_content += "## Tool Outputs\n\n"
        md_content += "<details>\n<summary>Click to expand raw tool outputs</summary>\n\n"
        for i, output in enumerate(tool_outputs, 1):
            md_content += f"### Tool Output {i}\n\n"
            md_content += "```\n"
            md_content += output[:2000]  # Limit length for readability
            if len(output) > 2000:
                md_content += "\n... (truncated)"
            md_content += "\n```\n\n"
        md_content += "</details>\n"
    
    # Write to file
    filepath.write_text(md_content)
    
    return filepath

async def answer_question(question: str):
    """Answer a question using all available MCP tools"""
    
    # Initialize MCP client
    client = MultiServerMCPClient({
        "mcp_server": {
            "command": sys.executable,
            "args": ["server.py"],
            "transport": "stdio"
        }
    })
    
    # Get all available tools
    tools = await client.get_tools()
    print(f"🔧 Available tools: {[tool.name for tool in tools]}\n")
    
    # Create OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create agent with all MCP tools
    agent = create_react_agent(llm, tools)
    
    print("🤔 Thinking...\n")
    
    # Get answer
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=question)]
    })
    
    # Extract tool outputs from messages
    tool_outputs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage):
            tool_outputs.append(msg.content)
    
    # Get final answer
    final_answer = response["messages"][-1].content
    
    # Save to markdown
    filepath = save_to_markdown(question, final_answer, tool_outputs)
    
    return final_answer, filepath

async def main():
    """Main function"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please add your OpenAI API key to the .env file")
        sys.exit(1)
    
    # Get question from command line
    if len(sys.argv) < 2:
        print("📋 Usage: uv run python mcp_client.py \"Your question here\"")
        print("\nExamples:")
        print('  uv run python mcp_client.py "What is today\'s astronomy picture about?"')
        print('  uv run python mcp_client.py "Show me the latest Mars rover photos"')
        print('  uv run python mcp_client.py "What are the top technology news headlines?"')
        print('  uv run python mcp_client.py "Search for news about SpaceX"')
        print('  uv run python mcp_client.py "Roll 3d6 for me"')
        sys.exit(1)
    
    # Get the question (all arguments after script name)
    question = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("🚀 MCP Client")
    print("=" * 60)
    print(f"\n📝 Question: {question}\n")
    print("-" * 60)
    
    try:
        answer, filepath = await answer_question(question)
        print("\n💡 Answer:\n")
        print(answer)
        print("\n" + "=" * 60)
        print(f"\n📄 Output saved to: {filepath}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your MCP server is working and all API keys are set.")

if __name__ == "__main__":
    asyncio.run(main())