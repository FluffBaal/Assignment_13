from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller
import requests
from datetime import datetime, timedelta
from typing import Optional

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # NASA provides DEMO_KEY for testing

contexts = {
    "general": {
        "overview": "Context7 is a knowledge management system for organizing and retrieving information.",
        "purpose": "Designed to provide contextual information based on queries.",
        "capabilities": "Can search across different types of contexts including general, technical, and documentation."
    },
    "technical": {
        "architecture": "Built using modular components for extensibility.",
        "api": "Provides RESTful API endpoints for integration.",
        "data": "Supports various data formats including JSON, XML, and plain text."
    },
    "documentation": {
        "usage": "Use the search function with appropriate context_type parameter.",
        "examples": "Query with specific keywords for best results.",
        "troubleshooting": "Check query syntax and context_type if no results are found."
    }
}

@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the web for current information about any topic.
    
    BEST FOR: Finding multiple results, recent information, specific topics like "nebula images",
    "black hole discoveries", "Hubble telescope photos", "James Webb telescope images", etc.
    
    NOTE: For James Webb Space Telescope images, this is the primary tool to use since JWST images
    are not in the NASA APOD or Mars Rover APIs.
    
    Args:
        query: Search query (e.g., "James Webb telescope latest images", "JWST nebula photos", "Webb telescope discoveries")
    
    Returns:
        Web search results with relevant links and snippets
    """
    try:
        # Use search method to get full results with URLs
        search_results = client.search(query=query, max_results=5)
        
        # Format results with URLs
        formatted_results = []
        for i, result in enumerate(search_results.get('results', []), 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   {result.get('content', 'No description')[:200]}...\n"
                f"   🔗 {result.get('url', 'No URL')}"
            )
        
        if formatted_results:
            return f"Search results for: {query}\n\n" + "\n\n".join(formatted_results)
        else:
            return f"No results found for: {query}"
            
    except Exception as e:
        # Fallback to get_search_context if search fails
        return client.get_search_context(query=query)

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

@mcp.tool()
def context7_search(query: str, context_type: str = "general") -> str:
    """
    Search and retrieve information from Context7's knowledge base.
    
    Args:
        query: The search query or question
        context_type: Type of context to search (general, technical, documentation)
    
    Returns:
        Relevant contextual information based on the query
    """
    
    query_lower = query.lower()
    context_data = contexts.get(context_type, contexts["general"])
    
    relevant_info = []
    for key, value in context_data.items():
        if key in query_lower or any(word in value.lower() for word in query_lower.split()):
            relevant_info.append(f"{key}: {value}")
    
    if relevant_info:
        return f"Context7 found the following information:\n" + "\n".join(relevant_info)
    else:
        return f"Context7: No specific information found for '{query}' in {context_type} context. Try refining your query or changing the context type."

@mcp.tool()
def get_top_headlines(country: str = "us", category: Optional[str] = None, query: Optional[str] = None) -> str:
    """
    Get top news headlines from NewsAPI.
    
    NOTE: The 'technology' category includes gaming news. For specific tech topics like AI,
    use the query parameter or use search_news instead.
    
    Args:
        country: 2-letter country code (e.g., 'us', 'gb', 'ca')
        category: Category of news (business, entertainment, general, health, science, sports, technology)
        query: Keywords to search for in headlines (e.g., "AI", "artificial intelligence", "machine learning")
    
    Returns:
        Top news headlines with title, description, and URL
    """
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not found in environment variables. Please add it to your .env file."
    
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "country": country,
        "pageSize": 5
    }
    
    if category:
        params["category"] = category
    if query:
        params["q"] = query
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = []
            for article in data["articles"][:5]:
                articles.append(f"📰 {article['title']}\n   {article.get('description', 'No description')}\n   🔗 {article['url']}")
            return "\n\n".join(articles)
        else:
            return "No news articles found for the specified criteria."
    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {str(e)}"

@mcp.tool()
def search_news(query: str, from_date: Optional[str] = None, sort_by: str = "relevancy") -> str:
    """
    Search for news articles about any topic.
    
    GOOD FOR: Finding recent news about space discoveries, NASA missions, telescope findings,
    "Hubble nebula discoveries", "James Webb black hole news", etc.
    
    Args:
        query: Keywords to search for (e.g., "nebula discovery", "black hole", "Hubble telescope")
        from_date: Oldest date for articles (YYYY-MM-DD format). Defaults to 7 days ago.
        sort_by: How to sort results (relevancy, popularity, publishedAt)
    
    Returns:
        Recent news articles with titles, dates, and links
    """
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not found in environment variables. Please add it to your .env file."
    
    url = "https://newsapi.org/v2/everything"
    
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "from": from_date,
        "sortBy": sort_by,
        "pageSize": 5,
        "language": "en"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = []
            for article in data["articles"][:5]:
                articles.append(f"📰 {article['title']}\n   📅 {article['publishedAt'][:10]}\n   {article.get('description', 'No description')}\n   🔗 {article['url']}")
            return "\n\n".join(articles)
        else:
            return f"No news articles found for '{query}'."
    except requests.exceptions.RequestException as e:
        return f"Error searching news: {str(e)}"

@mcp.tool()
def get_astronomy_picture(date: Optional[str] = None) -> str:
    """
    Get NASA's Astronomy Picture of the Day (APOD) - returns ONE picture for a specific date.
    
    IMPORTANT: This returns only ONE picture per date. To get multiple pictures, call with different dates.
    Cannot search for specific objects (nebulas, black holes, etc) - it's just the picture of that day.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today). 
              Example: "2024-12-25" for Christmas 2024's picture
    
    Returns:
        Single astronomy picture with title, explanation, and URL for the specified date
    """
    url = "https://api.nasa.gov/planetary/apod"
    params = {
        "api_key": NASA_API_KEY
    }
    
    if date:
        params["date"] = date
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        result = f"🌌 NASA Astronomy Picture of the Day\n"
        result += f"📅 Date: {data['date']}\n"
        result += f"📸 Title: {data['title']}\n"
        result += f"📝 Explanation: {data['explanation']}\n"
        
        if data.get('media_type') == 'image':
            result += f"🖼️ Image URL: {data['url']}\n"
            if data.get('hdurl'):
                result += f"🔭 HD Image: {data['hdurl']}"
        elif data.get('media_type') == 'video':
            result += f"🎥 Video URL: {data['url']}"
        
        return result
    except requests.exceptions.RequestException as e:
        return f"Error fetching astronomy picture: {str(e)}"

@mcp.tool()
def get_mars_rover_photos(rover: str = "curiosity", sol: Optional[int] = None, camera: Optional[str] = None) -> str:
    """
    Get photos from NASA's Mars rovers - returns multiple photos from a specific Martian day (sol).
    
    IMPORTANT: Returns photos from ONE specific sol (Martian day). To get varied photos, try different sols.
    Cannot search for specific features - returns whatever the rover photographed that day.
    
    Args:
        rover: Rover name (curiosity, opportunity, spirit, perseverance)
        sol: Martian day (sol) to get photos from. Try different sols for variety:
             - Curiosity: 1-4000+ (active since 2012)
             - Perseverance: 1-1000+ (active since 2021)
        camera: Specific camera (FHAZ, RHAZ, MAST, CHEMCAM, NAVCAM)
    
    Returns:
        Up to 5 Mars rover photos from the specified sol
    """
    # If no sol specified, use a recent sol that likely has photos
    if sol is None:
        sol = 1000 if rover == "curiosity" else 100
    
    url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
    params = {
        "api_key": NASA_API_KEY,
        "sol": sol
    }
    
    if camera:
        params["camera"] = camera
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("photos"):
            result = f"🚀 Mars Rover: {rover.capitalize()}\n"
            result += f"📅 Sol (Martian Day): {sol}\n"
            result += f"📸 Found {len(data['photos'])} photos\n\n"
            
            # Show first 5 photos
            for photo in data["photos"][:5]:
                result += f"📷 Camera: {photo['camera']['full_name']}\n"
                result += f"🌍 Earth Date: {photo['earth_date']}\n"
                result += f"🔗 Image: {photo['img_src']}\n\n"
            
            return result
        else:
            return f"No photos found for {rover} rover on sol {sol}. Try a different sol or camera."
    except requests.exceptions.RequestException as e:
        return f"Error fetching Mars rover photos: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")