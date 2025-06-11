import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🤖 Multi-Agent Voice RAG Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .query-response {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .source-item {
        background: #edf2f7;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 3px solid #4299e1;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .setup-progress {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4299e1;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for global settings
with st.sidebar:
    st.title("🔧 Global Settings")
    
    # API Key Management
    openai_api_key = os.getenv("OPENAI_API_KEY")
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    
    with st.expander("🔑 API Keys", expanded=True):
        if openai_api_key:
            st.success("✅ OpenAI API Key loaded")
            masked_key = openai_api_key[:8] + "..." + openai_api_key[-4:]
            st.text(f"Key: {masked_key}")
        else:
            st.warning("⚠️ OpenAI API Key missing")
            manual_openai_key = st.text_input("OpenAI API Key:", type="password", key="sidebar_openai")
            if manual_openai_key:
                os.environ["OPENAI_API_KEY"] = manual_openai_key
                st.success("API key saved!")
                st.rerun()
        
        if firecrawl_api_key:
            st.success("✅ Firecrawl API Key loaded")
            masked_key = firecrawl_api_key[:8] + "..." + firecrawl_api_key[-4:]
            st.text(f"Key: {masked_key}")
        else:
            st.info("ℹ️ Firecrawl API Key (for Web Crawl tab)")
            manual_firecrawl_key = st.text_input("Firecrawl API Key:", type="password", key="sidebar_firecrawl")
            if manual_firecrawl_key:
                os.environ["FIRECRAWL_API_KEY"] = manual_firecrawl_key
                st.success("Firecrawl API key saved!")
                st.rerun()
    
    # Global Voice Settings
    with st.expander("🎤 Global Voice Settings", expanded=False):
        voice_options = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        selected_voice = st.selectbox("Default Voice", voice_options, index=voice_options.index("nova"))
        st.session_state.global_voice = selected_voice
    
    # Environment Info
    with st.expander("ℹ️ Environment Info", expanded=False):
        st.markdown("""
        **Required .env variables:**
        ```
        OPENAI_API_KEY=sk-your-key-here
        FIRECRAWL_API_KEY=fc-your-key-here
        ```
        
        **Installation:**
        ```bash
        pip install streamlit openai openai-agents
        pip install firecrawl-py python-dotenv
        pip install faiss-cpu pydantic rich
        ```
        """)

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; text-align: center;">🤖 Multi-Agent Voice RAG Platform</h1>
    <p style="color: white; margin: 0; text-align: center; font-size: 1.2em;">
        Audio Tours • Customer Support • Web Crawling • Voice AI
    </p>
</div>
""", unsafe_allow_html=True)

# Check if basic requirements are met
if not os.getenv("OPENAI_API_KEY"):
    st.error("🔑 OpenAI API Key required. Please add it in the sidebar or your .env file.")
    st.stop()

# Initialize session state for web crawl functionality
def init_webcrawl_session_state():
    """Initialize session state for web crawl functionality"""
    defaults = {
        "webcrawl_setup_complete": False,
        "webcrawl_vector_store": None,
        "webcrawl_embedding_model": None,
        "webcrawl_crawled_pages": [],
        "webcrawl_query_history": [],
        "webcrawl_audio_files": [],
        "webcrawl_last_crawl_url": "",
        "webcrawl_last_search_results": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize webcrawl session state
init_webcrawl_session_state()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎧 Audio Tour Agent", "🎤 Customer Support Agent", "🕷️ Web Crawl Agent"])

# Tab 1: Audio Tour Agent
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    try:
        # Import and run audio tour agent
        import asyncio
        from pathlib import Path
        
        # Audio Tour Agent Implementation
        def tts_audio_tour(text, voice="nova"):
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            speech_file_path = Path(__file__).parent / f"audio_tour_{voice}.mp3"
                
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text
            )
            response.stream_to_file(speech_file_path)
            return speech_file_path

        def run_async_tour(func, *args, **kwargs):
            try:
                return asyncio.run(func(*args, **kwargs))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(*args, **kwargs))

        st.markdown("### 🎧 AI Audio Tour Generator")
        st.info("Create personalized audio tours for any location with AI-powered narration.")
        
        # Tour configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            location = st.text_input("📍 Location", placeholder="Enter a city, landmark, or location...", key="tour_location")
            interests = st.multiselect(
                "🎯 Interests",
                options=["History", "Architecture", "Culinary", "Culture"],
                default=["History", "Architecture"],
                key="tour_interests"
            )
        
        with col2:
            duration = st.slider("⏱️ Duration (minutes)", 5, 60, 10, 5, key="tour_duration")
            tour_voice = st.selectbox("🎙️ Voice", voice_options, 
                                    index=voice_options.index(st.session_state.get('global_voice', 'nova')),
                                    key="tour_voice_select")
        
        # Add the simple tour generation function before the button logic
        async def generate_simple_tour(location: str, interests: list, duration: int) -> str:
            """Simple tour generation using OpenAI directly"""
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            interests_str = ", ".join(interests)
            target_words = duration * 150  # 150 words per minute
            
            prompt = f"""Create a {duration}-minute audio tour of {location} focusing on {interests_str}.
            
            Guidelines:
            - Write approximately {target_words} words for a {duration}-minute tour
            - Use a conversational, engaging tone suitable for audio
            - Include specific landmarks, historical facts, and interesting stories
            - Structure with clear sections and smooth transitions
            - Use vivid descriptions that help listeners visualize the location
            - Make it feel like a personal guided tour experience
            
            Focus on: {interests_str}
            
            Create an engaging, informative tour that brings {location} to life for visitors."""
            
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert tour guide creating engaging audio tour scripts. Write in a conversational, warm tone suitable for text-to-speech conversion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=target_words + 500
            )
            
            return response.choices[0].message.content

        if st.button("🎧 Generate Audio Tour", type="primary", key="tour_btn"):
            if not location or not interests:
                st.error("Please enter a location and select interests.")
            else:
                with st.spinner(f"Creating your tour of {location}..."):
                    try:
                        # Try to import the full tour manager
                        try:
                            from manager import TourManager
                            mgr = TourManager()
                            final_tour = run_async_tour(mgr.run, location, interests, duration)
                        except ImportError:
                            # Fallback to simple tour generation
                            st.info("Using simplified tour generation (full agent system not available)")
                            final_tour = asyncio.run(generate_simple_tour(location, interests, duration))
                        
                        # Display tour content
                        with st.expander("📝 Tour Content", expanded=True):
                            st.markdown(final_tour)
                        
                        # Generate audio
                        with st.spinner("🎙️ Generating audio..."):
                            tour_audio = tts_audio_tour(final_tour, tour_voice)
                            
                            st.markdown("### 🔊 Audio Tour")
                            st.audio(tour_audio, format="audio/mp3")
                            
                            # Download button
                            with open(tour_audio, "rb") as file:
                                st.download_button(
                                    "📥 Download Audio Tour",
                                    data=file,
                                    file_name=f"{location.replace(' ', '_')}_tour.mp3",
                                    mime="audio/mp3",
                                    key="download_tour"
                                )
                    except Exception as e:
                        st.error(f"Tour generation failed: {str(e)}")
                        
                        # Show fallback option
                        if st.button("🔄 Try Simple Tour Generation", key="fallback_tour"):
                            try:
                                simple_tour = asyncio.run(generate_simple_tour(location, interests, duration))
                                st.markdown("### 📝 Simple Tour Content")
                                st.markdown(simple_tour)
                                
                                # Generate audio for simple tour
                                tour_audio = tts_audio_tour(simple_tour, tour_voice)
                                st.audio(tour_audio, format="audio/mp3")
                                
                            except Exception as fallback_error:
                                st.error(f"Fallback tour generation also failed: {str(fallback_error)}")
    
    except Exception as e:
        st.error(f"Audio Tour Agent error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Customer Support Agent
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    try:
        st.markdown("### 🎤 AI Customer Support Agent")
        st.info("Voice-powered customer support with real-time assistance and audio responses.")
        
        # Customer Support Implementation
        def tts_support(text, voice="nova"):
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            speech_file_path = Path(__file__).parent / f"support_{voice}.mp3"
                
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text
            )
            response.stream_to_file(speech_file_path)
            return speech_file_path
        
        # Support agent configuration
        col1, col2 = st.columns([3, 1])
        
        with col1:
            support_query = st.text_area(
                "💬 How can we help you today?",
                placeholder="Describe your issue or question...",
                height=100,
                key="support_query"
            )
        
        with col2:
            support_voice = st.selectbox("🎙️ Support Voice", voice_options,
                                       index=voice_options.index(st.session_state.get('global_voice', 'nova')),
                                       key="support_voice")
            support_style = st.selectbox("📞 Response Style", 
                                       ["Professional", "Friendly", "Technical", "Empathetic"],
                                       key="support_style")
        
        if st.button("🎤 Get Support", type="primary", key="support_btn"):
            if not support_query:
                st.error("Please describe your issue.")
            else:
                with st.spinner("Analyzing your request..."):
                    try:
                        from openai import OpenAI
                        
                        api_key = os.getenv("OPENAI_API_KEY")
                        client = OpenAI(api_key=api_key)
                        
                        # Generate support response
                        system_prompt = f"""You are a helpful customer support agent with a {support_style.lower()} tone. 
                        Provide clear, actionable solutions to customer issues. Be empathetic and professional."""
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": support_query}
                            ],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        
                        support_response = response.choices[0].message.content
                        
                        # Display response
                        st.markdown("### 💬 Support Response")
                        st.markdown(support_response)
                        
                        # Generate audio
                        with st.spinner("🎙️ Generating audio response..."):
                            support_audio = tts_support(support_response, support_voice)
                            
                            st.markdown("### 🔊 Audio Response")
                            st.audio(support_audio, format="audio/mp3")
                            
                            # Download button
                            with open(support_audio, "rb") as file:
                                st.download_button(
                                    "📥 Download Response",
                                    data=file,
                                    file_name="support_response.mp3",
                                    mime="audio/mp3",
                                    key="download_support"
                                )
                    
                    except Exception as e:
                        st.error(f"Support response failed: {str(e)}")
    
    except Exception as e:
        st.error(f"Customer Support Agent error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Web Crawl Agent - FULL INTEGRATION
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    try:
        # Import all necessary components for web crawling
        from typing import List, Dict, Optional
        import tempfile
        import uuid
        from datetime import datetime
        import time
        import asyncio
        import logging
        import pickle
        import numpy as np
        import faiss
        from firecrawl import FirecrawlApp
        from openai import AsyncOpenAI

        # Constants for web crawl
        FAISS_INDEX_PATH = "web_docs_faiss_index.bin"
        DOCUMENTS_METADATA_PATH = "web_docs_metadata.pkl"
        EMBEDDING_DIMENSIONS = 1536

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Web Crawl Classes
        class OpenAIEmbeddings:
            """OpenAI embeddings wrapper"""
            
            def __init__(self, api_key: str):
                self.client = AsyncOpenAI(api_key=api_key)
                self.model = "text-embedding-3-small"
                self.dimensions = EMBEDDING_DIMENSIONS
            
            async def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Generate embeddings for multiple texts"""
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts
                    )
                    return [data.embedding for data in response.data]
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    raise
            
            async def embed_query(self, text: str) -> List[float]:
                """Generate embedding for a single query"""
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=[text]
                    )
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"Error generating query embedding: {str(e)}")
                    raise

        class WebDocsFAISSStore:
            """FAISS-based vector store for web documentation"""
            
            def __init__(self, embedding_model: OpenAIEmbeddings, dimension: int = EMBEDDING_DIMENSIONS):
                self.embedding_model = embedding_model
                self.dimension = dimension
                self.index = faiss.IndexFlatIP(dimension)
                self.documents_metadata = []
                self.is_trained = False
                
            async def add_documents(self, pages: List[Dict]):
                """Add web pages and their embeddings to the FAISS index"""
                try:
                    # Extract content for embedding
                    texts = [page["content"] for page in pages if page.get("content")]
                    if not texts:
                        return
                    
                    # Generate embeddings
                    embeddings = await self.embedding_model.embed_documents(texts)
                    embeddings_array = np.array(embeddings, dtype='float32')
                    
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings_array)
                    
                    # Add to FAISS index
                    self.index.add(embeddings_array)
                    
                    # Store metadata
                    for page in pages:
                        if page.get("content"):
                            self.documents_metadata.append({
                                'content': page["content"],
                                'url': page["url"],
                                'metadata': page["metadata"]
                            })
                    
                    self.is_trained = True
                    logger.info(f"Added {len(pages)} web pages to FAISS index")
                    
                except Exception as e:
                    logger.error(f"Error adding documents to FAISS: {str(e)}")
                    raise
            
            async def search(self, query: str, k: int = 3) -> List[Dict]:
                """Search for similar documents"""
                try:
                    if not self.is_trained:
                        return []
                    
                    # Generate query embedding
                    query_embedding = await self.embedding_model.embed_query(query)
                    query_array = np.array([query_embedding], dtype='float32')
                    
                    # Normalize query embedding
                    faiss.normalize_L2(query_array)
                    
                    # Search
                    scores, indices = self.index.search(query_array, k)
                    
                    results = []
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if idx >= 0 and idx < len(self.documents_metadata):
                            results.append({
                                'content': self.documents_metadata[idx]['content'],
                                'url': self.documents_metadata[idx]['url'],
                                'metadata': self.documents_metadata[idx]['metadata'],
                                'score': float(score)
                            })
                    
                    return results
                    
                except Exception as e:
                    logger.error(f"Error searching FAISS index: {str(e)}")
                    return []
            
            def get_stats(self) -> Dict:
                """Get statistics about the vector store"""
                return {
                    'total_pages': len(self.documents_metadata),
                    'index_size': self.index.ntotal,
                    'is_trained': self.is_trained,
                    'dimension': self.dimension
                }

        # Web Crawl Functions
        async def crawl_documentation_async(firecrawl_api_key: str, url: str, limit: int = 5):
            """Crawl documentation with Firecrawl v1 API"""
            try:
                from firecrawl import FirecrawlApp, ScrapeOptions
                
                firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
                pages = []
                
                st.write(f"🔍 Starting crawl of {url} (limit: {limit} pages)")
                
                # Try crawling first
                try:
                    st.info("🕷️ Attempting to crawl multiple pages...")
                    
                    response = firecrawl.crawl_url(
                        url,
                        limit=limit,
                        scrape_options=ScrapeOptions(formats=['markdown'])
                    )
                    
                    st.success("✅ Crawl initiated successfully!")
                    
                    if response and hasattr(response, 'success') and response.success:
                        if hasattr(response, 'data') and response.data:
                            batch_pages = response.data
                        else:
                            batch_pages = []
                    else:
                        batch_pages = []
                    
                    st.write(f"📊 Processing {len(batch_pages)} pages from crawl response")
                    
                    crawl_progress = st.progress(0)
                    page_count = 0
                    
                    for page in batch_pages:
                        try:
                            if hasattr(page, 'markdown'):
                                content = page.markdown
                                metadata = getattr(page, 'metadata', {}) or {}
                                source_url = metadata.get('sourceURL', '') or metadata.get('url', '') or url
                            elif isinstance(page, dict):
                                content = page.get('markdown', '') or page.get('content', '') or page.get('html', '')
                                metadata = page.get('metadata', {})
                                source_url = metadata.get('sourceURL', '') or metadata.get('url', '') or page.get('url', '') or url
                            else:
                                content = str(page) if page else ''
                                metadata = {}
                                source_url = url
                            
                            if content and len(content.strip()) > 100:
                                pages.append({
                                    "content": content,
                                    "url": source_url,
                                    "metadata": {
                                        "title": metadata.get('title', f'Page {page_count + 1}'),
                                        "description": metadata.get('description', ''),
                                        "language": metadata.get('language', 'en'),
                                        "crawl_date": datetime.now().isoformat(),
                                        "word_count": len(content.split()),
                                        "char_count": len(content)
                                    }
                                })
                                
                                page_count += 1
                                crawl_progress.progress(min(page_count / limit, 1.0))
                                st.write(f"📄 Processed: {metadata.get('title', f'Page {page_count}')}")
                                
                                if page_count >= limit:
                                    break
                                    
                        except Exception as page_error:
                            st.warning(f"⚠️ Error processing page: {str(page_error)}")
                            continue
                    
                    crawl_progress.progress(1.0)
                    
                except Exception as crawl_error:
                    st.warning(f"⚠️ Multi-page crawling failed: {str(crawl_error)}")
                    st.info("🔄 Trying single page scraping instead...")
                    
                    # Fallback: Single page scraping
                    try:
                        single_page = firecrawl.scrape_url(url, formats=['markdown'])
                        
                        if single_page and hasattr(single_page, 'success') and single_page.success:
                            if hasattr(single_page, 'markdown') and single_page.markdown:
                                content = single_page.markdown
                                metadata = getattr(single_page, 'metadata', {}) or {}
                            else:
                                content = ''
                                metadata = {}
                            
                            if content and len(content.strip()) > 50:
                                pages.append({
                                    "content": content,
                                    "url": url,
                                    "metadata": {
                                        "title": metadata.get('title', 'Main Page'),
                                        "description": metadata.get('description', ''),
                                        "language": metadata.get('language', 'en'),
                                        "crawl_date": datetime.now().isoformat(),
                                        "word_count": len(content.split()),
                                        "char_count": len(content)
                                    }
                                })
                                
                                st.success("✅ Successfully scraped single page")
                        
                    except Exception as scrape_error:
                        raise Exception(f"All scraping methods failed. {str(scrape_error)}")
                
                return pages
                
            except Exception as e:
                st.error(f"❌ Crawling failed: {str(e)}")
                return []

        async def process_webcrawl_query(query: str, vector_store: WebDocsFAISSStore, openai_api_key: str, voice: str) -> Dict:
            """Process user query and generate voice response"""
            try:
                st.session_state.webcrawl_query_history.append(query)
                
                st.info("🔄 Step 1: Searching documentation...")
                
                search_results = await vector_store.search(query, k=3)
                st.write(f"Found {len(search_results)} relevant pages")
                
                if not search_results:
                    raise Exception("No relevant documentation found for your query")
                
                st.info("🔄 Step 2: Preparing context...")
                
                context = "Based on the following documentation:\n\n"
                sources = []
                
                for i, result in enumerate(search_results, 1):
                    content = result['content']
                    url = result['url']
                    metadata = result['metadata']
                    score = result['score']
                    title = metadata.get('title', 'Untitled')
                    
                    context += f"Source {i} - {title} ({url}):\n{content[:1000]}...\n\n"
                    sources.append(f"{title} - {url}")
                    st.write(f"📄 {i}. {title} (Score: {score:.3f})")
                
                context += f"\nUser Question: {query}\n\n"
                context += "Please provide a clear, helpful answer based on the documentation above."
                
                st.info("🔄 Step 3: Generating response...")
                
                client = AsyncOpenAI(api_key=openai_api_key)
                
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful documentation assistant. Answer questions based on the provided documentation clearly and concisely. Include relevant examples when available."},
                        {"role": "user", "content": context}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                text_response = response.choices[0].message.content
                st.write(f"Generated response of length: {len(text_response)}")
                
                st.info("🔄 Step 4: Generating audio...")
                
                audio_response = await client.audio.speech.create(
                    model="tts-1-hd",
                    voice=voice,
                    input=text_response,
                    response_format="mp3"
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"web_response_{timestamp}_{voice}.mp3"
                audio_path = os.path.join(tempfile.gettempdir(), audio_filename)
                
                with open(audio_path, "wb") as f:
                    f.write(audio_response.content)
                
                st.success("✅ Query processing complete!")
                
                return {
                    "status": "success",
                    "text_response": text_response,
                    "audio_path": audio_path,
                    "sources": sources,
                    "search_results": search_results
                }
            
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "query": query
                }

        # WEB CRAWL TAB INTERFACE
        st.markdown("### 🕷️ Web Crawling Voice RAG Agent")
        st.info("Crawl documentation websites and create voice-powered Q&A systems with local FAISS storage.")

        # Check for required API keys
        if not os.getenv("FIRECRAWL_API_KEY"):
            st.warning("⚠️ Firecrawl API Key required for web crawling. Add it in the sidebar.")
            st.info("Get your Firecrawl API key from [firecrawl.dev](https://firecrawl.dev)")
        else:
            # Show setup status
            if not st.session_state.webcrawl_setup_complete:
                st.markdown("#### 🚀 Getting Started")
                
                # Web crawl configuration
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    crawl_url = st.text_input(
                        "🌐 Documentation URL",
                        placeholder="https://docs.example.com",
                        key="webcrawl_url"
                    )
                
                with col2:
                    crawl_limit = st.slider("📄 Pages to Crawl", 1, 20, 5, key="webcrawl_limit")
                
                # Example URLs
                st.markdown("**📖 Example Documentation URLs:**")
                examples = [
                    "https://docs.streamlit.io/",
                    "https://docs.python.org/3/",
                    "https://platform.openai.com/docs/",
                    "https://docs.github.com/",
                    "https://firebase.google.com/docs/"
                ]
                
                example_cols = st.columns(len(examples))
                for i, (col, url) in enumerate(zip(example_cols, examples)):
                    with col:
                        if st.button(f"📝 {url.split('//')[1].split('/')[0].replace('docs.', '').replace('platform.', '').title()}", 
                                   key=f"webcrawl_example_{i}"):
                            st.session_state.webcrawl_url = url
                            st.rerun()
                
                # Start crawling
                if st.button("🕷️ Start Crawling", type="primary", key="webcrawl_start_btn"):
                    if not crawl_url:
                        st.error("Please enter a URL to crawl.")
                    else:
                        with st.spinner(f"Crawling and processing {crawl_url}..."):
                            try:
                                # Initialize vector store
                                embedding_model = OpenAIEmbeddings(os.getenv("OPENAI_API_KEY"))
                                vector_store = WebDocsFAISSStore(embedding_model)
                                
                                # Crawl documentation
                                pages = asyncio.run(crawl_documentation_async(
                                    os.getenv("FIRECRAWL_API_KEY"),
                                    crawl_url,
                                    limit=crawl_limit
                                ))
                                
                                if not pages:
                                    st.error("❌ No pages were crawled. Please check the URL and try again.")
                                else:
                                    st.success(f"✅ Successfully crawled {len(pages)} pages!")
                                    
                                    # Process and store embeddings
                                    st.info("🔄 Processing content and generating embeddings...")
                                    progress_bar = st.progress(0)
                                    asyncio.run(vector_store.add_documents(pages))
                                    progress_bar.progress(1.0)
                                    
                                    # Update session state
                                    st.session_state.webcrawl_vector_store = vector_store
                                    st.session_state.webcrawl_embedding_model = embedding_model
                                    st.session_state.webcrawl_crawled_pages = pages
                                    st.session_state.webcrawl_setup_complete = True
                                    st.session_state.webcrawl_last_crawl_url = crawl_url
                                    
                                    st.success("✅ Documentation processing complete! You can now ask questions.")
                                    
                                    # Show crawled pages summary
                                    st.markdown("### 📄 Crawled Pages Summary")
                                    for i, page in enumerate(pages, 1):
                                        st.markdown(f"**{i}.** {page['metadata'].get('title', 'Untitled')}")
                                        st.markdown(f"🔗 URL: {page['url']}")
                                        st.markdown(f"📝 Content length: {len(page['content'])} characters")
                                        if i < len(pages):
                                            st.markdown("---")
                                    
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"❌ Error during crawling process: {str(e)}")
            
            else:
                # Query interface - when setup is complete
                st.markdown("### 💬 Ask Questions About Your Documentation")
                
                # Show crawled documentation info
                if st.session_state.webcrawl_vector_store:
                    stats = st.session_state.webcrawl_vector_store.get_stats()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Pages Crawled", stats['total_pages'])
                    with col2:
                        st.metric("🔍 Vectors Stored", stats['index_size'])
                    with col3:
                        st.metric("🌐 Source", st.session_state.webcrawl_last_crawl_url.split('//')[-1][:20] + "...")
                
                # Query input and voice selection
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    webcrawl_query = st.text_input(
                        "What would you like to know about the documentation?",
                        placeholder="e.g., How do I get started? What are the main features?",
                        key="webcrawl_query_input"
                    )
                
                with col2:
                    webcrawl_voice = st.selectbox(
                        "🎙️ Response Voice",
                        voice_options,
                        index=voice_options.index(st.session_state.get('global_voice', 'nova')),
                        key="webcrawl_voice_select"
                    )
                
                # Query processing
                if st.button("💬 Ask Question", type="primary", key="webcrawl_ask_btn"):
                    if not webcrawl_query:
                        st.error("Please enter a question.")
                    else:
                        with st.status("🔄 Processing your query...", expanded=True) as status:
                            try:
                                result = asyncio.run(process_webcrawl_query(
                                    webcrawl_query,
                                    st.session_state.webcrawl_vector_store,
                                    os.getenv("OPENAI_API_KEY"),
                                    webcrawl_voice
                                ))
                                
                                if result["status"] == "success":
                                    status.update(label="✅ Query processed successfully!", state="complete")
                                    
                                    # Display text response
                                    st.markdown("### 💬 Response")
                                    st.markdown(f"""
                                    <div class="query-response">
                                        {result["text_response"]}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Audio response
                                    if "audio_path" in result:
                                        st.markdown(f"### 🔊 Audio Response (Voice: {webcrawl_voice})")
                                        
                                        try:
                                            with open(result["audio_path"], "rb") as audio_file:
                                                audio_bytes = audio_file.read()
                                                st.audio(audio_bytes, format="audio/mp3")
                                                
                                                st.download_button(
                                                    label="📥 Download Audio Response",
                                                    data=audio_bytes,
                                                    file_name=f"webcrawl_response_{webcrawl_voice}.mp3",
                                                    mime="audio/mp3",
                                                    key="webcrawl_download_audio"
                                                )
                                        except Exception as e:
                                            st.error(f"Error loading audio: {str(e)}")
                                    
                                    # Sources
                                    st.markdown("### 📚 Sources")
                                    for i, source in enumerate(result["sources"], 1):
                                        st.markdown(f"""
                                        <div class="source-item">
                                            📄 {i}. {source}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Store detailed results for display outside status context
                                    st.session_state.webcrawl_last_search_results = result["search_results"]
                                
                                else:
                                    status.update(label="❌ Error processing query", state="error")
                                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                            
                            except Exception as e:
                                status.update(label="❌ Error processing query", state="error")
                                st.error(f"Error processing query: {str(e)}")
                
                # Display detailed search results outside the status context
                if hasattr(st.session_state, 'webcrawl_last_search_results') and st.session_state.webcrawl_last_search_results:
                    with st.expander("🔍 Detailed Search Results", expanded=False):
                        for i, search_result in enumerate(st.session_state.webcrawl_last_search_results, 1):
                            st.markdown(f"**Result {i}** (Score: {search_result['score']:.3f})")
                            st.markdown(f"🔗 **URL:** {search_result['url']}")
                            st.markdown(f"📝 **Title:** {search_result['metadata'].get('title', 'Untitled')}")
                            
                            content_preview = search_result['content'][:300] + "..." if len(search_result['content']) > 300 else search_result['content']
                            st.text_area(
                                f"Content Preview {i}",
                                value=content_preview,
                                height=100,
                                key=f"webcrawl_search_content_{i}",
                                disabled=True
                            )
                            
                            if i < len(st.session_state.webcrawl_last_search_results):
                                st.markdown("---")
                
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Show Statistics", key="webcrawl_stats_btn"):
                        if st.session_state.webcrawl_vector_store:
                            stats = st.session_state.webcrawl_vector_store.get_stats()
                            
                            st.markdown("#### 📈 Documentation Statistics")
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric("Web Pages", stats['total_pages'])
                            
                            with metric_col2:
                                st.metric("Vector Count", stats['index_size'])
                            
                            with metric_col3:
                                st.metric("Embedding Dimension", stats['dimension'])
                            
                            st.success(f"Documentation from: {st.session_state.webcrawl_last_crawl_url}")
                
                with col2:
                    if st.button("🔄 Recrawl Documentation", key="webcrawl_recrawl_btn"):
                        st.session_state.webcrawl_setup_complete = False
                        st.session_state.webcrawl_vector_store = None
                        st.session_state.webcrawl_crawled_pages = []
                        st.success("Ready to crawl new documentation!")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Clear All Data", key="webcrawl_clear_btn"):
                        # Clear all webcrawl session state
                        for key in list(st.session_state.keys()):
                            if key.startswith('webcrawl_'):
                                del st.session_state[key]
                        # Reinitialize
                        init_webcrawl_session_state()
                        st.success("🗑️ All webcrawl data cleared!")
                        st.rerun()
                
                # Query history
                if st.session_state.webcrawl_query_history:
                    st.markdown("### 📝 Recent Queries")
                    
                    for i, past_query in enumerate(reversed(st.session_state.webcrawl_query_history[-5:]), 1):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i}.** {past_query}")
                        with col2:
                            if st.button(f"🔄", key=f"webcrawl_requery_{i}_{hash(past_query)}", help="Ask this question again"):
                                st.session_state.webcrawl_query_input = past_query
                                st.rerun()
                
                # System information
                with st.expander("ℹ️ Web Crawl System Information", expanded=False):
                    st.markdown("""
                    ### 🎙️ Web Crawling Voice RAG Agent - Integrated Version
                    
                    **Features:**
                    - 🕷️ **Web Crawling**: Uses Firecrawl v1 API to scrape documentation sites
                    - 🗂️ **Local Vector Storage**: FAISS for fast, local similarity search
                    - 🎤 **Voice Responses**: OpenAI TTS with multiple voice options
                    - 🔍 **Semantic Search**: Find relevant content using embeddings
                    - 💾 **Session Storage**: Maintains data during your session
                    
                    **Technology Stack:**
                    - Web Crawling: Firecrawl API v1
                    - Vector Database: FAISS (Local)
                    - Embeddings: OpenAI text-embedding-3-small
                    - LLM: OpenAI GPT-4
                    - TTS: OpenAI Text-to-Speech
                    - Framework: Streamlit
                    """)
                    
                    # Technical details
                    if st.session_state.webcrawl_vector_store:
                        stats = st.session_state.webcrawl_vector_store.get_stats()
                        st.json({
                            "vector_store_stats": stats,
                            "session_info": {
                                "total_queries": len(st.session_state.webcrawl_query_history),
                                "last_crawl_url": st.session_state.webcrawl_last_crawl_url,
                                "pages_crawled": len(st.session_state.webcrawl_crawled_pages)
                            }
                        })

    except Exception as e:
        st.error(f"Web Crawl Agent error: {str(e)}")
        st.info("""
        **Required dependencies for Web Crawl Agent:**
        ```bash
        pip install faiss-cpu numpy firecrawl-py
        ```
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 <strong>Multi-Agent Voice RAG Platform</strong></p>
    <p>Powered by OpenAI • Built with Streamlit • Voice AI Integration</p>
</div>
""", unsafe_allow_html=True)

# Session state management
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.global_voice = "nova"