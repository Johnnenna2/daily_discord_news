import os
import requests
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pytz
from openai import OpenAI
import feedparser
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyNewsBot:
    def __init__(self):
        print("🚀 INITIALIZING PRE-MARKET NEWS BOT")
        print("=" * 50)
        
        # Load dotenv for local development
        try:
            from dotenv import load_dotenv
            if os.path.exists('.env'):
                load_dotenv()
                print("📁 Using local .env file")
            else:
                print("🔧 Using GitHub Actions environment variables")
        except ImportError:
            print("🔧 Using system environment variables")
        
        # Get environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        news_key = os.getenv('NEWS_API_KEY')
        
        # Validate required environment variables
        if not openai_key:
            raise ValueError("❌ OPENAI_API_KEY environment variable is required")
        if not webhook_url:
            raise ValueError("❌ DISCORD_WEBHOOK_URL environment variable is required")
            
        self.openai_client = OpenAI(api_key=openai_key)
        self.webhook_url = webhook_url
        self.news_api_key = news_key
        
        # Trading-focused news sources
        self.news_sources = [
            'reuters', 'bloomberg', 'cnbc', 'marketwatch',
            'yahoo-finance', 'the-wall-street-journal'
        ]
        
        # Major stock symbols to monitor
        self.watchlist = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            'SPY', 'QQQ', 'IWM', 'DIA'
        ]
        
        self.est = pytz.timezone('US/Eastern')
        print("✅ Bot initialized successfully!")
        print("=" * 50)
    
    def test_openai(self):
        """Test OpenAI API connectivity"""
        print("🧪 Testing OpenAI API connection...")
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say 'API test successful' in exactly those words."}
                ],
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            print("✅ OpenAI API connection successful!")
            return True
            
        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            return False

    async def fetch_news_articles(self, hours_back: int = 16) -> List[Dict]:
        """Fetch recent financial news articles"""
        print(f"📰 Fetching news articles from last {hours_back} hours...")
        
        articles = []
        from_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # Try News API first if available
            if self.news_api_key:
                print("📡 Fetching from News API...")
                articles.extend(await self._fetch_from_newsapi(from_time))
            else:
                print("📡 News API key not provided, using RSS feeds only")
            
            # Always fetch from RSS feeds as primary/backup source
            print("📡 Fetching from RSS feeds...")
            rss_articles = await self._fetch_from_rss()
            articles.extend(rss_articles)
            
            # Remove duplicates and sort by relevance
            unique_articles = self._deduplicate_articles(articles)
            sorted_articles = sorted(unique_articles, key=lambda x: x.get('relevance_score', 0), reverse=True)[:12]
            
            print(f"📊 Found {len(sorted_articles)} relevant articles")
            return sorted_articles
            
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            logger.error(f"Error fetching news: {e}")
            return []
    
    async def _fetch_from_newsapi(self, from_time: datetime) -> List[Dict]:
        """Fetch from News API"""
        articles = []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'apiKey': self.news_api_key,
            'sources': ','.join(self.news_sources),
            'from': from_time.isoformat(),
            'sortBy': 'relevancy',
            'language': 'en',
            'pageSize': 50
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_articles = data.get('articles', [])
                        
                        for article in news_articles:
                            if article.get('title') and article.get('description'):
                                articles.append({
                                    'title': article['title'],
                                    'description': article['description'],
                                    'url': article['url'],
                                    'source': article['source']['name'],
                                    'published_at': article['publishedAt'],
                                    'relevance_score': self._calculate_relevance(article)
                                })
                    else:
                        print(f"⚠️ News API returned status {response.status}")
                        
            except Exception as e:
                print(f"❌ Error fetching from News API: {e}")
        
        return articles
    
    async def _fetch_from_rss(self) -> List[Dict]:
        """Fetch from RSS feeds"""
        articles = []
        rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.marketwatch.com/rss/topstories',
            'https://feeds.reuters.com/reuters/businessNews'
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:8]:  # Limit per feed
                    if hasattr(entry, 'title') and hasattr(entry, 'link'):
                        articles.append({
                            'title': entry.title,
                            'description': getattr(entry, 'summary', entry.title)[:200],
                            'url': entry.link,
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'published_at': getattr(entry, 'published', ''),
                            'relevance_score': self._calculate_relevance({
                                'title': entry.title, 
                                'description': getattr(entry, 'summary', '')
                            })
                        })
                        
            except Exception as e:
                print(f"❌ Error fetching RSS feed: {e}")
                
        return articles
    
    def _calculate_relevance(self, article: Dict) -> int:
        """Calculate relevance score based on keywords and symbols"""
        score = 0
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # High-priority market keywords
        high_keywords = [
            'earnings', 'guidance', 'merger', 'acquisition', 'ipo', 'fed', 
            'federal reserve', 'interest rate', 'inflation', 'gdp', 'jobs report'
        ]
        
        # Medium-priority keywords
        medium_keywords = [
            'stock', 'trading', 'analyst', 'upgrade', 'downgrade', 'premarket',
            'afterhours', 'dividend', 'split', 'buyback'
        ]
        
        for keyword in high_keywords:
            if keyword in text:
                score += 3
                
        for keyword in medium_keywords:
            if keyword in text:
                score += 2
                
        # Watchlist symbols (high value)
        for symbol in self.watchlist:
            if symbol.lower() in text or f"${symbol.lower()}" in text:
                score += 4
                
        return score
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_words = set(article['title'].lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
                if overlap > 0.7:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(article['title'].lower())
                
        return unique_articles
    
    async def generate_ai_summary(self, articles: List[Dict]) -> str:
        """Generate AI-powered market summary"""
        print("🤖 Generating AI market analysis...")
        
        if not articles:
            return "No significant pre-market news found for today's trading session."
            
        # Prepare articles for GPT
        news_text = "\n\n".join([
            f"**{article['title']}** ({article['source']})\n{article['description'][:200]}"
            for article in articles[:10]
        ])
        
        current_time = datetime.now(self.est)
        
        prompt = f"""
        As a professional financial analyst, provide a concise pre-market summary for {current_time.strftime('%A, %B %d, %Y')}. 
        
        Based on these overnight and pre-market news articles, analyze:
        
        1. **Key Market Catalysts**: What are the main stories that could move markets today?
        2. **Sector Focus**: Which sectors or individual stocks are in focus?
        3. **Economic Data**: Any important economic releases or Fed commentary?
        4. **Risk Factors**: What should traders watch out for today?
        5. **Trading Opportunities**: Brief mention of potential setups (bullish/bearish)
        
        Keep it concise (250-300 words), actionable, and professional. Use bullet points for key highlights.
        Include relevant stock symbols where appropriate (e.g., $AAPL, $SPY).
        
        News Articles:
        {news_text}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst providing pre-market briefings for active traders and investors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=450,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            print("✅ AI analysis generated successfully!")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating AI summary: {e}")
            logger.error(f"Error generating AI summary: {e}")
            return "Unable to generate AI analysis at this time. Please check the news headlines below for market updates."
    
    def send_discord_webhook(self, summary: str, articles: List[Dict]):
        """Send formatted message via Discord webhook with smart text splitting"""
        print("📤 Sending daily briefing to Discord...")
        
        current_time = datetime.now(self.est)
        
        # Smart splitting of AI summary to handle Discord's 1024 character field limit
        def split_summary(text: str, max_length: int = 1000) -> List[str]:
            """Split summary at natural breakpoints while respecting character limits"""
            if len(text) <= max_length:
                return [text]
            
            parts = []
            
            # Split by double newlines (paragraphs) first
            paragraphs = text.split('\n\n')
            current_part = ""
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed limit
                if len(current_part + paragraph + '\n\n') > max_length:
                    # If we have content, save it
                    if current_part.strip():
                        parts.append(current_part.strip())
                        current_part = ""
                    
                    # If single paragraph is too long, split by sentences
                    if len(paragraph) > max_length:
                        sentences = paragraph.split('. ')
                        temp_part = ""
                        
                        for sentence in sentences:
                            if len(temp_part + sentence + '. ') <= max_length:
                                temp_part += sentence + '. '
                            else:
                                if temp_part.strip():
                                    parts.append(temp_part.strip())
                                temp_part = sentence + '. '
                        
                        if temp_part.strip():
                            current_part = temp_part
                    else:
                        current_part = paragraph + '\n\n'
                else:
                    current_part += paragraph + '\n\n'
            
            # Add any remaining content
            if current_part.strip():
                parts.append(current_part.strip())
            
            return parts
        
        # Split the summary into manageable parts
        summary_parts = split_summary(summary)
        
        # Create fields for each part of the analysis
        analysis_fields = []
        for i, part in enumerate(summary_parts):
            # Only label the first section, leave others unlabeled for seamless flow
            field_name = "🤖 AI Market Analysis" if i == 0 else "\u200b"  # Zero-width space for continuation
            
            analysis_fields.append({
                "name": field_name,
                "value": part,
                "inline": False
            })
        
        # Create main embed with split analysis
        main_embed = {
            "title": "📈 Pre-Market News & Analysis",
            "description": f"*{current_time.strftime('%A, %B %d, %Y - %I:%M %p EST')}*",
            "color": 0x00ff00,  # Green
            "fields": analysis_fields,
            "footer": {
                "text": "Automated pre-market analysis • For informational purposes only"
            },
            "timestamp": current_time.isoformat()
        }
        
        # Create headlines embed
        if articles:
            headlines_text = "\n".join([
                f"• [{article['title'][:65]}...]({article['url']})"
                for article in articles[:6]
            ])
            
            headlines_embed = {
                "title": "📰 Key Headlines",
                "description": headlines_text,
                "color": 0x0099ff,  # Blue
            }
        else:
            headlines_embed = {
                "title": "📰 Headlines",
                "description": "No major headlines found for today's session.",
                "color": 0x999999,  # Gray
            }
        
        # Market hours reminder
        market_embed = {
            "title": "⏰ Trading Schedule",
            "description": "**Pre-Market:** 4:00 AM - 9:30 AM EST\n**Regular Hours:** 9:30 AM - 4:00 PM EST\n**After-Hours:** 4:00 PM - 8:00 PM EST",
            "color": 0xffaa00,  # Orange
        }
        
        # Prepare webhook payload
        payload = {
            "content": "🌅 **Good morning traders!** Here's your daily pre-market briefing:",
            "embeds": [main_embed, headlines_embed, market_embed],
            "username": "Pre-Market News Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        }
        
        try:
            # Debug: Print payload size info
            payload_size = len(json.dumps(payload))
            analysis_size = sum(len(field['value']) for field in analysis_fields)
            print(f"📊 Analysis split into {len(summary_parts)} parts ({analysis_size} total chars)")
            print(f"📦 Total payload size: {payload_size} characters")
            
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            print("✅ Daily briefing sent successfully to Discord!")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send Discord webhook: {e}")
            logger.error(f"Failed to send Discord webhook: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response status: {e.response.status_code}")
                print(f"   Response text: {e.response.text}")
            raise

async def main():
    """Main function to run the daily news bot"""
    print("🚀 DAILY PRE-MARKET NEWS BOT")
    print("=" * 40)
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    print(f"⏰ Starting at: {current_time.strftime('%A, %B %d, %Y - %I:%M %p EST')}")
    print("=" * 40)
    
    try:
        # Initialize bot
        bot = DailyNewsBot()
        
        # Test OpenAI connectivity
        if not bot.test_openai():
            print("❌ OpenAI connection failed - aborting")
            return
            
        print("\n📰 GENERATING DAILY BRIEFING")
        print("=" * 40)
        
        # Fetch news articles
        logger.info("Fetching news articles...")
        articles = await bot.fetch_news_articles(hours_back=16)
        
        # Generate AI summary
        logger.info("Generating AI summary...")
        summary = await bot.generate_ai_summary(articles)
        
        # Send to Discord
        logger.info("Sending to Discord...")
        bot.send_discord_webhook(summary, articles)
        
        print("\n🎉 SUCCESS!")
        print("=" * 20)
        logger.info("Daily news summary completed successfully!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {str(e)}")
        logger.error(f"Error in main execution: {e}")
        
        # Try to send error notification to Discord
        try:
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if webhook_url:
                error_payload = {
                    "content": f"❌ **Error in daily news bot:** {str(e)[:200]}",
                    "username": "Pre-Market News Bot - ERROR"
                }
                requests.post(webhook_url, json=error_payload)
        except:
            pass
        raise

if __name__ == "__main__":
    asyncio.run(main())