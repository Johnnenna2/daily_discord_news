
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
        print("\n🔧 INITIALIZING BOT - DEBUG MODE")
        print("=" * 50)
        
        # Debug environment variables
        print("📋 Environment Variables Check:")
        openai_key = os.getenv('OPENAI_API_KEY')
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        news_key = os.getenv('NEWS_API_KEY')
        
        print(f"  OPENAI_API_KEY: {'✅ SET (' + openai_key[:20] + '...)' if openai_key else '❌ MISSING'}")
        print(f"  DISCORD_WEBHOOK_URL: {'✅ SET (' + webhook_url[:50] + '...)' if webhook_url else '❌ MISSING'}")
        print(f"  NEWS_API_KEY: {'✅ SET' if news_key else '❌ MISSING (optional)'}")
        
        # Load dotenv for local development
        try:
            from dotenv import load_dotenv
            if os.path.exists('.env'):
                load_dotenv()
                print("📁 Loaded .env file for local development")
            else:
                print("🔧 Using system environment variables (GitHub Actions)")
        except ImportError:
            print("🔧 python-dotenv not available, using system environment variables")
        
        # Initialize APIs
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
        
    def test_webhook(self):
        """Test Discord webhook connectivity"""
        print("\n🧪 TESTING DISCORD WEBHOOK")
        print("-" * 30)
        
        if not self.webhook_url:
            print("❌ No webhook URL available")
            return False
            
        payload = {
            "content": f"🧪 **Debug Test from GitHub Actions**\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "username": "Debug Bot"
        }
        
        try:
            print(f"📤 Sending test message to webhook...")
            print(f"   Webhook URL: {self.webhook_url[:50]}...")
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            print("✅ Test webhook sent successfully!")
            print(f"   Response status: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Webhook test failed: {e}")
            print(f"   Response status: {getattr(e.response, 'status_code', 'No response')}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response text: {e.response.text}")
            return False
    
    def test_openai(self):
        """Test OpenAI API connectivity"""
        print("\n🧪 TESTING OPENAI API")
        print("-" * 30)
        
        try:
            print("📤 Sending test request to OpenAI...")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say 'OpenAI test successful' in exactly those words."}
                ],
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            print("✅ OpenAI test successful!")
            print(f"   Response: {result}")
            return True
            
        except Exception as e:
            print(f"❌ OpenAI test failed: {e}")
            return False

    async def fetch_news_articles(self, hours_back: int = 16) -> List[Dict]:
        """Fetch recent financial news articles"""
        print(f"\n📰 FETCHING NEWS ARTICLES (last {hours_back} hours)")
        print("-" * 30)
        
        articles = []
        from_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # Try News API first if available
            if self.news_api_key:
                print("📡 Fetching from News API...")
                articles.extend(await self._fetch_from_newsapi(from_time))
            else:
                print("⚠️  No News API key, skipping News API")
            
            # Always fetch from RSS feeds as primary/backup source
            print("📡 Fetching from RSS feeds...")
            rss_articles = await self._fetch_from_rss()
            articles.extend(rss_articles)
            
            print(f"📊 Total articles fetched: {len(articles)}")
            
            # Remove duplicates and sort by relevance
            unique_articles = self._deduplicate_articles(articles)
            print(f"📊 Unique articles after deduplication: {len(unique_articles)}")
            
            sorted_articles = sorted(unique_articles, key=lambda x: x.get('relevance_score', 0), reverse=True)[:12]
            print(f"📊 Top articles selected: {len(sorted_articles)}")
            
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
                        print(f"   📈 News API returned {len(news_articles)} articles")
                        
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
                        print(f"   ⚠️  News API returned status {response.status}")
                        
            except Exception as e:
                print(f"   ❌ Error fetching from News API: {e}")
        
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
                print(f"   📡 Fetching from {feed_url.split('//')[1].split('/')[0]}...")
                feed = feedparser.parse(feed_url)
                feed_articles = 0
                
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
                        feed_articles += 1
                        
                print(f"      ✅ Got {feed_articles} articles")
                        
            except Exception as e:
                print(f"      ❌ Error fetching RSS feed: {e}")
                
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
                # If 70% of words overlap, consider it a duplicate
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
        print(f"\n🤖 GENERATING AI SUMMARY")
        print("-" * 30)
        
        if not articles:
            summary = "No significant pre-market news found for today's trading session."
            print("⚠️  No articles to summarize")
            return summary
            
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
            print("📤 Sending request to OpenAI...")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Changed from gpt-4 to avoid quota issues
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst providing pre-market briefings for active traders and investors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=450,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            print("✅ AI summary generated successfully!")
            print(f"   Summary length: {len(summary)} characters")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating AI summary: {e}")
            logger.error(f"Error generating AI summary: {e}")
            return "Unable to generate AI analysis at this time. Please check the news headlines below for market updates."
    
    def send_discord_webhook(self, summary: str, articles: List[Dict]):
        """Send formatted message via Discord webhook"""
        print(f"\n📤 SENDING TO DISCORD")
        print("-" * 30)
        
        current_time = datetime.now(self.est)
        
        # Create main embed with AI summary
        main_embed = {
            "title": "📈 Pre-Market News & Analysis",
            "description": f"*{current_time.strftime('%A, %B %d, %Y - %I:%M %p EST')}*",
            "color": 0x00ff00,  # Green
            "fields": [
                {
                    "name": "🤖 AI Market Analysis",
                    "value": summary[:1000],  # Discord field limit
                    "inline": False
                }
            ],
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
            print("📤 Sending Discord webhook...")
            print(f"   Payload size: {len(json.dumps(payload))} characters")
            
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            print("✅ Successfully sent daily news summary to Discord")
            print(f"   Response status: {response.status_code}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send Discord webhook: {e}")
            logger.error(f"Failed to send Discord webhook: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response status: {e.response.status_code}")
                print(f"   Response text: {e.response.text}")
            raise

async def main():
    """Main function to run the daily news bot"""
    print("🚀 STARTING DAILY PRE-MARKET NEWS BOT")
    print("=" * 60)
    print(f"⏰ Current time: {datetime.now()}")
    print(f"🌍 Current UTC time: {datetime.utcnow()}")
    print(f"🇺🇸 Current EST time: {datetime.now(pytz.timezone('US/Eastern'))}")
    print("=" * 60)
    
    try:
        # Initialize bot
        bot = DailyNewsBot()
        
        # Run connectivity tests
        print("\n🧪 RUNNING CONNECTIVITY TESTS")
        print("=" * 40)
        
        webhook_test = bot.test_webhook()
        openai_test = bot.test_openai()
        
        if not webhook_test:
            print("❌ Webhook test failed - check your Discord webhook URL")
            
        if not openai_test:
            print("❌ OpenAI test failed - check your API key and billing")
        
        if not webhook_test or not openai_test:
            print("\n⚠️  Some tests failed, but continuing with news generation...")
            
        # Fetch news articles
        print(f"\n📰 NEWS GENERATION PROCESS")
        print("=" * 40)
        
        logger.info("Fetching news articles...")
        articles = await bot.fetch_news_articles(hours_back=16)
        logger.info(f"Found {len(articles)} relevant articles")
        
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
        print(f"\n💥 CRITICAL ERROR")
        print("=" * 20)
        print(f"❌ Error: {str(e)}")
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