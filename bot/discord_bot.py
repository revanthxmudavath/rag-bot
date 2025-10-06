import asyncio
import aiohttp
import discord
from discord.ext import commands
import logging
import time
import json
from typing import Dict, Any, Optional
import os
from datetime import datetime

from app.config import get_settings

# Configure logging for Discord bot
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# PM Accelerator AI Community branding colors
COMMUNITY_COLORS = {
    'primary': 0x6366f1,    # Indigo for PM Accelerator brand
    'secondary': 0x3b82f6,  # Blue for info
    'warning': 0xf59e0b,    # Amber for warnings
    'error': 0xef4444,      # Red for errors
    'success': 0x10b981,    # Emerald for success
    'ai_purple': 0x8b5cf6   # Purple for AI theme
}

class PMAcceleratorBot(commands.Bot):

    def __init__(self):
        # Configure bot intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.members = True

        super().__init__(
            command_prefix='!',
            intents=intents,
            description='🤖 AI Community Assistant by PM Accelerator - Your intelligent community companion!',
            case_insensitive=True
        )

        # Bot analytics and tracking
        self.start_time = time.time()
        self.question_count = 0
        self.feedback_count = 0
        self.user_interactions = {}

        # FastAPI backend connection
        self.api_base_url = f"http://localhost:{settings.port}"
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info("🤖 PM Accelerator AI Community Bot initialized")

    async def setup_hook(self):
        """Setup async resources when bot starts."""
        self.session = aiohttp.ClientSession()
        logger.info("✅ HTTP session created for FastAPI communication")

    async def close(self):
        """Clean up resources when bot shuts down."""
        if self.session:
            await self.session.close()
        await super().close()
        logger.info("🔽 PM Accelerator AI Community Bot shut down")

    async def on_ready(self):
        """Event triggered when bot is ready and connected to Discord."""
        logger.info(f"🚀 {self.user.name} is online and ready to help the AI Community!")
        logger.info(f"📊 Connected to {len(self.guilds)} server(s)")

        # Set bot status
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="AI Community by PM Accelerator 🤖"
        )
        await self.change_presence(activity=activity, status=discord.Status.online)

        # Log community presence
        for guild in self.guilds:
            logger.info(f"📍 Active in: {guild.name} (ID: {guild.id})")

    async def on_message(self, message):
        """Handle incoming messages for AI/ML keyword responses."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Track user interactions
        user_id = str(message.author.id)
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {
                'message_count': 0,
                'first_interaction': datetime.now().isoformat(),
                'last_interaction': datetime.now().isoformat()
            }

        self.user_interactions[user_id]['message_count'] += 1
        self.user_interactions[user_id]['last_interaction'] = datetime.now().isoformat()

        # Auto-respond to AI/ML/community keywords
        content = message.content.lower()
        ai_keywords = [
            'machine learning', 'ml', 'deep learning', 'neural network', 'ai',
            'artificial intelligence', 'python', 'pytorch', 'tensorflow',
            'community', 'pm accelerator', 'project', 'collaboration', 'help'
        ]

        if any(keyword in content for keyword in ai_keywords) and not content.startswith('!'):
            # Only respond sometimes to avoid spam
            if self.user_interactions[user_id]['message_count'] % 5 == 1:
                embed = discord.Embed(
                    title="🔍 I noticed you're discussing AI/ML!",
                    description="I'm here to help with your AI Community questions! "
                               "Try using `!ask <your question>` for detailed assistance.",
                    color=COMMUNITY_COLORS['secondary']
                )
                embed.add_field(
                    name="Quick Commands",
                    value="`!community` - Overview\n`!ask` - Ask questions\n`!resources` - Learning materials",
                    inline=True
                )
                embed.set_footer(text="AI Community Project by PM Accelerator")
                await message.channel.send(embed=embed)

        # Process commands normally
        await self.process_commands(message)

    async def on_reaction_add(self, reaction, user):
        """Handle feedback reactions on bot responses."""
        # Ignore reactions from the bot itself
        if user == self.user:
            return

        # Check if reaction is on a bot message
        if reaction.message.author == self.user:
            emoji = str(reaction.emoji)

            # Track feedback
            if emoji in ['👍', '👎', '🤔']:
                self.feedback_count += 1
                feedback_type = {
                    '👍': 'helpful',
                    '👎': 'not_helpful',
                    '🤔': 'needs_clarification'
                }.get(emoji, 'unknown')

                # Send feedback to backend
                await self.track_bootcamp_feedback(
                    user_id=str(user.id),
                    message_id=str(reaction.message.id),
                    feedback_type=feedback_type,
                    emoji=emoji
                )

                logger.info(f"📊 Feedback received: {feedback_type} from {user.name}")

    async def on_mention(self, message):
        """Respond to @mentions with context-aware assistance."""
        if self.user.mentioned_in(message) and message.author != self.user:
            embed = discord.Embed(
                title="👋 Hello! I'm your AI Bootcamp Assistant",
                description="I'm here to help with your AI Engineering journey! "
                           "Ask me anything about the bootcamp, AI/ML concepts, or your projects."
            )
            embed.add_field(
                name="🎯 Popular Commands",
                value="`!ask <question>` - Get detailed answers\n"
                      "`!bootcamp` - Bootcamp overview\n"
                      "`!schedule` - Timeline and phases\n"
                      "`!projects` - Available projects",
                inline=True
            )
            embed.add_field(
                name="📚 Learning Resources",
                value="`!resources` - AI/ML materials\n"
                      "`!faq` - Common questions\n"
                      "`!status` - Bot health check",
                inline=True
            )
            embed.set_footer(text="AI Engineering Bootcamp | Powered by RAG")
            await message.channel.send(embed=embed)

    async def call_rag_api(self, query: str, user_id: str, channel_id: str) -> Dict[str, Any]:
        """Call the FastAPI RAG endpoint for question answering."""
        try:
            payload = {
                "query": f"[AI Community Question] {query}",
                "user_id": user_id,
                "channel_id": channel_id,
                "max_chunks": 5,
                "include_sources": True,
                "temperature": 0.7
            }

            async with self.session.post(
                f"{self.api_base_url}/api/rag-query",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"RAG API error {response.status}: {error_text}")
                    return {"error": f"API returned status {response.status}"}

        except asyncio.TimeoutError:
            logger.error("RAG API timeout")
            return {"error": "Request timed out"}
        except Exception as e:
            logger.error(f"RAG API error: {e}")
            return {"error": str(e)}

    async def track_community_question(self, question: str, user_id: str, response_data: Dict[str, Any]):
        """Track community question analytics."""
        try:
            # This would send analytics to backend in a real implementation
            analytics_data = {
                "question": question,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "response_length": len(response_data.get("answer", "")),
                "sources_count": len(response_data.get("sources", [])),
                "question_type": self._classify_question_type(question)
            }
            logger.info(f"📈 Question tracked: {analytics_data['question_type']}")
        except Exception as e:
            logger.error(f"Analytics tracking error: {e}")

    async def track_community_feedback(self, user_id: str, message_id: str, feedback_type: str, emoji: str):
        """Track feedback for community responses."""
        try:
            # This would send to /api/feedback endpoint in a real implementation
            feedback_data = {
                "user_id": user_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "emoji": emoji,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"💭 Feedback tracked: {feedback_type}")
        except Exception as e:
            logger.error(f"Feedback tracking error: {e}")

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of community question for analytics."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['event', 'meeting', 'when', 'schedule']):
            return 'events'
        elif any(word in question_lower for word in ['project', 'collaboration', 'contribute']):
            return 'projects'
        elif any(word in question_lower for word in ['resource', 'material', 'link', 'tutorial']):
            return 'resources'
        elif any(word in question_lower for word in ['concept', 'explain', 'what is', 'how does']):
            return 'concepts'
        elif any(word in question_lower for word in ['help', 'stuck', 'error', 'problem']):
            return 'troubleshooting'
        elif any(word in question_lower for word in ['community', 'guideline', 'rule', 'how to']):
            return 'community_help'
        else:
            return 'general'

    def create_community_response_embed(self, response_data: Dict[str, Any], question: str, user: discord.User) -> discord.Embed:
        """Create a branded embed for RAG responses."""
        if "error" in response_data:
            # Error response
            embed = discord.Embed(
                title="⚠️ I encountered an issue",
                description="I'm having trouble accessing my knowledge base right now. "
                           "This might be due to connectivity issues or I might need more context.",
                color=COMMUNITY_COLORS['warning']
            )
            embed.add_field(
                name="What you can try:",
                value="• Rephrase your question with more details\n"
                      "• Ask about a specific AI/ML topic\n"
                      "• Try again in a few moments\n"
                      "• Use `!community` for community overview",
                inline=False
            )
            embed.set_footer(text=f"Asked by {user.display_name} • AI Community Project by PM Accelerator")
            return embed

        # Successful response
        answer = response_data.get("answer", "I don't have a specific answer for that question.")
        sources = response_data.get("sources", [])
        metadata = response_data.get("metadata", {})

        # Truncate very long answers
        if len(answer) > 1500:
            answer = answer[:1500] + "...\n\n*Answer truncated for readability*"

        embed = discord.Embed(
            title="🤖 AI Community Assistant Response",
            description=answer,
            color=COMMUNITY_COLORS['primary']
        )

        # Add question context
        question_preview = question[:100] + "..." if len(question) > 100 else question
        embed.add_field(
            name="📝 Your Question",
            value=f"*{question_preview}*",
            inline=False
        )

        # Add sources if available
        if sources:
            sources_text = ""
            for i, source in enumerate(sources[:3], 1):  # Show max 3 sources
                title = source.get("document_title", "Knowledge Base")
                score = source.get("similarity_score", 0)
                sources_text += f"{i}. **{title}** (relevance: {score:.2f})\n"

            embed.add_field(
                name="📚 Sources",
                value=sources_text,
                inline=True
            )

        # Add metadata if available
        if metadata:
            processing_time = metadata.get("processing_time_ms", 0)
            chunks_used = metadata.get("chunks_retrieved", 0)

            embed.add_field(
                name="⚡ Processing Info",
                value=f"Time: {processing_time:.0f}ms\nKnowledge chunks: {chunks_used}",
                inline=True
            )

        # Add helpful suggestions
        embed.add_field(
            name="💡 Need more help?",
            value="React with 👍 if helpful, 👎 if not, or 🤔 if you need clarification!",
            inline=False
        )

        embed.set_footer(
            text=f"Asked by {user.display_name} • AI Community Project by PM Accelerator | Powered by RAG",
            icon_url=user.avatar.url if user.avatar else None
        )
        embed.timestamp = datetime.now()

        return embed

    async def setup_commands(self):
        """Load all command extensions."""
        try:
            # Import and setup the commands cog
            from bot.commands import CommunityCommands
            await self.add_cog(CommunityCommands(self))
            logger.info("✅ Community commands loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load commands: {e}")

# Initialize the bot instance
bot = PMAcceleratorBot()

# Setup and run the bot
async def run_bot():
    """Setup and run the Discord bot."""
    try:
        # Setup commands
        await bot.setup_commands()

        # Start the bot
        token = settings.discord_bot_token
        if not token or token == "test_token_placeholder":
            logger.warning("⚠️ Discord bot token not set - running in mock mode")
            logger.info("🔧 Set DISCORD_BOT_TOKEN environment variable to run bot")
            return

        logger.info("🚀 Starting PM Accelerator AI Community Discord Bot...")
        await bot.start(token)

    except Exception as e:
        logger.error(f"❌ Bot startup failed: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    # Run the bot
    asyncio.run(run_bot())