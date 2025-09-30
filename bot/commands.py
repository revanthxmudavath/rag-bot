import discord
from discord.ext import commands
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Import the bot colors from discord_bot.py
COMMUNITY_COLORS = {
    'primary': 0x6366f1,    # Indigo for PM Accelerator brand
    'secondary': 0x3b82f6,  # Blue for info
    'warning': 0xf59e0b,    # Amber for warnings
    'error': 0xef4444,      # Red for errors
    'success': 0x10b981,    # Emerald for success
    'ai_purple': 0x8b5cf6   # Purple for AI theme
}

class CommunityCommands(commands.Cog):
    """PM Accelerator AI Community specific commands."""

    def __init__(self, bot):
        self.bot = bot
        logger.info("🎯 Community Commands loaded")

    @commands.command(name="ask")
    async def ask_community_question(self, ctx, *, question: str):
        """
        Ask any AI/ML question using RAG pipeline from community knowledge.

        Usage: !ask What is machine learning?
        """
        if not question.strip():
            embed = discord.Embed(
                title="❓ Please provide a question",
                description="Usage: `!ask <your question>`\n\nExample: `!ask What is deep learning?`",
                color=COMMUNITY_COLORS['warning']
            )
            await ctx.reply(embed=embed)
            return

        # Show typing indicator
        async with ctx.typing():
            # Track the question
            self.bot.question_count += 1

            try:
                # Call RAG API through the bot's method
                response_data = await self.bot.call_rag_api(
                    query=question,
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id)
                )

                # Create AI community branded embed
                embed = self.bot.create_community_response_embed(
                    response_data=response_data,
                    question=question,
                    user=ctx.author
                )

                # Send the response
                bot_message = await ctx.reply(embed=embed)

                # Add community feedback reactions
                await bot_message.add_reaction("👍")  # Helpful
                await bot_message.add_reaction("👎")  # Not helpful
                await bot_message.add_reaction("🤔")  # Need more details

                # Track community question analytics
                await self.bot.track_community_question(question, str(ctx.author.id), response_data)

                logger.info(f"📚 Community question answered for {ctx.author.name}: {question[:50]}...")

            except Exception as e:
                logger.error(f"Error in ask command: {e}")
                error_embed = discord.Embed(
                    title="🔧 Technical Issue",
                    description="I'm experiencing technical difficulties. Please try again in a moment.",
                    color=COMMUNITY_COLORS['error']
                )
                error_embed.add_field(
                    name="What you can try:",
                    value="• Check if your question is clear and specific\n"
                          "• Try rephrasing your question\n"
                          "• Use `!community` for community overview",
                    inline=False
                )
                await ctx.reply(embed=error_embed)

    @commands.command(name="community")
    async def community_overview(self, ctx):
        """Show AI Engineering Bootcamp overview and help."""
        embed = discord.Embed(
            title="🤖 AI Engineering Bootcamp Assistant",
            description="Welcome to your AI Engineering journey! I'm here to help you succeed in the bootcamp.",
            color=COMMUNITY_COLORS['primary']
        )

        embed.add_field(
            name="🎯 Main Commands",
            value="`!ask <question>` - Ask any bootcamp question\n"
                  "`!schedule` - View bootcamp timeline\n"
                  "`!projects` - See available projects\n"
                  "`!resources` - Access learning materials",
            inline=True
        )

        embed.add_field(
            name="🔧 Utility Commands",
            value="`!faq` - Common questions\n"
                  "`!status` - Bot health check\n"
                  "`!help` - Command list",
            inline=True
        )

        embed.add_field(
            name="🚀 What I Can Help With",
            value="• AI/ML concepts and theory\n"
                  "• Python programming for AI\n"
                  "• Project guidance and troubleshooting\n"
                  "• Bootcamp schedule and deadlines\n"
                  "• Learning resources and materials\n"
                  "• Career advice in AI/ML",
            inline=False
        )

        embed.add_field(
            name="💡 Pro Tips",
            value="• Be specific in your questions for better answers\n"
                  "• Use `!ask` for detailed technical questions\n"
                  "• React with 👍👎🤔 to give feedback on answers\n"
                  "• Tag me (@bot) for quick assistance",
            inline=False
        )

        embed.set_footer(text="AI Community Project by PM Accelerator | Powered by RAG")
        embed.set_thumbnail(url="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg")

        await ctx.reply(embed=embed)

    @commands.command(name="info")
    async def community_info(self, ctx):
        """Display PM Accelerator AI Community information."""
        embed = discord.Embed(
            title="📊 AI Community Project Info",
            description="Learn more about the PM Accelerator AI Community",
            color=COMMUNITY_COLORS['secondary']
        )

        embed.add_field(
            name="🏢 About PM Accelerator",
            value="A community of AI enthusiasts, practitioners, and learners\n"
                  "working together to advance AI/ML knowledge and skills.",
            inline=False
        )

        embed.add_field(
            name="🎯 Community Goals",
            value="• Share AI/ML knowledge and resources\n"
                  "• Collaborate on AI projects\n"
                  "• Support learning and career growth\n"
                  "• Stay updated with latest AI trends",
            inline=True
        )

        embed.add_field(
            name="🔗 Get Involved",
            value="• Ask questions using `!ask`\n"
                  "• Share resources and insights\n"
                  "• Participate in discussions\n"
                  "• Join community projects",
            inline=True
        )

        embed.set_footer(text="AI Community Project by PM Accelerator")
        await ctx.reply(embed=embed)

    @commands.command(name="projects")
    async def community_projects(self, ctx):
        """List available AI/ML community projects."""
        embed = discord.Embed(
            title="🛠️ AI Community Projects",
            description="Collaborative AI/ML projects in our community",
            color=COMMUNITY_COLORS['ai_purple']
        )

        projects = [
            {
                "name": "🤖 Community RAG Bot",
                "type": "NLP/LLM",
                "description": "Discord bot for answering community questions using RAG",
                "tech": "Python, FastAPI, Discord.py, Vector DB"
            },
            {
                "name": "📈 AI Learning Tracker",
                "type": "Data Science",
                "description": "Track and visualize community learning progress",
                "tech": "Pandas, Plotly, Streamlit"
            },
            {
                "name": "📚 Resource Recommender",
                "type": "ML/RecSys",
                "description": "Recommend learning resources based on user interests",
                "tech": "Scikit-learn, Content-based filtering"
            },
            {
                "name": "🌐 AI News Aggregator",
                "type": "NLP/Web Scraping",
                "description": "Collect and summarize latest AI news and papers",
                "tech": "BeautifulSoup, NLTK, APIs"
            }
        ]

        for project in projects:
            embed.add_field(
                name=f"{project['name']}",
                value=f"**Type:** {project['type']}\n"
                      f"**Description:** {project['description']}\n"
                      f"**Tech Stack:** {project['tech']}",
                inline=False
            )

        embed.add_field(
            name="📋 How to Contribute",
            value="1. Choose a project that interests you\n"
                  "2. Join the project discussion channel\n"
                  "3. Connect with project maintainers\n"
                  "4. Ask `!ask` for technical guidance!",
            inline=False
        )

        embed.set_footer(text="AI Community Project by PM Accelerator")
        await ctx.reply(embed=embed)

    @commands.command(name="resources")
    async def bootcamp_resources(self, ctx):
        """Share AI/ML learning resources and documentation links."""
        embed = discord.Embed(
            title="📚 AI/ML Learning Resources",
            description="Curated resources for your AI Engineering journey",
            color=COMMUNITY_COLORS['secondary']
        )

        embed.add_field(
            name="📖 Essential Books",
            value="• *Hands-On Machine Learning* by Aurélien Géron\n"
                  "• *Deep Learning* by Ian Goodfellow\n"
                  "• *Pattern Recognition and ML* by Christopher Bishop\n"
                  "• *The Elements of Statistical Learning* by Hastie",
            inline=True
        )

        embed.add_field(
            name="🎓 Online Courses",
            value="• [Andrew Ng's ML Course](https://coursera.org/learn/machine-learning)\n"
                  "• [Fast.ai Practical Deep Learning](https://fast.ai)\n"
                  "• [CS229 Stanford ML](https://cs229.stanford.edu)\n"
                  "• [MIT 6.034 Artificial Intelligence](https://ocw.mit.edu)",
            inline=True
        )

        embed.add_field(
            name="🛠️ Tools & Frameworks",
            value="**Python Libraries:**\n"
                  "• NumPy, Pandas, Matplotlib\n"
                  "• Scikit-learn, XGBoost\n"
                  "• PyTorch, TensorFlow, Keras\n"
                  "• Hugging Face Transformers",
            inline=True
        )

        embed.add_field(
            name="🧮 Math & Statistics",
            value="• [Khan Academy Linear Algebra](https://khanacademy.org)\n"
                  "• [3Blue1Brown Essence of LA](https://youtube.com/3blue1brown)\n"
                  "• [StatQuest Statistics](https://youtube.com/statquest)\n"
                  "• [Probability Theory Textbook](https://example.com)",
            inline=True
        )

        embed.add_field(
            name="💻 Practice Platforms",
            value="• [Kaggle Competitions](https://kaggle.com)\n"
                  "• [Google Colab](https://colab.research.google.com)\n"
                  "• [Papers With Code](https://paperswithcode.com)\n"
                  "• [Towards Data Science](https://towardsdatascience.com)",
            inline=True
        )

        embed.add_field(
            name="🔬 Research & Papers",
            value="• [arXiv.org](https://arxiv.org) - Latest research papers\n"
                  "• [Google Scholar](https://scholar.google.com)\n"
                  "• [Distill.pub](https://distill.pub) - Visual explanations\n"
                  "• [OpenAI Blog](https://openai.com/blog)",
            inline=True
        )

        embed.add_field(
            name="🎯 Quick Tips",
            value="• Start with fundamentals before advanced topics\n"
                  "• Practice coding daily with real datasets\n"
                  "• Join AI/ML communities and forums\n"
                  "• Build projects to showcase your skills",
            inline=False
        )

        embed.set_footer(text="AI Engineering Bootcamp | Keep learning!")
        await ctx.reply(embed=embed)


    @commands.command(name="status")
    async def community_status(self, ctx):
        """Show bot health and knowledge base status."""
        # Calculate uptime
        uptime_seconds = time.time() - self.bot.start_time
        uptime = str(timedelta(seconds=int(uptime_seconds)))

        # Get bot stats
        guild_count = len(self.bot.guilds)
        user_count = len(set(user.id for guild in self.bot.guilds for user in guild.members))

        embed = discord.Embed(
            title="🔍 AI Community Bot Status",
            description="Current health and performance metrics",
            color=COMMUNITY_COLORS['primary']
        )

        embed.add_field(
            name="🤖 Bot Health",
            value=f"**Status:** 🟢 Online\n"
                  f"**Uptime:** {uptime}\n"
                  f"**Latency:** {round(self.bot.latency * 1000)}ms",
            inline=True
        )

        embed.add_field(
            name="📊 Usage Stats",
            value=f"**Servers:** {guild_count}\n"
                  f"**Users:** {user_count}\n"
                  f"**Questions:** {self.bot.question_count}\n"
                  f"**Feedback:** {self.bot.feedback_count}",
            inline=True
        )

        # Try to get backend health
        try:
            response_data = await self.bot.call_rag_api(
                query="health check",
                user_id=str(ctx.author.id),
                channel_id=str(ctx.channel.id)
            )

            if "error" in response_data:
                backend_status = "🟡 Limited (Mock Mode)"
                backend_info = "Using fallback responses"
            else:
                backend_status = "🟢 Connected"
                backend_info = "RAG pipeline operational"
        except:
            backend_status = "🟡 Limited (Mock Mode)"
            backend_info = "Backend unavailable"

        embed.add_field(
            name="🧠 Knowledge Base",
            value=f"**Backend:** {backend_status}\n"
                  f"**Info:** {backend_info}",
            inline=True
        )

        embed.add_field(
            name="💡 Quick Actions",
            value="`!ask` - Test question answering\n"
                  "`!community` - Bot overview\n"
                  "`!help` - All commands",
            inline=False
        )

        embed.set_footer(text="AI Community Project by PM Accelerator | Always here to help!")
        embed.timestamp = datetime.now()

        await ctx.reply(embed=embed)

# Add the cog to the bot
async def setup(bot):
    await bot.add_cog(CommunityCommands(bot))