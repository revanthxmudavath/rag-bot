import discord
from discord.ext import commands
import re
import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

# PM Accelerator Community branding colors
COMMUNITY_COLORS = {
    'primary': 0x6366f1,    # Indigo for PM Accelerator brand
    'secondary': 0x3b82f6,  # Blue for info
    'warning': 0xf59e0b,    # Amber for warnings
    'error': 0xef4444,      # Red for errors
    'success': 0x10b981,    # Emerald for success
    'ai_purple': 0x8b5cf6   # Purple for AI theme
}

class CodeBlockView(discord.ui.View):
    """Interactive view for code blocks with syntax highlighting."""

    def __init__(self, code: str, language: str = "python"):
        super().__init__(timeout=300)  # 5 minutes timeout
        self.code = code
        self.language = language

    @discord.ui.button(label="📋 Copy Code", style=discord.ButtonStyle.secondary)
    async def copy_code(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Button to help users copy code snippets."""
        embed = discord.Embed(
            title="📋 Code Copied!",
            description="```" + self.language + "\n" + self.code + "\n```",
            color=COMMUNITY_COLORS['secondary']
        )
        embed.add_field(
            name="💡 Pro Tip",
            value="You can copy this code and paste it into your IDE or Jupyter notebook!",
            inline=False
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(label="❓ Explain Code", style=discord.ButtonStyle.primary)
    async def explain_code(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Button to get code explanation."""
        await interaction.response.defer(ephemeral=True)

        # This would call the RAG API for code explanation
        explanation = self._generate_code_explanation(self.code, self.language)

        embed = discord.Embed(
            title="🔍 Code Explanation",
            description=explanation,
            color=COMMUNITY_COLORS['purple']
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    def _generate_code_explanation(self, code: str, language: str) -> str:
        """Generate a basic code explanation (would use RAG in full implementation)."""
        if "import" in code:
            return f"This {language} code imports necessary libraries and implements core functionality. " \
                   "The imports at the top bring in external modules needed for the implementation."
        elif "def " in code:
            return f"This {language} code defines a function. Functions are reusable blocks of code " \
                   "that perform specific tasks and can accept parameters and return values."
        elif "class " in code:
            return f"This {language} code defines a class. Classes are blueprints for creating objects " \
                   "and are fundamental to object-oriented programming."
        else:
            return f"This {language} code snippet demonstrates programming concepts. " \
                   "Each line executes specific operations to achieve the desired functionality."

class AITopicView(discord.ui.View):
    """Interactive view for AI/ML topic exploration."""

    def __init__(self, topic: str):
        super().__init__(timeout=300)
        self.topic = topic

    @discord.ui.button(label="📚 Learn More", style=discord.ButtonStyle.primary)
    async def learn_more(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Provide additional learning resources for the topic."""
        resources = self._get_topic_resources(self.topic)

        embed = discord.Embed(
            title=f"📚 Learn More: {self.topic}",
            description=f"Additional resources to master {self.topic}",
            color=COMMUNITY_COLORS['secondary']
        )

        for resource_type, resource_list in resources.items():
            embed.add_field(
                name=resource_type,
                value="\n".join(resource_list),
                inline=False
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(label="🛠️ Practice", style=discord.ButtonStyle.success)
    async def practice_exercises(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Suggest practice exercises for the topic."""
        exercises = self._get_practice_exercises(self.topic)

        embed = discord.Embed(
            title=f"🛠️ Practice: {self.topic}",
            description=f"Hands-on exercises to practice {self.topic}",
            color=COMMUNITY_COLORS['primary']
        )

        for i, exercise in enumerate(exercises, 1):
            embed.add_field(
                name=f"Exercise {i}: {exercise['title']}",
                value=f"**Difficulty:** {exercise['difficulty']}\n"
                      f"**Description:** {exercise['description']}\n"
                      f"**Goal:** {exercise['goal']}",
                inline=False
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    def _get_topic_resources(self, topic: str) -> Dict[str, List[str]]:
        """Get learning resources for a specific AI/ML topic."""
        topic_lower = topic.lower()

        if "machine learning" in topic_lower or "ml" in topic_lower:
            return {
                "📖 Essential Reading": [
                    "• Hands-On Machine Learning by Aurélien Géron",
                    "• The Elements of Statistical Learning",
                    "• Pattern Recognition and Machine Learning"
                ],
                "🎥 Video Tutorials": [
                    "• Andrew Ng's Machine Learning Course",
                    "• 3Blue1Brown Neural Networks Series",
                    "• StatQuest Machine Learning Playlist"
                ],
                "💻 Interactive Learning": [
                    "• Kaggle Learn ML Course",
                    "• Google's Machine Learning Crash Course",
                    "• Fast.ai Machine Learning Course"
                ]
            }
        elif "deep learning" in topic_lower or "neural" in topic_lower:
            return {
                "📖 Essential Reading": [
                    "• Deep Learning by Ian Goodfellow",
                    "• Neural Networks and Deep Learning",
                    "• Deep Learning with Python by François Chollet"
                ],
                "🎥 Video Tutorials": [
                    "• CS231n Stanford Convolutional Neural Networks",
                    "• Fast.ai Deep Learning for Coders",
                    "• DeepLearning.ai Specialization"
                ],
                "💻 Frameworks": [
                    "• PyTorch Tutorials",
                    "• TensorFlow/Keras Documentation",
                    "• Hugging Face Transformers"
                ]
            }
        else:
            return {
                "📖 General Resources": [
                    "• MIT OpenCourseWare AI Materials",
                    "• Stanford CS229 Machine Learning",
                    "• Berkeley CS188 Artificial Intelligence"
                ],
                "🎥 Video Content": [
                    "• YouTube: Two Minute Papers",
                    "• YouTube: Yannic Kilcher",
                    "• YouTube: AI Explained"
                ]
            }

    def _get_practice_exercises(self, topic: str) -> List[Dict[str, str]]:
        """Get practice exercises for a specific AI/ML topic."""
        topic_lower = topic.lower()

        if "machine learning" in topic_lower:
            return [
                {
                    "title": "Iris Classification",
                    "difficulty": "🟢 Beginner",
                    "description": "Build a classifier for the famous Iris dataset",
                    "goal": "Learn basic classification with scikit-learn"
                },
                {
                    "title": "House Price Prediction",
                    "difficulty": "🟡 Intermediate",
                    "description": "Predict house prices using regression techniques",
                    "goal": "Master feature engineering and model evaluation"
                },
                {
                    "title": "Customer Segmentation",
                    "difficulty": "🟠 Advanced",
                    "description": "Use clustering to segment customers by behavior",
                    "goal": "Understand unsupervised learning methods"
                }
            ]
        elif "deep learning" in topic_lower:
            return [
                {
                    "title": "MNIST Digit Recognition",
                    "difficulty": "🟢 Beginner",
                    "description": "Train a neural network to recognize handwritten digits",
                    "goal": "Learn basic neural network concepts"
                },
                {
                    "title": "CIFAR-10 Image Classification",
                    "difficulty": "🟡 Intermediate",
                    "description": "Build a CNN for color image classification",
                    "goal": "Master convolutional neural networks"
                },
                {
                    "title": "Text Sentiment Analysis",
                    "difficulty": "🟠 Advanced",
                    "description": "Use RNNs/Transformers for sentiment classification",
                    "goal": "Understand sequence modeling and NLP"
                }
            ]
        else:
            return [
                {
                    "title": "Data Exploration",
                    "difficulty": "🟢 Beginner",
                    "description": "Explore and visualize a dataset of your choice",
                    "goal": "Learn data analysis fundamentals"
                },
                {
                    "title": "Algorithm Implementation",
                    "difficulty": "🟡 Intermediate",
                    "description": "Implement a basic ML algorithm from scratch",
                    "goal": "Understand algorithm mechanics"
                }
            ]

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from text for syntax highlighting."""
    # Pattern to match code blocks with optional language specification
    pattern = r'```(\w+)?\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = []
    for language, code in matches:
        if not language:
            language = "text"
        code_blocks.append({
            "language": language,
            "code": code.strip()
        })

    return code_blocks

def format_ai_response(text: str) -> str:
    """Format AI responses with better readability for Discord."""
    # Replace common AI/ML terms with emoji versions
    replacements = {
        "machine learning": "🤖 machine learning",
        "deep learning": "🧠 deep learning",
        "neural network": "🔗 neural network",
        "artificial intelligence": "🤖 artificial intelligence",
        "AI": "🤖 AI",
        "python": "🐍 Python",
        "pytorch": "🔥 PyTorch",
        "tensorflow": "📊 TensorFlow",
        "scikit-learn": "⚙️ scikit-learn",
        "numpy": "🔢 NumPy",
        "pandas": "🐼 Pandas"
    }

    formatted_text = text
    for term, replacement in replacements.items():
        # Only replace if not already replaced and not in code blocks
        if term in formatted_text.lower() and replacement not in formatted_text:
            formatted_text = re.sub(
                rf'\b{re.escape(term)}\b',
                replacement,
                formatted_text,
                flags=re.IGNORECASE
            )

    return formatted_text

def create_progress_bar(current: int, total: int, length: int = 10) -> str:
    """Create a progress bar for bootcamp progress tracking."""
    if total == 0:
        return "🔘" * length

    filled = int(length * current / total)
    bar = "🟢" * filled + "⚪" * (length - filled)
    percentage = int(100 * current / total)

    return f"{bar} {percentage}%"

def get_ai_topic_emoji(topic: str) -> str:
    """Get appropriate emoji for AI/ML topics."""
    topic_lower = topic.lower()

    emoji_map = {
        "machine learning": "🤖",
        "deep learning": "🧠",
        "neural network": "🔗",
        "computer vision": "👁️",
        "natural language processing": "💬",
        "nlp": "💬",
        "reinforcement learning": "🎮",
        "python": "🐍",
        "pytorch": "🔥",
        "tensorflow": "📊",
        "data science": "📈",
        "statistics": "📊",
        "algorithm": "⚙️",
        "model": "🧮",
        "training": "🏋️",
        "dataset": "📋",
        "feature": "🎯",
        "prediction": "🔮",
        "classification": "🏷️",
        "regression": "📈",
        "clustering": "🔗"
    }

    for term, emoji in emoji_map.items():
        if term in topic_lower:
            return emoji

    return "🎯"  # Default emoji

class BootcampPaginator:
    """Paginator for long bootcamp content."""

    def __init__(self, content_list: List[str], title: str = "Content", items_per_page: int = 5):
        self.content_list = content_list
        self.title = title
        self.items_per_page = items_per_page
        self.total_pages = (len(content_list) + items_per_page - 1) // items_per_page

    def get_page(self, page_number: int) -> discord.Embed:
        """Get a specific page of content."""
        if page_number < 1:
            page_number = 1
        elif page_number > self.total_pages:
            page_number = self.total_pages

        start_idx = (page_number - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_content = self.content_list[start_idx:end_idx]

        embed = discord.Embed(
            title=f"{self.title} (Page {page_number}/{self.total_pages})",
            color=COMMUNITY_COLORS['primary']
        )

        for i, content in enumerate(page_content, start=start_idx + 1):
            embed.add_field(
                name=f"Item {i}",
                value=content,
                inline=False
            )

        embed.set_footer(text=f"AI Community Project by PM Accelerator | Page {page_number} of {self.total_pages}")

        return embed

async def send_typing_message(channel, messages: List[str], delay: float = 2.0):
    """Send a sequence of typing messages to simulate bot thinking."""
    for message in messages:
        embed = discord.Embed(
            description=f"🤔 {message}",
            color=COMMUNITY_COLORS['secondary']
        )
        msg = await channel.send(embed=embed)
        await asyncio.sleep(delay)
        await msg.delete()

def validate_community_question(question: str) -> Dict[str, Any]:
    """Validate and categorize community questions."""
    question = question.strip()

    if len(question) < 10:
        return {
            "valid": False,
            "error": "Question is too short. Please provide more details for a better answer.",
            "suggestion": "Try asking something like: 'How does gradient descent work in machine learning?'"
        }

    if len(question) > 500:
        return {
            "valid": False,
            "error": "Question is too long. Please break it into smaller, more specific questions.",
            "suggestion": "Focus on one concept at a time for clearer answers."
        }

    # Check for AI/ML relevance
    ai_keywords = [
        "machine learning", "deep learning", "neural network", "ai", "python",
        "algorithm", "model", "data", "training", "prediction", "classification",
        "regression", "community", "project", "collaboration"
    ]

    has_ai_keyword = any(keyword in question.lower() for keyword in ai_keywords)

    return {
        "valid": True,
        "has_ai_context": has_ai_keyword,
        "category": _categorize_question(question),
        "suggested_improvements": _suggest_question_improvements(question)
    }

def _categorize_question(question: str) -> str:
    """Categorize the type of question for better routing."""
    question_lower = question.lower()

    if any(word in question_lower for word in ["what is", "define", "explain", "meaning"]):
        return "concept_explanation"
    elif any(word in question_lower for word in ["how to", "how do", "steps", "tutorial"]):
        return "how_to_guide"
    elif any(word in question_lower for word in ["error", "bug", "problem", "issue", "stuck"]):
        return "troubleshooting"
    elif any(word in question_lower for word in ["project", "assignment", "homework"]):
        return "project_help"
    elif any(word in question_lower for word in ["career", "job", "interview", "salary"]):
        return "career_advice"
    elif any(word in question_lower for word in ["schedule", "timeline", "event", "when"]):
        return "community_logistics"
    else:
        return "general_question"

def _suggest_question_improvements(question: str) -> List[str]:
    """Suggest ways to improve the question for better answers."""
    suggestions = []

    if "?" not in question:
        suggestions.append("Consider ending with a question mark to clarify what you're asking")

    if len(question.split()) < 5:
        suggestions.append("Provide more context or specific details for a more comprehensive answer")

    question_lower = question.lower()

    # Suggest specificity improvements
    if "machine learning" in question_lower and "algorithm" not in question_lower:
        suggestions.append("Specify which ML algorithm or technique you're interested in")

    if "python" in question_lower and "library" not in question_lower:
        suggestions.append("Mention specific Python libraries (pandas, scikit-learn, etc.) if relevant")

    if any(word in question_lower for word in ["project", "collaboration"]) and "type" not in question_lower:
        suggestions.append("Specify which type of project or collaboration you're interested in")

    return suggestions