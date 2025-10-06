#!/usr/bin/env python3
"""
Discord RAG Bot Backend Entry Point

This script serves as the main entry point for the Discord RAG Bot backend.
It can run the FastAPI server, Discord bot, or both depending on configuration.
"""

import os
import sys
import asyncio
import signal
from typing import Optional
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from app.config import get_settings


settings = get_settings()


def setup_logging():
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Add file logger with rotation
    logger.add(
        logs_dir / "app.log",
        rotation="1 day",
        retention="30 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )

    # Add console logger
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True
    )


def run_api_server():
    """Run the FastAPI server."""
    import uvicorn
    from app.main import app

    logger.info("Starting FastAPI server...")
    logger.info(f"Server will be available at http://{settings.host}:{settings.port}")
    logger.info(f"API documentation at http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug and settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True
    )


async def run_discord_bot():
    """Run the Discord bot."""
    try:
        from bot.discord_bot import bot
        logger.info("Starting Discord bot...")
        await bot.setup_commands()  
        await bot.start(settings.discord_bot_token)
    except ImportError:
        logger.error("Discord bot module not found. Ensure bot/discord_bot.py exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {e}")
        sys.exit(1)


async def run_both():
    """Run both FastAPI server and Discord bot."""
    logger.info("Starting both FastAPI server and Discord bot...")

    # Create tasks for both services
    api_task = asyncio.create_task(run_api_server_async())
    bot_task = asyncio.create_task(run_discord_bot())

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        api_task.cancel()
        bot_task.cancel()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Wait for both tasks to complete
        await asyncio.gather(api_task, bot_task, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Services cancelled, shutting down...")
    except Exception as e:
        logger.error(f"Error running services: {e}")
    finally:
        logger.info("All services stopped.")


async def run_api_server_async():
    """Run FastAPI server in async context."""
    import uvicorn
    from app.main import app

    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug and settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True
    )

    server = uvicorn.Server(config)
    await server.serve()


def validate_environment():
    """Validate required environment variables."""
    required_vars = []

    # Check run mode specific requirements
    run_mode = os.getenv("RUN_MODE", "api").lower()

    if run_mode in ["api", "both"]:
        # FastAPI requirements are mostly optional with defaults
        pass

    if run_mode in ["bot", "both"]:
        if not settings.discord_bot_token or settings.discord_bot_token == "your_discord_bot_token_here":
            required_vars.append("DISCORD_BOT_TOKEN")

    # Check for database connection (required for production)
    if settings.environment == "production":
        if not settings.mongodb_uri or "your_username" in settings.mongodb_uri:
            required_vars.append("MONGODB_URI")

        if not settings.azure_openai_api_key or settings.azure_openai_api_key == "your_azure_openai_api_key_here":
            required_vars.append("AZURE_OPENAI_API_KEY")

    if required_vars:
        logger.error("Missing required environment variables:")
        for var in required_vars:
            logger.error(f"  - {var}")
        logger.error("Please check your .env file or environment configuration.")
        sys.exit(1)


def main():
    """Main entry point."""
    # Setup logging first
    setup_logging()

    logger.info(f"Starting {settings.app_name} v1.0.0")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")

    # Validate environment
    validate_environment()

    # Determine run mode
    run_mode = os.getenv("RUN_MODE", "api").lower()
    logger.info(f"Run mode: {run_mode}")

    try:
        if run_mode == "api":
            # Run only FastAPI server
            run_api_server()

        elif run_mode == "bot":
            # Run only Discord bot
            asyncio.run(run_discord_bot())

        elif run_mode == "both":
            # Run both services
            asyncio.run(run_both())

        else:
            logger.error(f"Invalid RUN_MODE: {run_mode}. Valid options: api, bot, both")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()