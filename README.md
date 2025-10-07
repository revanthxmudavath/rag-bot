# RAG Bot

## Docker Commands

- Build image: `docker build -t rag-bot .`
- Run FastAPI API: `docker run --rm -p 8000:8000 --env-file .env rag-bot`
- Run Discord bot: `docker run --rm --env-file .env -e RUN_MODE=bot rag-bot`

## Uploading Documents / Files

Example ingestion request (replace placeholders):

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "documents": [
          {
            "title": "Getting Started Guide",
            "content": "ou’ve heard of coding, and you’ve definitely heard of vibes. But what do they have to do with each other? Vibe coding is an emerging field of development, thanks to AI. It’s helping people build websites, apps and more. To get a better idea of how vibe coding works, why it’s becoming increasingly popular and what you can do with it, we talked to product director Kelly Schaefer, who leads a portfolio of AI-powered products in Google Labs. What do you do at Google? My teams and I build what we call “future of” products, which focus on the future of design, writing and even software development. In the software arena, we’re thinking about how to democratize building products. It’s not just engineers who will be building in the future! And vibe coding can help with that democratization. What’s your definition of vibe coding?   Vibe coding lets you build what you envisioned in your head even if you don't have traditional coding skills. It’s a process where, for example, you can use an AI tool and explain what you want to make and what you want it to look like, and that tool will generate something for you that you can see and use. In the past, you would have had to manually write lines of code to do that. Do you need to have any coding skills to vibe code? You actually don’t — you can make simple apps just by vibe coding. But it might not be the best solution depending on what you’re trying to build and how many people you want to use it. If you want to bring a vibe-coded app all the way to being a fully launched product that a lot of people can use, you still need coding skill and precision. Sometimes people think “I just need to write two sentences about my app and I’ll have an app in the Google Play store that everyone can use!” So it’s not just like you can think of something, see it in your mind and poof — it’s vibe coded perfectly, working exactly as you imagined? Right. You can describe something in simple terms and get a vibe-coded app — but to turn it into a real product you’ll need to keep going. It’s great to start by opening a vibe coding tool and trying something simple — for example, the Canvas option in Gemini allows you to enter a prompt like “make me a web app prototype.” You’ll get a basic product. If you wanted to turn this into something lots of people could use, then you could take the next step and start coding, or sharing your basic web app with a developer who would take it to the next stage. For that step, there are tools like Jules, an AI coding agent from Labs, which connects with your code and adds its own code based on what you’ve already made — plus you can ask it to make changes using natural language. Starting this whole process with vibe coding means more of what you saw in your mind’s eye can make it into the final product.  Together, Stitch and Jules show how vibe coding isn’t only about generating snapshots of an experience, but about making the full loop from idea to design to production-ready code accessible. I’m guessing what I could vibe code would be very different from what an engineer could vibe code, right? Well, sure, but your purposes are probably different, too. For example, Stitch is great when you want to quickly describe or visualize an idea, while Jules can carry that forward into live prototypes and all the way into production. Used together, they mirror the way an engineer and a designer might collaborate. If you’re not an engineer or a designer, vibe coding is a way to visualize what you want an engineer to build. Instead of starting with a doc, start with an interactive visual. Also, vibe coding tools are totally something to just have fun with! You can make whatever you want for yourself or to share with friends for no reason other than your own entertainment.",
            "metadata": {"source": "manual"}
          }
        ],
        "chunk_size": 500,
        "chunk_overlap": 50,
        "replace_existing": true
      }'
```

