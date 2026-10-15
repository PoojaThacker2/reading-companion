# Reading Companion

A terminal chat app where you load an ebook (PDF or TXT) and talk to it. The agent **decides** per message:

- **Answer path** – a question about the book → retrieve relevant chunks from Chroma → grounded answer.
- **Schedule path** – you want help staying consistent → propose a reading session → **pause and ask you to confirm** → only then write `schedule.json`.

Every message first prints its routing decision, e.g. `→ routing: answer from book`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then put your key in .env  (macOS/Linux: cp)
python main.py sample_book.pdf
```

### Using Ollama instead of OpenAI (free, local)

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Then in `.env` set `OLLAMA_URL=http://localhost:11434/v1` (see `.env.example` for the model names and `MIN_SCORE`). No API key needed. `llm.py` is the only place that picks the provider.

`sample_book.pdf` (also provided as `sample_book.txt`) is a short fictional book, *The Lighthouse Keeper's Almanac*.

## Try these

**Answer path** (expect `→ routing: answer from book`)
- Who was Elias Marrow's predecessor at the lighthouse?
- What did Elias do when the clockwork failed during the storm?
- What did Ruth Aldane give Elias?
- What was written on the compass? *(and try something not in the book, like "Who won the 1990 World Cup?" – it should say it can't find it)*

**Schedule path** (expect `→ routing: schedule a reading session (needs confirmation)`)
- I keep forgetting to read, can you help me stay consistent?
- Remind me to keep reading tomorrow at 7pm
- I keep losing track of where I am

You'll be asked `[y/n]`. `y` writes the session to `schedule.json`; `n` writes nothing.

## How it works

| File | Role |
|---|---|
| `main.py` | Terminal loop. Starts the API in a background thread and talks to it over HTTP; no AI logic. |
| `api.py` | FastAPI: `POST /ingest`, `POST /chat`, `POST /confirm`. |
| `agent.py` | LangGraph graph: `router` → `answer` **or** `schedule_propose` → `schedule_confirm` (uses `interrupt()`). |
| `ingest.py` | Load PDF/TXT, chunk (500 tokens, 75 overlap), embed (`text-embedding-3-small`), store/search in Chroma (`./chroma_db`). |
| `llm.py` | Picks OpenAI or Ollama from `.env`; holds model names and `MIN_SCORE`. |
| `schedule_store.py` | Reads/writes `schedule.json` – the whole "scheduling system". |

The confirmation flow: `/chat` runs the graph until `interrupt()` pauses it and returns `needs_confirmation`; the terminal asks you; `/confirm` resumes the graph with your yes/no via `Command(resume=...)`.

Models: OpenAI `gpt-4o-mini` + `text-embedding-3-small` by default, or your Ollama models. `MIN_SCORE` drops weakly related chunks so off-topic questions get a "couldn't find it in the book" reply instead of a guess.

## Scope

One retrieval path, one tool, one book per run, no web UI, no real calendar. Paused conversations live in memory (`MemorySaver`), so they don't survive a restart.
