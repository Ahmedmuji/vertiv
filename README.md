# Vertiv Knowledge

A private, local-first chatbot and document search app for the Vertiv product library in this folder.

## What it does

- Incrementally indexes PDF, DOCX, PPTX, XLSX, and TXT files.
- Searches locally with a persistent ChromaDB vector store.
- Cites the source document and page, slide, or sheet for every result.
- Opens source PDFs and images in the browser and downloads Office source files.
- Can synthesize answers with Ollama or any OpenAI-compatible chat-completions endpoint.
- Falls back to grounded retrieval answers when no language model is configured.

The current dataset contains legacy `.ppt`, `.xls`, email, image, archive, video, and CAD files too. Those are intentionally skipped in this first version because they need format-specific conversion or OCR.

The default ChromaDB embedding function is local and deterministic, so the app can run privately without downloading an embedding model at startup. You can still add a stronger embedding model later if you want deeper semantic matching.

## Run

From PowerShell:

```powershell
.\run.ps1
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The first launch starts indexing automatically. Progress is shown in the header, and later launches only process new or changed files.

If you use a regular Python installation rather than the bundled Codex runtime:

```powershell
python -m pip install -r requirements.txt
python app.py
```

## Enable conversational answers

### Ollama (local and private)

Install Ollama, pull a model, and set the model before running the app:

```powershell
ollama pull llama3.1:8b
$env:LLM_MODEL = "llama3.1:8b"
.\run.ps1
```

The default endpoint is `http://127.0.0.1:11434/v1`.

### Groq

Add your Groq key to `.env`:

```text
GROQ_API_KEY=your-key
```

The app automatically maps that to Groq's OpenAI-compatible endpoint and uses `llama-3.3-70b-versatile` by default. To choose a different Groq model:

```text
GROQ_MODEL=your-groq-model-id
```

Groq answers send the retrieved document excerpts for each question to Groq's API.

### Other OpenAI-compatible endpoint

```powershell
$env:LLM_BASE_URL = "https://your-endpoint.example/v1"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "your-model-name"
.\run.ps1
```

`LLM_API_KEY` is read only by the server process and should not be committed to source control.

## Configuration

| Variable               | Default                            | Purpose                                                  |
| ---------------------- | ---------------------------------- | -------------------------------------------------------- |
| `VERTIV_DATASET_DIR` | `./Vertiv`                       | Document library path                                    |
| `VERTIV_CHROMA_PATH` | `./data/chroma`                  | Local ChromaDB vector store                              |
| `VERTIV_HOST`        | `127.0.0.1`                      | Bind address                                             |
| `VERTIV_PORT`        | `8000`                           | Web server port                                          |
| `VERTIV_AUTO_INDEX`  | `1`                              | Set to`0` to disable indexing on launch                |
| `LLM_BASE_URL`       | Ollama local endpoint              | OpenAI-compatible API base                               |
| `LLM_API_KEY`        | `ollama`                         | API credential                                           |
| `LLM_MODEL`          | unset                              | Enables synthesized answers when set                     |
| `GROQ_API_KEY`       | unset                              | Enables Groq if`LLM_MODEL` is not otherwise configured |
| `GROQ_MODEL`         | `llama-3.3-70b-versatile`        | Optional Groq model override                             |
| `GROQ_BASE_URL`      | `https://api.groq.com/openai/v1` | Optional Groq endpoint override                          |

## Deploy over SSH

For a Linux company server, use the helper script from Windows PowerShell:

```powershell
.\deploy\deploy-linux.ps1 -HostName "server.example.com" -User "deployuser" -RemotePath "/opt/vertiv-knowledge"
```

By default, `.env` is not included because it contains secrets. After the first deploy, edit `/opt/vertiv-knowledge/.env` on the server and add `GROQ_API_KEY`. If you explicitly want to copy the local `.env`, add `-IncludeEnv`.

The helper copies the app, dataset, and prebuilt Chroma index, creates a Python virtual environment, installs `requirements.txt`, and leaves a systemd service template in `deploy/vertiv-knowledge.service`.

## Notes

- Scanned PDFs without embedded text will be reported as indexing errors until OCR is added.
- The app binds to localhost by default. Do not expose it on a network without authentication and TLS.
- Product specifications should still be verified in the cited source before use in a quotation or design.
