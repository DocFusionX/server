## DocFusionX Server

### Installation & Dev

#### 1. Clone the repo

```bash
git clone https://github.com/DocFusionX/server.git
cd server
```

#### 2. Setup env variables

`.env` template:

```ini
MISTRAL_API_KEY=
CHROMA_DB_PATH=./chroma_db
LOG_LEVEL=INFO
MISTRAL_MODEL=mistral-small-latest
MISTRAL_MAX_TOKENS=64000
```

#### 3. Set up virtual environment

```bash
uv venv
```
```bash
soruce .venv/bin/activate
```

#### 4. Install dependencies

```bash
uv sync
```

#### 5. Start development server

```bash
uv run fastapi dev
```

### Running the Server

```bash
docker compose up --build
```

### API Usage

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
