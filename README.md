## DocFusionX Server

### Installation

#### 1. Clone the repo

```bash
git clone https://github.com/your-username/DocFusionX.git
cd DocFusionX/server
```

#### 2. Setup env variables

`.env` template:

```ini
MISTRAL_API_KEY=
CHROMA_DB_PATH=./chroma_db
LOG_LEVEL=INFO
```

#### 3. Install dependencies

```bash
uv sync
```

### Running the Server

```bash
docker compose up --build
```

### API Usage

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
