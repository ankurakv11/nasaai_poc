# NASA AI – Sizing API (Dockerized Setup)

This repository contains a **Dockerized FastAPI-based Sizing API** split into two services:

- **API service** – Handles HTTP requests, routing, validation, and user interaction
- **Worker service** – Handles heavy ML/AI processing (MediaPipe, U2Net, image processing)

The setup is designed for:
- Easy local execution
- Better scalability
- Separation of user load and processing load

---

## 📋 Prerequisites

Before starting, make sure the system has:

- **Docker** (Docker Desktop or Docker Engine)
- **Docker Compose v2**

Verify installation:

```bash
docker --version
docker compose version
⚠️ If using WSL, ensure Docker Desktop is running and WSL integration is enabled.

📥 Clone the Repository
bash
Copy code
git clone https://github.com/ankurakv11/nasaai_poc.git
cd nasaai_poc
📁 Project Structure (Important)
text
Copy code
nasaai_poc/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── .dockerignore
│
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── app/
│       ├── routes/
│       ├── middleware/
│       ├── services/
│       ├── utils/
│       └── models/
│
├── worker/
│   ├── main.py
│   ├── requirements.txt
│   └── app/
│       └── utils/
│
└── uploads/
🐳 Build and Run with Docker Compose
From the project root directory:

bash
Copy code
docker compose up -d --build
What this does:
Builds both API and Worker Docker images

Starts both containers in detached (background) mode

Creates an internal Docker network for inter-service communication

✅ Verify Containers Are Running
bash
Copy code
docker ps
You should see two running containers:

text
Copy code
sizing_api
sizing_worker
🌐 Access the API
API Root
text
Copy code
http://localhost:8000/
Swagger Docs
text
Copy code
http://localhost:8000/docs
Health Check
text
Copy code
http://localhost:8000/health
📜 View Logs (Optional)
API logs
bash
Copy code
docker logs -f sizing_api
Worker logs
bash
Copy code
docker logs -f sizing_worker
Stop following logs with CTRL + C.

⏹️ Stop the Application
bash
Copy code
docker compose down
🔄 Restart the Application
bash
Copy code
docker compose restart
🧹 Clean Rebuild (If Needed)
If you want a fresh rebuild:

bash
Copy code
docker compose down
docker compose build --no-cache
docker compose up -d
