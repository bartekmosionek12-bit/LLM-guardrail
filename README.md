# LLM Guardrail Engine (Multi-layer AI Safety Pipeline) 🛡️

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI Safety](https://img.shields.io/badge/Focus-AI_Safety_&_Compliance-red.svg)

## 📖 Project Overview
I am developing a comprehensive, multi-layered security guardrail system designed to protect Large Language Models (LLMs) from adversarial attacks, prompt injections, and jailbreaks. Acting as a proactive security layer between the user and the core AI model, the architecture seamlessly combines low-latency heuristic filters with deep semantic analysis.

## 🛠️ Key Technical Features
*   **Semantic Intent Recognition:** Uses `Sentence-Transformers` for vector-based similarity search (beyond simple keyword blacklisting).
*   **AI-Driven Classification:** Integration with `Llama-Guard-3` for deep content moderation.
*   **Automated Pattern Learning:** Dynamic expansion of the vector store with newly detected threats (IPS-like behavior).
*   **Professional API Documentation:** Fully documented via **Swagger UI (OpenAPI)** for easy integration.


## 🚀 Vision: EU AI Act Compliance
The foundational vision for this project is to build a scalable framework capable of meeting the strict safety, transparency, and accountability requirements outlined in the upcoming **EU AI Act**.

## 🏗️ Current Status
The project is in an **active development phase**. I am independently engineering, red-teaming, and testing the architecture to refine its accuracy and defensive capabilities.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Framework:** FastAPI
*   **Models:** Llama-3 (via Ollama/Local), Llama-Guard-3, SBERT
*   **Database:** Vector Store (e.g., ChromaDB / FAISS)
*   **Documentation:** Swagger UI


## 🛡️ Live Demo & Examples

### 1. Full API Structure
Comprehensive security console for monitoring and system management.
![API Endpoints](assets/api.png)

### 2. Guardrail in Action: Blocked Attack
The system detects "jailbreak" attempts and harmful prompts in real-time.
![Blocked Attack](assets/malicious_prompt_attempt.png)

### 3. Standard Operation: Safe Request
Valid technical queries are processed normally with low latency.
![Safe Query](assets/safe_prompt_attempt.png)
