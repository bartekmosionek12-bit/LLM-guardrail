# 🛡️ LLM Guardrail Engine (Multi-layer AI Safety Pipeline)

[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](#current-status)
[![Compliance](https://img.shields.io/badge/Compliance-EU%20AI%20Act-blue.svg)](#eu-ai-act-roadmap)
[![Security](https://img.shields.io/badge/Focus-Adversarial%20Robustness-red.svg)](#defense-capabilities)

## 📖 Overview
The **LLM Guardrail Engine** is a sophisticated, multi-layered security middleware designed to intercept and neutralize adversarial attacks directed at Large Language Models (LLMs). Acting as a proactive security layer between the user and the core AI model, the architecture seamlessly combines low-latency heuristic filters with deep semantic analysis. 

This engine acts as a **Semantic Firewall**, evaluating user intent through hybrid analysis before any prompt reaches the inference stage.

## 🏗️ System Architecture
The pipeline follows a "Defense-in-Depth" strategy, ensuring that if one layer is bypassed, subsequent layers catch the anomaly.

1.  **Heuristic & Regex Layer:** Rapid scanning for known malicious patterns and PII (Personally Identifiable Information).
2.  **Semantic Similarity Engine:** Utilizing `Sentence-Transformers` to perform semantic similarity searches against a dynamic, dual-vector database.
3.  **Neural Classification Layer:** Integration with `Llama-Guard-3` for intelligent intent classification, leveraging In-Context Learning (ICL).
4.  **Feedback & Continuous Learning:** Automated pattern learning mechanism which dynamically expands the vector store with newly detected threats.

### 🛠️ Interactive Documentation (Swagger UI)
To ensure professional-grade observability and seamless integration, I implemented a robust administrative API suite, fully documented and accessible via **Swagger UI (OpenAPI standard)**.
![API Overview](api.png)

## 🛡️ Defense Capabilities
The engine is specifically engineered to mitigate the following high-risk attack vectors:
*   **Prompt Injection:** Indirect and direct attempts to hijack the model's instructions.
*   **Jailbreaking:** Sophisticated role-playing or logic-based attempts to bypass safety filters (e.g., DAN-style prompts).
*   **Payload Splitting & Obfuscation:** Detecting malicious intent hidden through Base64 encoding or character manipulation.
*   **Semantic Bypassing:** Intercepting prompts that use synonyms or complex phrasing to hide harmful intent from standard keyword filters.

### Performance Examples
| Action | Input / Output |
| :--- | :--- |
| **Blocked Attack** | ![Blocked Demo](malicious_prompt_attempt.png) |
| **Safe Processing** | ![Safe Demo](safe_prompt_attempt.png) |

## ⚖️ EU AI Act Roadmap
The foundational vision for this project is to build a scalable framework capable of meeting the strict safety, transparency, and accountability requirements outlined in the upcoming **EU AI Act**:

*   **Art. 15 (Robustness & Accuracy):** Implementing measures to prevent "adversarial examples" and model manipulation.
*   **Art. 10 (Data Governance):** Ensuring inputs are scanned for PII and sensitive data before processing.
*   **Art. 13 (Transparency):** Detailed logging and telemetry of blocked attempts for auditability.

## 🚀 Current Status & Public Release
The system is currently in an active development phase, where I am independently engineering, red-teaming, and thoroughly testing the architecture.

> **Notice:** The codebase is currently under **Private Security Audit** and intensive **Adversarial Red-Teaming**. 

*   **Phase 1 (Current):** Engineering & Stress-Testing.
*   **Phase 2 (Late 2026):** Beta release for selected security researchers.
*   **Phase 3 (2027):** Open-source core with EU AI Act compliance templates.

## 🛠️ Tech Stack
*   **Backend:** FastAPI (Python)
*   **AI Models:** Llama-Guard-3, SBERT (Sentence-Transformers)
*   **Security Documentation:** OpenAPI / Swagger UI
*   **Testing:** Custom Adversarial Prompt Dataset
