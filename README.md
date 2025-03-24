# 🏥 MEDICAL KIOSK AI – INTELLIGENT HEALTHCARE ASSISTANT

## Overview
**Medical Kiosk AI** is an advanced **multilingual AI-powered healthcare assistant** designed for **interactive patient interaction**. This system utilizes **speech recognition, natural language processing, and text-to-speech synthesis** to provide seamless communication in multiple languages. The project is integrated with an **IoT-based kiosk** for real-time medical assistance.

## 🚀 Features
✅ **Multilingual Pipeline**: Supports multiple languages using **Whisper for ASR**, **LLM-based medical response generation**, and **TTS conversion**.  
✅ **Fine-Tuned AI Models**: Uses **Mistral-7B-Instruct-v0.3** and **DeepSeek R1**, fine-tuned on **medical Q&A datasets** for accurate patient interactions.  
✅ **IoT Integration**: Runs on a **Raspberry Pi**, offloading heavy processing to a local **server using ngrok API**.  
✅ **Interactive AI Chatbot**: Provides **medical insights, symptom explanations, and health recommendations**.  
✅ **Cloud & Edge Deployment**: Optimized to work efficiently with **limited computing resources**.  

## 🏗️ System Workflow
1️⃣ **Speech Input (ASR)**: Patient speaks into the **medical kiosk**, and **Whisper ASR** converts speech to text.  
2️⃣ **Language Detection & Processing**: The text is sent to the server, which detects the language and forwards it to the appropriate **fine-tuned LLM (Mistral-7B / DeepSeek R1)**.  
3️⃣ **Medical AI Response**: The chatbot generates an accurate **medical response** based on the patient’s query.  
4️⃣ **Text-to-Speech Conversion (TTS)**: The AI response is converted into **speech in the patient’s language**.  
5️⃣ **IoT Output**: The **Raspberry Pi** receives the **generated speech** and plays it back to the patient.  

## ⚙️ Technologies Used
- **LLMs**: Fine-tuned **Mistral-7B-Instruct-v0.3** & **DeepSeek R1**  
- **Speech Recognition (ASR)**: **OpenAI Whisper**  
- **Text-to-Speech (TTS)**: Multilingual TTS engine  
- **IoT & Edge Deployment**: **Raspberry Pi + ngrok API for remote LLM access**  
- **Backend**: Python, FastAPI  
- **Integration**: AI processing on **local system**, remote access via **ngrok API**  

## 📌 Future Enhancements
🔹 Add **real-time patient monitoring with sensors**  
🔹 Improve **medical recommendation accuracy** using **clinical knowledge graphs**  
🔹 Optimize for **low-power IoT devices**  

---
🚀 **Contributions & feedback are welcome!** Let’s revolutionize **AI-driven healthcare** together! 🔥
