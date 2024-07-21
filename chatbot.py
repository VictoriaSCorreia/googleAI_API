""" pip install -q -U google-generativeai """

import google.generativeai as genai
import os

GOOGLE_API_KEY="" 
genai.configure(api_key=GOOGLE_API_KEY) 
''' genai.configure(api_key="") '''

# Verifying Existing Genai Models
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# Configuration
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}
safety_settings = {
    "HARASSMENT": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_NONE",
    "SEXUAL": "BLOCK_NONE",
    "HATE": "BLOCK_NONE"
}

# Creating a new model
model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

''' response = model.generate_content("Write a story about a dark castle.")
print(response.text) '''

chat = model.start_chat(history=[]) # Starting a chat (history will allow it to "remember" the previous questions and the context)
while True:
  prompt = input("Esperando prompt: ") 
  response = chat.send_message(prompt)
  print(response.text)
  if prompt == "fim":
    break
