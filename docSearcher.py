""" pip install -q -U google-generativeai """

import google.generativeai as genai 
import numpy as np
import pandas as pd

def embed_fn(title, text, model):
  # Generating an embed of each document content
  return genai.embed_content(model=model,
                             content=text,
                             title=title,
                             task_type="RETRIEVAL_DOCUMENT")["embedding"]

def request_embed(request, df, model):
  # Generating an embed of the request about the documents
  requestEmbedding = genai.embed_content(model=model,
                                         content=request,
                                         task_type="RETRIEVAL_QUERY"
                                           )["embedding"]     
  scalar_products = np.dot(np.stack(df["Embeddings"]), requestEmbedding)     # Calc
  index = np.argmax(scalar_products)    # Seeing the maximum argument (nearest)
  return df.iloc[index]["Conteudo"]        # return the index with it's right content

GOOGLE_API_KEY="" 
genai.configure(api_key=GOOGLE_API_KEY) 
''' genai.configure(api_key="") '''

# Verifying Existing Embed Models
for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)

# Creating documents

DOCUMENT1 = {
    "Título": "Barras",
    "Conteúdo": "A barra oblíqua [ / ] é um sinal gráfico usado: Para indicar disjunção e exclusão."
}
DOCUMENT2 = {
     "Título": "Interrogações",
     "Conteúdo": "O ponto de interrogação [?] é um sinal de pontuação que indica uma pergunta. É usado apenas em frases interrogativas diretas."
}
documents = [DOCUMENT1, DOCUMENT2]

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

modelEmb = "models/embedding-001"
model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config)

# Creating a data frame (PANDAS)
df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]

# Creating a new column called Embeddings (PANDAS)
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"], modelEmb), axis=1)    
df    





