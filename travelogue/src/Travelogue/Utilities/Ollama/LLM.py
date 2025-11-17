# Use Ollama LLM to provide a comprehensive answer
def inference_server_model(relevant_chunks, q):                    
    from ollama import Client
    client = Client()
    prompt = f"""
    Based on the following college admission document search results, provide a comprehensive answer to the user's query: "{q}"
    Search Results:
    {relevant_chunks}
    Please provide:
    1. A direct answer to the user's query based on the search results
    2. Key insights from the most relevant results
    3. How the search results relate to the user's question
    Answer: """
    response = client.generate(model='llama3.1', prompt=prompt)
    return response['response']