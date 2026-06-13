from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)
def _format_context(chunks):
    """
    Turn a list of retrieved chunks into a numbered context block for the prompt.
    Filters out weak matches (distance > 1.0) before formatting.
    """
    strong = [c for c in chunks if c.get("distance", 0) <= 1.0]
    if not strong:
        return None

    lines = []
    for i, chunk in enumerate(strong, 1):
        lines.append(
            f"[{i}] (Source: {chunk['game']})\n{chunk['text'].strip()}"
        )
    return "\n\n".join(lines)

def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """

    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )
    context = _format_context(retrieved_chunks)

    if context is None:
        return (
            "The retrieved passages didn't match your question closely enough to give a "
            "reliable answer. Try rephrasing, or check that the right rule book is loaded."
        )

    user_message = f"""Context:
{context}

Question: {query}
Answer using only the context above. Cite which source each piece of information comes from."""

    
    
    response = _client.chat.completions.create(
          model=LLM_MODEL,
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": user_message},
          ],
      )

    return response.choices[0].message.content or (
          "The model returned an empty response. Please try again."
      )