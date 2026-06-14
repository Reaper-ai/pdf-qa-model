import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def load_model() -> InferenceClient:
    """
    Initializes a direct Hugging Face Inference Client.
    Points to a native, high-availability model to bypass provider routing 404s.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set in environment variables.")

    # Swapping to Qwen 2.5 Instruct which sits natively on HF core infrastructure
    return InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=hf_token
    )

class HallucinationChecker:
    def __init__(self, client: InferenceClient):
        self.client = client

    def verify_faithfulness(self, generated_answer: str, retrieved_chunks: list) -> bool:
        """Audits the generated answer using direct chat completions."""
        if not generated_answer or not retrieved_chunks:
            return False
            
        context_text = " ".join([chunk["content"] for chunk in retrieved_chunks])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an unbiased factual auditing judge. Your job is to determine if the given Answer "
                    "is 100% supported by the Reference Context. Do not use outside assumptions.\n\n"
                    "Respond with exactly one word: 'YES' if the answer is completely supported, or 'NO' if it introduces unmentioned facts."
                )
            },
            {
                "role": "user",
                "content": f"Reference Context: {context_text}\n\nAnswer to Evaluate: {generated_answer}\n\nIs the Answer fully supported? (YES/NO):"
            }
        ]
        
        response = self.client.chat_completion(messages=messages, max_tokens=5, temperature=0.1)
        response_text = response.choices[0].message.content.strip().upper()
        return "YES" in response_text


def answer_question(client: InferenceClient, question: str, retrieved_chunks: list, is_confident: bool) -> dict:
    """
    Answers a question by mapping inputs directly to InferenceClient.chat_completion
    as recommended by Hugging Face documentation.
    """
    if not is_confident or not retrieved_chunks:
        return {
            "answer": "I cannot find sufficient verified information in the provided documents to answer this question.",
            "has_answer": False,
            "citations": []
        }

    context_data = json.dumps([
        {"text": c["content"], "metadata": c["metadata"]} for c in retrieved_chunks
    ], indent=2)

    # Structuring clean, explicit role boundaries using the message format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict production-grade Q&A assistant. Your task is to answer the user's question "
                "using ONLY the provided verified context blocks. Do not use outside knowledge.\n\n"
                "Instructions:\n"
                "1. Every claim must be backed by a context block.\n"
                "2. Output your response in raw JSON matching the schema below. No markdown wrapping.\n"
                "3. Include accurate matching citation metadata blocks.\n\n"
                "SCHEMA:\n"
                "{\n"
                "  \"answer\": \"Your precise text answer here.\",\n"
                "  \"has_answer\": true,\n"
                "  \"citations\": [ { \"source\": \"file.pdf\", \"type\": \"pdf\", \"page_number\": 1 } ]\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": f"Verified Context Blocks:\n{context_data}\n\nQuestion: {question}"
        }
    ]

    # Execute structured chat completion
    response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.1)
    raw_content = response.choices[0].message.content.strip()
    
    # Strip down markdown fence block cleanups if they leak
    if raw_content.startswith("```json"):
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
    elif raw_content.startswith("```"):
        raw_content = raw_content.replace("```", "").strip()

    try:
        parsed_response = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed_response = {
            "answer": raw_content,
            "has_answer": True,
            "citations": [chunk["metadata"] for chunk in retrieved_chunks]
        }

    # Run Fallback Verification Pass
    if parsed_response.get("has_answer", False):
        checker = HallucinationChecker(client=client)
        is_faithful = checker.verify_faithfulness(parsed_response["answer"], retrieved_chunks)
        
        if not is_faithful:
            return {
                "answer": "Fallback Triggered: The generated answer could not be verified with absolute certainty against the source texts.",
                "has_answer": False,
                "citations": []
            }

    return parsed_response