from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()
def load_model() -> HuggingFaceEndpoint:
    """
    Loads the model securely using HF token from environment variable.
    :return: HuggingFaceEndpoint object
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set in environment variables.")

    # noinspection PyArgumentList
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=256,
        huggingfacehub_api_token=hf_token
    )

    return llm

def answer_question(llm:HuggingFaceEndpoint, question:str, context:str) -> str:
    """
    Answers a question using the provided context and  model.
    :param llm: LLM model
    :param question: Question to be answered
    :param context: Context relevant to the question
    :return: Answer as string
    """

    model = ChatHuggingFace(llm=llm)
    response = model.invoke(f"Refer to the following context: {context} and answer thw question: {question}")
    return response.content  # or just print(response)
