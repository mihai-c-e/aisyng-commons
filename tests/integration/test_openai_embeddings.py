from dotenv import load_dotenv
load_dotenv()

import asyncio
from aisyng.commons.embeddings.embedding_service import embed_documents, aembed_documents

async def amain():
    try:
        documents = ["111 222 333"] * 10000
        results = await aembed_documents(
            documents=documents,
            class_name="OpenAIEmbeddings",
            module_name="langchain_openai",
            model="text-embedding-3-small",
            dimensions=2
        )
        return results
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(amain())
    print(results)
