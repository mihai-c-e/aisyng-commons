from typing import List, Callable, Any
import importlib


def _embedding_class_for_name(
        class_name: str,
        module_name: str,
        method_name: str,
        **kwargs
) -> Any:
    if not class_name or not module_name:
        raise ValueError("Both class_name and module_name must be specified")

    try:
        # Import the specified module
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"The module '{module_name}' could not be found.")

    try:
        # Get the specified class
        EmbeddingClass = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"The class '{class_name}' does not exist in the module '{module_name}'.")

    if not callable(EmbeddingClass):
        raise TypeError(f"The attribute '{class_name}' in module '{module_name}' is not callable.")

    try:
        # Instantiate the class with any additional keyword arguments
        embedder = EmbeddingClass(**kwargs)
    except Exception as e:
        raise RuntimeError(f"An error occurred while instantiating '{class_name}': {e}")

    # Ensure the embedder has an `embed` method
    if not hasattr(embedder, method_name) or not callable(getattr(embedder, method_name)):
        raise AttributeError(f"The class '{class_name}' does not have a callable {method_name} method.")

    return embedder


def embed_documents(
        documents: List[str],
        class_name: str = None,
        module_name: str = None,
        **kwargs
) -> List[List[float]]:
    """
    Calculates document embeddings based on the python module and langchain class
    specified.
    :param documents: A list of documents to embed
    :param class_name: The name of the class to instantiate
    :param module_name: The module where the class is defined
    :param kwargs: Additional keyword arguments to pass to the class constructor
    :return: A list of embeddings, where each embedding is a list of floats
    """

    embedder = _embedding_class_for_name(
        class_name=class_name,
        module_name=module_name,
        method_name="embed_documents",
        **kwargs
    )

    try:
        # Call the `embed` method to get embeddings
        embeddings = embedder.embed_documents(documents)
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating embeddings: {e}")

    return embeddings


async def aembed_documents(
        documents: List[str],
        class_name: str = None,
        module_name: str = None,
        **kwargs
) -> List[List[float]]:
    """
    Calculates document embeddings based on the python module and langchain class
    specified.
    :param documents: A list of documents to embed
    :param class_name: The name of the class to instantiate
    :param module_name: The module where the class is defined
    :param kwargs: Additional keyword arguments to pass to the class constructor
    :return: A list of embeddings, where each embedding is a list of floats
    """
    embedder = _embedding_class_for_name(
        class_name=class_name,
        module_name=module_name,
        method_name="embed_documents",
        **kwargs
    )

    try:
        # Call the `aembed` method to get embeddings
        embeddings = await embedder.aembed_documents(documents)
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating embeddings: {e}")

    return embeddings
