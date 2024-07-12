import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from aisyng.commons.embeddings.embedding_service import embed_documents, aembed_documents


class TestEmbedDocuments(unittest.TestCase):

    @patch('importlib.import_module')
    def test_missing_class_or_module_name(self, mock_import_module):
        with self.assertRaises(ValueError):
            embed_documents(["doc1", "doc2"], class_name=None, module_name="langchain")
        with self.assertRaises(ValueError):
            embed_documents(["doc1", "doc2"], class_name="OpenAIEmbeddings", module_name=None)

    @patch('importlib.import_module')
    def test_module_not_found(self, mock_import_module):
        mock_import_module.side_effect = ModuleNotFoundError

        with self.assertRaises(ImportError):
            embed_documents(["doc1", "doc2"], class_name="SomeClass", module_name="non_existent_module")

    def test_class_not_found(self):
        with self.assertRaises(ImportError):
            embed_documents(["doc1", "doc2"], class_name="NonExistentClass", module_name="langchain")

    @patch('importlib.import_module')
    def test_class_not_callable(self, mock_import_module):
        module = MagicMock()
        setattr(module, 'NotCallableClass', None)
        mock_import_module.return_value = module

        with self.assertRaises(TypeError):
            embed_documents(["doc1", "doc2"], class_name="NotCallableClass", module_name="some_module")

    @patch('importlib.import_module')
    def test_class_instantiation_error(self, mock_import_module):
        MockClass = MagicMock(side_effect=Exception("Instantiation error"))
        module = MagicMock()
        setattr(module, 'ErrorClass', MockClass)
        mock_import_module.return_value = module

        with self.assertRaises(RuntimeError):
            embed_documents(["doc1", "doc2"], class_name="ErrorClass", module_name="some_module")

    @patch('importlib.import_module')
    def test_missing_embed_method(self, mock_import_module):
        MockClass = MagicMock()
        instance = MockClass.return_value
        del instance.embed

        module = MagicMock()
        setattr(module, 'EmbedClass', MockClass)
        mock_import_module.return_value = module

        with self.assertRaises(AttributeError):
            embed_documents(["doc1", "doc2"], class_name="EmbedClass", module_name="some_module")

    @patch('importlib.import_module')
    def test_embed_success(self, mock_import_module):
        MockClass = MagicMock()
        instance = MockClass.return_value
        instance.embed.return_value = [[0.1, 0.2, 0.3]]

        module = MagicMock()
        setattr(module, 'EmbedClass', MockClass)
        mock_import_module.return_value = module

        embeddings = embed_documents(["doc1"], class_name="EmbedClass", module_name="some_module")
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])


class TestAEmbedDocuments(unittest.TestCase):

    @patch('importlib.import_module')
    def test_missing_class_or_module_name(self, mock_import_module):
        with self.assertRaises(ValueError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name=None, module_name="some_module"))
        with self.assertRaises(ValueError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="SomeClass", module_name=None))

    @patch('importlib.import_module')
    def test_module_not_found(self, mock_import_module):
        mock_import_module.side_effect = ModuleNotFoundError

        with self.assertRaises(ImportError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="SomeClass", module_name="non_existent_module"))

    def test_class_not_found(self):
        with self.assertRaises(ImportError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="NonExistentClass", module_name="langchain"))

    @patch('importlib.import_module')
    def test_class_not_callable(self, mock_import_module):
        module = AsyncMock()
        setattr(module, 'NotCallableClass', None)
        mock_import_module.return_value = module

        with self.assertRaises(TypeError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="NotCallableClass", module_name="some_module"))

    @patch('importlib.import_module')
    def test_class_instantiation_error(self, mock_import_module):
        MockClass = MagicMock(side_effect=Exception("Instantiation error"))
        module = MagicMock()
        setattr(module, 'ErrorClass', MockClass)
        mock_import_module.return_value = module

        with self.assertRaises(RuntimeError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="ErrorClass", module_name="some_module"))


    @patch('importlib.import_module')
    def test_missing_aembed_method(self, mock_import_module):
        MockClass = AsyncMock()
        instance = MockClass.return_value
        del instance.aembed

        module = AsyncMock()
        setattr(module, 'EmbedClass', MockClass)
        mock_import_module.return_value = module

        with self.assertRaises(AttributeError):
            asyncio.run(aembed_documents(["doc1", "doc2"], class_name="EmbedClass", module_name="some_module"))

    @patch('importlib.import_module')
    def test_aembed_success(self, mock_import_module):
        MockClass = MagicMock()
        instance = MockClass.return_value
        instance.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        module = MagicMock()
        setattr(module, 'EmbedClass', MockClass)
        mock_import_module.return_value = module

        embeddings = asyncio.run(aembed_documents(["doc1"], class_name="EmbedClass", module_name="some_module"))
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])

if __name__ == '__main__':
    unittest.main()
