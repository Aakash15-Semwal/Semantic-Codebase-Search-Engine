from transformers import AutoTokenizer, AutoModel
from langdetect import detect, DetectorFactory
from tree_sitter import Language, Parser
from datasets import load_dataset
import tree_sitter_python

class DatasetPreparer:
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        PY_LANGUAGE = Language(tree_sitter_python.language())
        self.parser = Parser(PY_LANGUAGE)
        ds = load_dataset("code-search-net/code_search_net", "python")
        self.dataset = ds.select_columns(['func_code_string', 'func_documentation_string'])
        DetectorFactory.seed = 0

    def filter_func(self, example):
        # 1. Tokenize documentation and check length
        doc_tokens = self.tokenizer.tokenize(example['func_documentation_string'])
        if len(doc_tokens) < 15:
            return False

        # 2. Tokenize code and check length
        code_tokens = self.tokenizer.tokenize(example['func_code_string'])
        if len(code_tokens) > 512:
            return False

        # 3. Language detection (English only)
        try:
            if detect(example['func_documentation_string']) != 'en':
                return False
        except:
            return False

        return True

    def find_function(self, node):
        """Find the first function definition recursively."""
        if node.type == "function_definition":
            return node

        if node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition is not None:
                return definition

        for child in node.children:
            result = self.find_function(child)
            if result is not None:
                return result

        return None


    def strip_docstring(self, code: str) -> str:
        """
        Remove the leading docstring from the first function in `code`.
        """
        if not code:
            return code

        source = code.encode("utf-8")
        tree = self.parser.parse(source)
        root = tree.root_node

        func = self.find_function(root)
        if func is None:
            return code

        body = func.child_by_field_name("body")
        if body is None or body.named_child_count == 0:
            return code

        first_stmt = body.named_children[0]

        if (
            first_stmt.type == "expression_statement"
            and first_stmt.named_child_count == 1
            and first_stmt.named_children[0].type == "string"
        ):
            return (
                source[:first_stmt.start_byte]
                + source[first_stmt.end_byte:]
            ).decode("utf-8")

        return code


    def strip_fn(self, example):
        example['func_code_string'] = self.strip_docstring(example['func_code_string'])
        return example

    def prepare_dataset(self): 
        # Apply the filter to the DatasetDict
        self.dataset = self.dataset.filter(self.filter_func)
        self.filtered_dataset = self.dataset.map(self.strip_fn)