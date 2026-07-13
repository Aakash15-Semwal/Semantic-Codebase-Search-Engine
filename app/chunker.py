import uuid
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query, QueryCursor
from transformers import RobertaTokenizer

class CodeChunker:
    def __init__(self, tokenizer, max_tokens=512, overlap_pct=0.3):
      self.tokenizer = tokenizer
      self.max_tokens = max_tokens
      self.overlap_pct = overlap_pct
      PY_LANGUAGE = Language(tspython.language())
      self.parser = Parser(PY_LANGUAGE)
      self.query = Query(PY_LANGUAGE, """
      ; --- Module-level functions ---
      (module (function_definition) @func)

      (module (decorated_definition definition: (function_definition)) @decorated_func)

      ; --- Module-level classes ---
      (module (class_definition) @class)

      (module (decorated_definition definition: (class_definition)) @decorated_class)

      ; --- Methods inside classes ---
      (class_definition
        body: (block
          (function_definition) @class_func))

      (class_definition
        body: (block
          (decorated_definition definition: (function_definition)) @class_decorated_func))

      ; --- Nested functions / local classes ---
      (function_definition
        body: (block
          [(function_definition) @nested_func
          (class_definition) @local_class]))
      """)

    def file_to_tree(self, file_path):
      """Converts the file into tree"""
      self.file_path = file_path
      with open(self.file_path, 'r') as f:
        FILE = f.read()
      tree = self.parser.parse(bytes(FILE, "utf8"))
      return tree

    def get_preceding_comments(self, node):
      """Walk backward through consecutive sibling comment nodes."""
      comments = []
      sibling = node.prev_sibling
      while sibling is not None and sibling.type == "comment":
          comments.insert(0, sibling.text.decode("utf8"))
          sibling = sibling.prev_sibling
      return "\n".join(comments)

    def get_class_name(self, node):
      """Get the class(s) name the function belongs to."""
      classes = []
      while node != None:
        if node.type == 'class_definition':
          classes.append(node.child_by_field_name('name').text.decode('utf8'))
        node = node.parent
      if len(classes) > 0:
        return '.'.join(classes[::-1])
      else:
        return None

    def get_captures(self, root_node):
      """Return the captures according to the query."""
      cursor = QueryCursor(self.query)
      captures = cursor.captures(root_node)
      return captures

    def create_chunks(self, captures):
      """Creates chunks from the captures."""
      chunks = []
      for capture_group_name, nodes_in_group in captures.items():
        for node in nodes_in_group:
          func_or_class_name = None
          # For decorated definitions, the actual definition (which has the name) is in the 'definition' child
          if capture_group_name in ['decorated_func', 'decorated_class', 'class_decorated_func']:
              name_source_node = node.child_by_field_name("definition")
          else:
              # For other types, the 'name' is a direct child
              name_source_node = node

          if name_source_node:
              name_child = name_source_node.child_by_field_name('name')
              if name_child:
                  func_or_class_name = name_child.text.decode('utf8')

          chunk = {
            'location': self.file_path,
            "chunk_id": str(uuid.uuid4()),
            'start': node.start_point[0]+1,
            'end': node.end_point[0]+1,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'class': self.get_class_name(node),
            'type': node.type,
            'function': func_or_class_name,
            'preceding_comments': self.get_preceding_comments(node),
            'code': node.text.decode('utf8')
          }
          chunks.append(chunk)
      return chunks

    def get_line_token_counts(self, code_lines):
      """
      Build a map of line_number -> token_count for a function's lines.
      code_lines: list of strings, one per line (0-indexed list, but keyed by actual line number)
      """
      line_token_counts = {}
      for i, line in enumerate(code_lines):
          tokens = self.tokenizer.tokenize(line)
          line_token_counts[i] = len(tokens)
      return line_token_counts


    def forward_pass(self, start_idx, line_token_counts):
      """
      Walk forward from start_idx, accumulating tokens until adding the
      next line would exceed max_tokens. Returns the end_idx of this sub-chunk.
      """
      running_total = 0
      end_idx = start_idx

      for i in range(start_idx, len(line_token_counts)):
          line_tokens = line_token_counts[i]
          if running_total + line_tokens > self.max_tokens:
              break
          running_total += line_tokens
          end_idx = i

      return end_idx, running_total


    def backward_overlap(self, end_idx, line_token_counts, sub_chunk_total_tokens):
      """
      Walk backward from end_idx, accumulating tokens until reaching ~overlap_pct
      of sub_chunk_total_tokens. Returns the start_idx for the next sub-chunk.
      """
      target_overlap = sub_chunk_total_tokens * self.overlap_pct
      running_total = 0
      start_idx = end_idx

      for i in range(end_idx, -1, -1):
          line_tokens = line_token_counts[i]
          if running_total + line_tokens > target_overlap:
              break
          running_total += line_tokens
          start_idx = i

      return start_idx


    def split_into_subchunks(self, code_lines):
      """
      Full splitting algorithm: returns a list of (start_idx, end_idx) tuples,
      each representing one sub-chunk's line range (0-indexed, inclusive).
      """
      line_token_counts = self.get_line_token_counts(code_lines)
      function_end_idx = len(code_lines) - 1

      sub_chunks = []
      current_start = 0

      while True:
          end_idx, sub_total = self.forward_pass(current_start, line_token_counts)
          sub_chunks.append((current_start, end_idx))

          if end_idx >= function_end_idx:
              break

          current_start = self.backward_overlap(end_idx, line_token_counts, sub_total)

      return sub_chunks

    def expand_chunk_if_needed(self, chunk):
      """
      Takes a single chunk dict from pass 1. If its code fits within max_tokens,
      returns it unchanged (wrapped in a list). If too big, splits it into
      multiple linked sub-chunks sharing a parent_id.
      """
      full_token_count = len(self.tokenizer.tokenize(chunk['code']))

      if full_token_count <= self.max_tokens:
          chunk['chunk_id'] = str(uuid.uuid4())
          chunk['parent_id'] = None
          chunk['sequence_index'] = 0
          return [chunk]

      code_lines = chunk['code'].split('\n')
      sub_ranges = self.split_into_subchunks(code_lines)

      parent_id = str(uuid.uuid4())
      sub_chunks = []

      for seq_idx, (start_idx, end_idx) in enumerate(sub_ranges):
          sub_code = '\n'.join(code_lines[start_idx:end_idx + 1])
          sub_chunk = chunk.copy()
          sub_chunk['code'] = sub_code
          sub_chunk['chunk_id'] = str(uuid.uuid4())
          sub_chunk['parent_id'] = parent_id
          sub_chunk['sequence_index'] = seq_idx
          # adjust line numbers relative to the original chunk's start line
          sub_chunk['start'] = chunk['start'] + start_idx
          sub_chunk['end'] = chunk['start'] + end_idx
          sub_chunks.append(sub_chunk)

      return sub_chunks

    def chunk_file(self, file_path):
      """
      Full pipeline: file -> tree -> captures -> pass 1 chunks -> pass 2 (splitting).
      Returns the final flat list of chunk dicts ready for embedding.
      """
      tree = self.file_to_tree(file_path)
      captures = self.get_captures(tree.root_node)
      raw_chunks = self.create_chunks(captures)

      final_chunks = []
      for chunk in raw_chunks:
          final_chunks.extend(self.expand_chunk_if_needed(chunk))

      return final_chunks