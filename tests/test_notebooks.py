import ast
from pathlib import Path
import nbformat
import pytest

NOTEBOOKS=list(Path(__file__).parents[1].glob('*/*.ipynb'))

@pytest.mark.parametrize('path',NOTEBOOKS,ids=lambda p:p.parent.name)
def test_notebook_schema_syntax_and_clean_outputs(path):
    book=nbformat.read(path,as_version=4);nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs
