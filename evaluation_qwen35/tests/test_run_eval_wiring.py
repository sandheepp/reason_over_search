import ast
from pathlib import Path


def _read_run_eval_ast():
    path = Path(__file__).resolve().parents[1] / "run_eval.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def test_method_name_default_is_search_r1():
    tree = _read_run_eval_ast()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--method_name":
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value == "search-r1"
                        found = True
    assert found


def test_func_dict_contains_only_search_r1_key_for_pipeline():
    tree = _read_run_eval_ast()
    dict_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "func_dict":
                    dict_node = node.value
                    break
    assert isinstance(dict_node, ast.Dict)
    keys = [k.value for k in dict_node.keys if isinstance(k, ast.Constant)]
    assert "search-r1" in keys
    assert "research" not in keys

