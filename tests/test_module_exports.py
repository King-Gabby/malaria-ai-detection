import ast
import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _collect_export_names(file_path: Path):
    module = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    exported = set()
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        exported.update(
                            elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported.add(alias.asname or alias.name)
    return exported


class ModuleExportTests(unittest.TestCase):
    def test_detection_module_exports_model_loaders(self):
        exports = _collect_export_names(ROOT / "app" / "modules" / "detection" / "__init__.py")
        expected = {
            "MalariaDetector",
            "SickleCellDetector",
            "ALLDetector",
            "IronDeficiencyDetector",
            "load_malaria_model",
            "load_sickle_cell_model",
            "load_all_model",
            "load_iron_deficiency_model",
        }
        self.assertTrue(expected.issubset(exports))

    def test_radiology_module_exports_model_loaders(self):
        exports = _collect_export_names(ROOT / "app" / "modules" / "radiology" / "__init__.py")
        expected = {
            "MRIDetector",
            "CTScanDetector",
            "XRayDetector",
            "load_mri_model",
            "load_ct_model",
            "load_xray_model",
        }
        self.assertTrue(expected.issubset(exports))

    def test_chatbot_module_import_is_safe_without_optional_dependencies(self):
        modules_to_remove = [
            "sentence_transformers",
            "chromadb",
            "langchain",
            "langchain_core",
            "langchain_community",
            "langgraph",
        ]
        existing = {name: sys.modules.get(name) for name in modules_to_remove}
        for name in modules_to_remove:
            sys.modules.pop(name, None)

        try:
            module = importlib.import_module("app.modules.chatbot")
            self.assertTrue(hasattr(module, "render_chatbot_ui"))
        finally:
            for name, value in existing.items():
                if value is not None:
                    sys.modules[name] = value
                else:
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
