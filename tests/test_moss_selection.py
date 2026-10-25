"""Pure model-selection regressions; no Discord session or provider calls."""

import ast
import math
import re
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, TypeGuard


ROOT = Path(__file__).resolve().parents[1]


def selection_functions():
    # Load the unchanged pure functions without initializing the Discord cog.
    namespace = {
        "re": re,
        "math": math,
        "Mapping": Mapping,
        "NamedTuple": NamedTuple,
        "TypeGuard": TypeGuard,
        "MODEL_FAMILIES": ("moss", "anima", "illustrious", "noobai"),
    }
    exec(compile((ROOT / "arcenciel/moss_schema.py").read_text(), "moss_schema.py", "exec"), namespace)
    sources = {
        "moss.py": {
            "is_object",
            "is_array",
            "is_strings",
            "json_object",
            "positive_int",
            "positive_float",
            "string_setting",
            "moss_models",
            "resolve_profile",
        },
        "agent_utils.py": {
            "ModelChoice",
            "_token",
            "_base",
            "_family",
            "_model_choice",
            "available_models",
            "_matches",
            "_name_matches",
            "choose_model",
        },
    }
    for filename, names in sources.items():
        source = ast.parse((ROOT / "arcenciel" / filename).read_text())
        definitions = [
            node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class MossSelectionTests(unittest.TestCase):
    def setUp(self):
        self.select = selection_functions()["choose_model"]
        self.builtin = "moss_image_sft_int8_convrot.safetensors"
        self.finetune = "16338_18538_Mossquerade_v1.safetensors"
        self.options = {
            "limits": {
                "moss": {
                    "available": True,
                    "modes": ["txt2img"],
                    "maxBoxes": 8,
                    "coordinateSpace": "normalized_1000",
                    "defaults": {
                        "modelName": self.builtin,
                        "steps": 28,
                        "cfg": 4,
                        "shift": 5,
                        "samplerName": "euler",
                        "scheduler": "moss",
                        "width": 1024,
                        "height": 1024,
                    },
                }
            }
        }
        self.catalog = {"checkpoints": [self.entry(self.builtin), self.entry(self.finetune)]}

    def entry(self, name):
        return {
            "name": name,
            "storageState": "loaded",
            "baseModel": "Moss SFT",
            "modelCapabilities": ["moss", "moss_sft"],
        }

    def test_default_stays_stable_with_multiple_sft_models(self):
        self.assertEqual(self.select(self.options, self.catalog).name, self.builtin)
        del self.options["limits"]["moss"]["defaults"]["modelName"]
        self.assertEqual(self.select(self.options, self.catalog).name, self.builtin)

    def test_explicit_finetune_accepts_the_search_filename(self):
        self.assertEqual(
            self.select(self.options, self.catalog, checkpoint=self.finetune.removesuffix(".safetensors")).name,
            self.finetune,
        )

    def test_advertised_default_is_used_only_when_accessible(self):
        self.options["limits"]["moss"]["defaults"]["modelName"] = self.finetune
        self.assertEqual(self.select(self.options, self.catalog).name, self.finetune)
        self.catalog["checkpoints"].pop()
        self.assertEqual(self.select(self.options, self.catalog).name, self.builtin)

    def test_not_loaded_is_not_reported_as_an_account_denial(self):
        self.catalog["checkpoints"][1]["storageState"] = "not_loaded"
        with self.assertRaisesRegex(ValueError, "visible but is not loaded"):
            self.select(self.options, self.catalog, checkpoint=self.finetune)

    def test_unsupported_family_has_a_distinct_error(self):
        self.catalog["checkpoints"] = [
            {"name": "other.safetensors", "storageState": "loaded", "baseModel": "unsupported"}
        ]
        with self.assertRaisesRegex(ValueError, "model family is not supported"):
            self.select(self.options, self.catalog, checkpoint="other")

    def test_absent_model_does_not_claim_a_confirmed_permission_failure(self):
        with self.assertRaisesRegex(ValueError, "absent from the accessible generator catalog"):
            self.select(self.options, self.catalog, checkpoint="missing")


if __name__ == "__main__":
    unittest.main()
