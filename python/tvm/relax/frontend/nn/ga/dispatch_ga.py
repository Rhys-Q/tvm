"""A pass that rewrites KV cache creation functions in IRModule."""

import json
from typing import Any, Dict, List

import tvm
from tvm import IRModule, relax

from tvm.relax.frontend.nn.ga.genetic_algorithm import DefaultGeneticAlgorithm

import logging

logger = logging.getLogger(__name__)



@tvm.transform.module_pass(opt_level=0, name="DispatchGACreation")
class DispatchGACreation:  # pylint: disable=too-many-instance-attributes
    """Rewrite GA creation functions to IRModule."""

    def __init__(
        self
    ) -> None:
        """Initializer.
        """

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        func_dict = {}
        creation_func = None
        for g_var, func in mod.functions_items():
            # Try to find the `create_genetic_algorithm` func.
            if g_var.name_hint == "create_genetic_algorithm":
                creation_func = func
            else:
                func_dict[g_var] = func

        if creation_func is None:
            return mod

        new_mod = IRModule(func_dict)
        if mod.attrs is not None:
            new_mod = new_mod.with_attrs(mod.attrs)

        kwargs = {}
        bb = relax.BlockBuilder(new_mod)
        extern_mods = []
        extern_mods += self.create_normal_genetic_algorithm(bb, kwargs)

        mod = bb.finalize()
        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        mod = mod.with_attr("external_mods", mod_attrs.get("external_mods", []) + extern_mods)
        return mod



    def create_normal_genetic_algorithm(
        self, bb: relax.BlockBuilder, kwargs: Dict[str, Any]
    ) -> List[tvm.runtime.Module]:
        """Create default GeneticAlgorithm"""

        with bb.function(
            name="create_normal_genetic_algorithm",
            params=[],
        ):
            cache = DefaultGeneticAlgorithm()
            bb.emit_func_output(cache._expr)  # pylint: disable=protected-access

        return cache.extern_mods

