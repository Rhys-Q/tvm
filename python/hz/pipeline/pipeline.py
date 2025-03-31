from typing import List, Optional

import tvm
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm import IRModule
import logging
logger = logging.getLogger( "opt_llm" )
@tvm.transform.module_pass(opt_level=0, name="_LogProgress")
class _LogProgress:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging."""

    def __init__(self, *args):
        self.args = args

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation"""
        logger.info(*self.args)
        print(*self.args, flush=True)
        return mod
@register_pipeline("opt_llm")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: List[nn.ExternModule] = None,
):
    ext_mods = ext_mods or []
    work_dir = "/home/hz/qzq_work/tvm/outputs/ms"
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 1. Passes on high-level operator graph
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                tvm.relax.backend.DispatchSampling(),
                tvm.relax.backend.DispatchSortScan(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),
                # Phase 4. Low-level Optimizations
                # # metascheduler
                # relax.transform.MetaScheduleTuneIRMod(
                #     params={}, work_dir=work_dir, max_trials_global=200
                # ),
                _LogProgress("Start dlight"),
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    # dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                _LogProgress("End dlight"),
                
                # Phase 5. Lowering to VM bytecode
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                _LogProgress("Begin RewriteCUDAGraph"),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                _LogProgress("Begin AttachExternModules"),
                relax.transform.AttachExternModules(ext_mods),
                _LogProgress("Success"),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline