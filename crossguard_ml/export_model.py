"""Rebuild the supplied ExecuTorch XNNPACK FP32 artifact from local weights."""
import argparse, json, platform, time
from pathlib import Path
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
from crossguard.model import EdgeDetector,load_detector,WEIGHTS,ROOT

def main():
    """Export PyTorch SSDLite320 detector to an ExecuTorch FP32 program and verify numerical parity.
    
    Export workflow:
    1. Loads local PyTorch weights and wraps model inside EdgeDetector module.
    2. Uses torch.export.export() to generate an ExportedProgram with strict schema.
    3. Lowers graph using to_edge_transform_and_lower() with the XnnpackPartitioner delegate.
    4. Serializes the final ExecuTorch binary (.pte) to disk.
    5. Loads the exported binary with the ExecuTorch runtime and runs multi-seed random input
       parity checks (seeds 42, 43, 44) to ensure mathematical fidelity against PyTorch eager output.
    6. Writes export validation metrics to reports/export_validation.json.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'models/crossguard_ssdlite_fp32.pte'); a=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(42)
    model=EdgeDetector(load_detector(a.weights)).eval()
    inputs=(torch.rand(1,3,320,320),)
    with torch.no_grad():
        expected=model(*inputs)
        ep=torch.export.export(model,inputs,strict=True)
        program=to_edge_transform_and_lower(ep,partitioner=[XnnpackPartitioner()]).to_executorch()
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(program.buffer)
    runtime=Runtime.get(); loaded=runtime.load_program(a.output); method=loaded.load_method('forward')
    results=[]
    # Check multiple random inputs for graph numerical fidelity; this is NOT accuracy validation.
    with torch.inference_mode():
        for seed in [42,43,44]:
            torch.manual_seed(seed); x=torch.rand(1,3,320,320)
            ref=model(x); actual=method.execute([x])
            errors=[float((v-r).abs().max()) for v,r in zip(actual,ref)]
            torch.testing.assert_close(actual[0],ref[0],atol=0.02,rtol=1e-4)
            torch.testing.assert_close(actual[1],ref[1],atol=2e-5,rtol=1e-3)
            results.append({'seed':seed,'max_box_abs_error':errors[0],'max_score_abs_error':errors[1]})
    report={'export_passed':True,'runtime_load_and_execution_passed':True,'random_input_parity':results,
        'output_shapes':[list(v.shape) for v in expected],'size_bytes':a.output.stat().st_size,
        'backend':'XNNPACK','precision':'FP32','host_arch':platform.machine(),
        'pi_tested':False,'torch_version':torch.__version__,
        'note':'Numerical parity on random tensors is not object-detection accuracy or Raspberry Pi validation.'}
    (ROOT/'reports/export_validation.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
