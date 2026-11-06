"""Check PyTorch/ExecuTorch parity on included real photos; not an accuracy score."""
import json
import cv2
import torch
from executorch.runtime import Runtime
from crossguard.model import EdgeDetector,load_detector,ROOT
from crossguard.pipeline import preprocess,postprocess

def main():
    """Verify numerical and functional parity between PyTorch and ExecuTorch on real sample images.
    
    Loads both the eager PyTorch EdgeDetector and the exported ExecuTorch program, passes each
    sample JPEG in samples/, and verifies that:
    1. Raw bounding box outputs agree within absolute tolerance (atol=0.02, rtol=1e-4).
    2. Softmax classification scores agree within tolerance (atol=2e-5, rtol=1e-3).
    3. Postprocessed detection classes match identically.
    
    Results are saved to reports/real_image_parity.json.
    """
    torch.set_num_threads(2)
    model=EdgeDetector(load_detector()).eval()
    program=Runtime.get().load_program(ROOT/'models/crossguard_ssdlite_fp32.pte')
    method=program.load_method('forward'); results=[]
    with torch.inference_mode():
        for p in sorted((ROOT/'samples').glob('*.jpg')):
            bgr=cv2.imread(str(p)); x=torch.from_numpy(preprocess(bgr))
            reference=model(x); actual=method.execute([x])
            torch.testing.assert_close(actual[0],reference[0],atol=.02,rtol=1e-4)
            torch.testing.assert_close(actual[1],reference[1],atol=2e-5,rtol=1e-3)
            a=postprocess(actual[0].numpy(),actual[1].numpy(),bgr.shape[1],bgr.shape[0])
            b=postprocess(reference[0].numpy(),reference[1].numpy(),bgr.shape[1],bgr.shape[0])
            assert [d['class'] for d in a]==[d['class'] for d in b]
            results.append({'image':p.name,'max_box_abs_error':float((actual[0]-reference[0]).abs().max()),
                'max_score_abs_error':float((actual[1]-reference[1]).abs().max()),'detection_classes_match':True})
    (ROOT/'reports/real_image_parity.json').write_text(json.dumps({'passed':True,'samples':results,'note':'Parity only, no accuracy ground truth supplied.'},indent=2))
    print('PASS: ExecuTorch and PyTorch outputs agree on included images')

if __name__=='__main__': main()
