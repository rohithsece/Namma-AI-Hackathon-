import torch
from crossguard.model import EdgeDetector,load_detector
from crossguard.pipeline import postprocess

def test_wrapper_matches_torchvision_decoded_outputs():
    """Verify that EdgeDetector wrapper precisely matches eager torchvision detector decoding.
    
    Validates that:
    1. Input tensor normalization matches torchvision's internal transformation ((x - 0.5) / 0.5).
    2. Decoded box predictions match anchor generation and single-box decoding.
    3. Selected class softmax probabilities match slicing from the 91-class output.
    """
    torch.set_num_threads(2); torch.manual_seed(9)
    model=load_detector(); edge=EdgeDetector(model).eval()
    image=torch.rand(3,320,320)
    with torch.inference_mode():
        raw=edge(image[None]); transformed,_=model.transform([image],None)
        torch.testing.assert_close(transformed.tensors,(image[None]-.5)/.5)
        features=list(model.backbone(transformed.tensors).values()); head=model.head(features)
        anchors=model.anchor_generator(transformed,features)
        boxes=model.box_coder.decode_single(head['bbox_regression'][0],anchors[0]).clamp(0,320)
        torch.testing.assert_close(raw[0],boxes)
        probs=head['cls_logits'][0].softmax(-1)[:,[1,2,3,4,6,8]]
        torch.testing.assert_close(raw[1],probs)
