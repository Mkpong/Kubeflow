import torch
import json

def handle(data, context):
    model = context.get_model()

    # 입력 파싱
    inputs = data[0].get("data") or data[0].get("body") or data[0]
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    tensor = torch.tensor(inputs)  # numpy-like float list
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # [C,H,W] → [1,C,H,W]

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    # 결과 반환
    return output.tolist()

