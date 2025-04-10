import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import subprocess  # <-- .mar 만들 때 사용

# 1. Data 준비 (MNIST → 3채널로 변경)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST 클래스 10개
model = model.to(device)

# 3. 학습
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1):  # 간단히 1 epoch
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 4. TorchScript 저장 (.pt)
model.eval()
example_input = torch.rand(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)

# PVC 상위 디렉토리 (KServe에서 storageUri => /mnt/pvc/resnet-mnist → /mnt/models)
top_dir = "/mnt/pvc/resnet-mnist"

# model-store 디렉토리: .pt와 .mar를 둘 위치
model_store_dir = os.path.join(top_dir, "model-store")
os.makedirs(model_store_dir, exist_ok=True)

model_pt_path = os.path.join(model_store_dir, "model.pt")
traced_model.save(model_pt_path)
print(f"✅ TorchScript model saved: {model_pt_path}")

# 5. .mar 파일 생성 (torch-model-archiver)
archiver_cmd = [
    "torch-model-archiver",
    "--model-name", "resnet",
    "--version", "1.0",
    "--serialized-file", model_pt_path,
    "--handler", "image_classifier",
    "--export-path", model_store_dir  # resnet.mar 파일도 model-store 폴더에 생성
]
print("Running:", " ".join(archiver_cmd))
# cwd 지정으로 현재 디렉토리 문제 예방
subprocess.run(archiver_cmd, check=True, cwd=model_store_dir)
mar_path = os.path.join(model_store_dir, "resnet.mar")
print(f"✅ .mar file created: {mar_path}")

# 6. config.properties 생성 (상위 폴더에 "config" 디렉토리)
config_dir = os.path.join(top_dir, "config")
os.makedirs(config_dir, exist_ok=True)

config_path = os.path.join(config_dir, "config.properties")

with open(config_path, "w") as f:
    f.write("inference_address=http://0.0.0.0:8085\n")
    f.write("management_address=http://0.0.0.0:8086\n")
    f.write("metrics_address=http://0.0.0.0:8082\n")
    f.write("grpc_inference_port=7070\n")
    f.write("grpc_management_port=7071\n")
    f.write("model_store=/mnt/models/model-store\n")
    f.write('model_snapshot={"name":"startup.cfg","models":{"resnet":{"1.0":{"defaultVersion":true,"marName":"resnet.mar"}}}}\n')

print(f"✅ config.properties saved: {config_path}")
print("🎉 All done!")

