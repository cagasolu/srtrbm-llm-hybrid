import sys
import cv2
import math
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.first_convolution = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.first_normalization = nn.BatchNorm2d(out_ch)

        self.second_convolution = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.second_normalization = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.residual_projection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual_projection = nn.Identity()

    def forward(self, xerox):
        identity = self.residual_projection(xerox)

        out = self.first_convolution(xerox)
        out = self.first_normalization(out)
        out = F.relu(out)

        out = self.second_convolution(out)
        out = self.second_normalization(out)

        out += identity
        out = F.relu(out)

        return out


class MNISTEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_residual_stage = ResidualBlock(1, 32, stride=1)
        self.second_residual_stage = ResidualBlock(32, 64, stride=2)
        self.third_residual_stage = ResidualBlock(64, 128, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_projection_layer = nn.Linear(128, 256)
        self.classification_layer = nn.Linear(256, 10)

    def forward(self, xcode):
        xcode = self.first_residual_stage(xcode)
        xcode = self.second_residual_stage(xcode)
        xcode = self.third_residual_stage(xcode)

        xcode = self.global_pool(xcode)
        xcode = torch.flatten(xcode, 1)

        embedding = F.normalize(self.embedding_projection_layer(xcode), dim=1)
        logit = self.classification_layer(embedding)

        return logit


model = MNISTEmbeddingModel().to(device)

ckpt = torch.load("zeta_mnist_hybrid.pt", map_location=device, weights_only=False)

if "model_state" in ckpt:
    raw_state = ckpt["model_state"]
else:
    raw_state = ckpt

mapped = {}

for k, v in raw_state.items():
    nk = k

    nk = nk.replace("block1", "first_residual_stage")
    nk = nk.replace("block2", "second_residual_stage")
    nk = nk.replace("block3", "third_residual_stage")

    nk = nk.replace("conv1", "first_convolution")
    nk = nk.replace("conv2", "second_convolution")

    nk = nk.replace("bn1", "first_normalization")
    nk = nk.replace("bn2", "second_normalization")

    nk = nk.replace("shortcut", "residual_projection")

    nk = nk.replace("embedding", "embedding_projection_layer")
    nk = nk.replace("fc", "classification_layer")

    nk = nk.replace("fc1", "embedding_projection_layer")
    nk = nk.replace("fc2", "classification_layer")

    mapped[nk] = v

model.load_state_dict(mapped, strict=True)
model.eval()

data = torch.load("stan.dgts", map_location="cpu")
images = data["images"].float()
labels = data["labels"]

if images.max() > 1:
    images /= 255.0

reference_bank = []

SAMPLES_PER_CLASS = 1

for d in range(10):
    cls = images[labels == d]

    center = cls.mean(dim=0, keepdim=True)
    dists = ((cls - center) ** 2).mean(dim=(1, 2))

    best = torch.argsort(dists)[:SAMPLES_PER_CLASS]
    reference_bank.append(cls[best].to(device))

print(", ↘\n")


def normalize_digit(patchX):
    patchX = torch.tensor(patchX).float()

    thr = patchX.mean()
    maskY = patchX > thr

    if maskY.sum() == 0:
        return patchX.numpy()

    coordinates0 = maskY.nonzero()
    y0, x0 = coordinates0.min(dim=0).values
    y1, x1 = coordinates0.max(dim=0).values + 1

    digits = patchX[y0:y1, x0:x1]

    digits = digits.unsqueeze(0).unsqueeze(0)
    digits = F.interpolate(digits, size=(20, 20), mode='bilinear', align_corners=False)
    digits = digits.squeeze()

    canvasX = torch.zeros(28, 28)
    canvasX[4:24, 4:24] = digits

    return canvasX.numpy()


def process_image(input_paths, output_paths):
    image = Image.open(input_paths).convert("L")
    image = np.array(image).astype(np.float32) / 255.0

    patch_size = 28
    step = 30

    patches = []
    coordinates = []

    for y1 in range(0, image.shape[0] - patch_size + 1, step):
        for x1 in range(0, image.shape[1] - patch_size + 1, step):

            patch = image[y1:y1 + 28, x1:x1 + 28]

            if patch.mean() < 0.01:
                continue

            patch = normalize_digit(patch)

            patches.append(patch)
            coordinates.append((y1, x1))

    if len(patches) == 0:
        print(f"{input_paths} → No valid patches")
        return

    patches = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
    print(f"{input_paths} → Patches:", patches.shape)

    with torch.no_grad():
        logits_variable = model(patches)
        predictions = torch.argmax(logits_variable, dim=1)

    canvas = np.zeros(image.shape, dtype=np.float32)
    mse_values = []

    for integer, (y1, x1) in enumerate(coordinates):
        digit = predictions[integer].item()
        bank = reference_bank[digit]

        patch = patches[integer].squeeze(0)

        distance = ((patch.unsqueeze(0) - bank) ** 2).mean(dim=(1, 2))
        best_idx = torch.argmin(distance)

        best_img = bank[best_idx].cpu().numpy()

        _, bw = cv2.threshold(best_img, 0.2, 1.0, cv2.THRESH_BINARY)

        num_labels, elastic, stats, _ = cv2.connectedComponentsWithStats((bw * 255).astype(np.uint8))

        clean = np.zeros_like(best_img)

        for into in range(1, num_labels):
            if stats[into, cv2.CC_STAT_AREA] > 20:
                clean[elastic == into] = 1.0

        best_img = clean
        original = patch.cpu().numpy()

        mse = ((original - best_img) ** 2).mean()
        mse_values.append(mse)

        canvas[y1:y1 + 28, x1:x1 + 28] = best_img

    print(f"{input_paths} → Avg MSE:", sum(mse_values) / len(mse_values))

    output = (canvas * 255).astype(np.uint8)
    Image.fromarray(output).save(output_paths)

    print(f"Saved: {output_paths}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 cluster.py <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]

    output_path = sys.argv[2]

    process_image(input_path, output_path)
