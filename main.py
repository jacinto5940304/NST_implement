import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入影像函式（根據內容圖片尺寸調整）

# 定義VGG模型
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.selected_layers = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features

# Gram矩陣 (正確版)
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

# 內容損失
def get_content_loss(target, content):
    return torch.mean((target - content)**2)

# 風格損失
def get_style_loss(target, style):
    G = gram_matrix(target)
    S = gram_matrix(style)
    return torch.mean((G - S)**2)

# 最終影像存儲函數（修正後）
def save_img(tensor, path):
    tensor = tensor.cpu().clone().detach()
    tensor = tensor.squeeze(0)
    unloader = transforms.Compose([
        transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                             std=[4.367, 4.464, 4.444]),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    img = unloader(tensor)
    save_image(img, path)

# 載入圖片函式
def load_image(path, shape=None):
    image = Image.open(path).convert('RGB')
    if shape is None:
        shape = image.size[::-1] # (height, width)
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = transform(image).unsqueeze(0)
    return image.to(device)

content_img = load_image('content.jpg')
style_img = load_image('style.jpg', shape=content_img.shape[-2:])
generate_img = content_img.clone().requires_grad_(True)

# 模型
vgg = VGG().to(device).eval()

# 優化器
optimizer = optim.Adam([generate_img], lr=0.003)

# 權重係數
alpha = 1
beta = 1e6

steps = 5000
for step in tqdm(range(steps)):
    target_features = vgg(generate_img)
    content_features = vgg(content_img)
    style_features = vgg(style_img)

    c_loss = get_content_loss(target=target_features[-1], content=content_features[-1])

    s_loss = 0
    for t_feat, s_feat in zip(target_features, style_features):
        s_loss += get_style_loss(t_feat, s_feat)

    total_loss = alpha * c_loss + beta * s_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 每隔100步儲存一次影像
    if step % 100 == 0:
        print(f"Step {step}, Content loss: {c_loss.item()}, Style loss: {s_loss.item()}")
        temp_img = generate_img.clone().detach()
        save_img(temp_img, f"result_{step}.png")



save_img(generate_img, "final_result.png")
