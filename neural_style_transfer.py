import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess image
def load_image(img_path, size=512):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Show tensor as image
def show_image(tensor, title='Image'):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Load pre-trained VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Define loss layers
class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.selected_layers = ['0', '5', '10', '19', '28']
        self.model = vgg[:29]

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if str(name) in self.selected_layers:
                features.append(x)
        return features

# Content and style features
model = StyleTransfer().to(device)
target = content.clone().requires_grad_(True)
optimizer = optim.Adam([target], lr=0.003)
style_features = model(style)
content_features = model(content)

# Loss calculation
mse = nn.MSELoss()

for step in range(201):
    target_features = model(target)
    content_loss = mse(target_features[2], content_features[2])
    style_loss = 0
    for t_feat, s_feat in zip(target_features, style_features):
        _, c, h, w = t_feat.shape
        t_gram = torch.matmul(t_feat.view(c, -1), t_feat.view(c, -1).t())
        s_gram = torch.matmul(s_feat.view(c, -1), s_feat.view(c, -1).t())
        style_loss += mse(t_gram, s_gram)
    
    total_loss = content_loss + 0.01 * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss {total_loss.item():.4f}")
        show_image(target, title=f'Step {step}')

# Final output
show_image(target, title='Final Styled Image')
