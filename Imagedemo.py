from final import ConvNet
import torch
from PIL import Image
from torchvision.transforms import v2 

#Convert to RGB for color picture, PNG might be RGBA
image = Image.open("demo_happy.jpg").convert('RGB')
#demo_happy.jpg
#demosad.png

#Test Augumentation to make sure output tensor
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((48, 48)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
    ])
    
#The class name correspond to class name in folder
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

img_tensor = test_transform(image)

model = ConvNet()
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()
with torch.no_grad():
    out = model(img_tensor.unsqueeze(0))
#out = model(img_tensor) # print nicely
_, predicted_idx = torch.max(out, 1)
predicted_label = class_names[predicted_idx.item()]

print(predicted_label)