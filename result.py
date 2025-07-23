import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import random

from train_cv import net50  

def batch_attention_vis(weight_path, fer2013_csv, device='cuda', n_samples=12):

    net = net50
    model = net.to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    
    df = pd.read_csv(fer2013_csv)
    label_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

  
    samples = []
    for label in range(len(label_names)):
        subset = df[df['emotion'] == label]
        if len(subset) == 0:
            print(f'Label {label_names[label]} 没有数据，跳过')
            continue
        row = subset.sample(n=1).iloc[0]
        samples.append((row, label))

   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples * 2.5))
    if n_samples == 1:
        axes = [axes]

    for idx, (row, label) in enumerate(samples):
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode='L').resize((224, 224))
        label_str = label_names[label]
        input_tensor = transform(img).unsqueeze(0).to(device)

        
        features = []
        def hook_fn(module, input, output):
            features.append(output.cpu().data.numpy())
        last_block = model[4][-1]
        conv = last_block.conv3
        handle = conv.register_forward_hook(hook_fn)

        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        handle.remove()
        feature_map = features[0][0]  # [C, H, W]

        # Grad-CAM
        params = list(model[-1].parameters())
        weight_softmax = params[0].cpu().data.numpy()
        class_weights = weight_softmax[pred]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(class_weights):
            cam += w * feature_map[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) > 0:
            cam /= np.max(cam)

        heatmap = cam  # [224,224], float
        img_np = np.array(img).astype(np.float32) / 255  # [224,224], float
        overlay = 0.5 * heatmap + 0.5 * img_np

        
        ax1, ax2, ax3 = axes[idx]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f'Label: {label_str}',fontsize=7)
        ax1.axis('off')

        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Grad-CAM')
        ax2.axis('off')

        ax3.imshow(overlay, cmap='jet')
        ax3.set_title(f'Pred: {label_names[pred]}',fontsize=7)
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


batch_attention_vis(
    'weights/model_net50_lr0.025.pth',
    'resnet/fer2013.csv',
    device='cuda',
    n_samples=7
)