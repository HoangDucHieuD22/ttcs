!pip install stardist
# Upgrade numpy to a potentially compatible version
!pip install numpy==1.25.2
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
# Set a valid string value for image.interpolation, e.g., 'nearest'
matplotlib.rcParams["image.interpolation"] = 'nearest'
import matplotlib.pyplot as plt


from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
#Random color map labels
np.random.seed(42)
lbl_cmap = random_label_cmap()
!pip install xmltodict pillow
import os
import xmltodict
import numpy as np
from PIL import Image, ImageDraw
from glob import glob

# Thư mục chứa ảnh và annotation
image_dir = '/content/drive/MyDrive/source/Jpeg'
anno_dir = '/content/drive/MyDrive/source/annotations'
save_mask_dir = '/content/drive/MyDrive/source/mask'
os.makedirs(save_mask_dir, exist_ok=True)

# Map nhãn -> số
label_map = {"RBC": 1, "WBC": 2, "Platelets": 3}

xml_files = glob(os.path.join(anno_dir, '*.xml'))

for xml_path in xml_files:
    with open(xml_path) as f:
        doc = xmltodict.parse(f.read())

    filename = doc['annotation']['filename']
    size = doc['annotation']['size']
    width = int(size['width'])
    height = int(size['height'])

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    objects = doc['annotation'].get('object', [])
    if isinstance(objects, dict):
        objects = [objects]

    for obj in objects:
        label = obj['name']
        bbox = obj['bndbox']
        xmin = int(float(bbox['xmin']))
        ymin = int(float(bbox['ymin']))
        xmax = int(float(bbox['xmax']))
        ymax = int(float(bbox['ymax']))

        value = label_map.get(label, 0)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=value)

    mask.save(os.path.join(save_mask_dir, filename.replace('.jpg', '_mask.tif')))
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mask = np.array(Image.open('/content/drive/MyDrive/source/mask/BloodImage_00000_mask.tif'))
plt.imshow(mask, cmap='nipy_spectral')
plt.title('Mask Visualization')
plt.colorbar()
plt.show()
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Đường dẫn ảnh và mask cụ thể
image_path = '/content/drive/MyDrive/source/Jpeg/BloodImage_00000.jpg'
mask_path = '/content/drive/MyDrive/source/mask/BloodImage_00000_mask.tif'

# Ánh xạ giá trị mask -> màu RGB
label_colors = {
    0: [0, 0, 0],        # Nền - đen
    1: [255, 0, 0],      # RBC - đỏ
    2: [255, 255, 255],  # WBC - trắng
    3: [0, 0, 255],      # Platelet - xanh
}

# Load ảnh và mask
img = Image.open(image_path).convert('RGB')
mask = Image.open(mask_path).convert('L')
mask_array = np.array(mask)

# Tạo ảnh mask màu
color_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
for label_val, color in label_colors.items():
    color_mask[mask_array == label_val] = color

# Hiển thị
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(color_mask)
plt.title('Ảnh mask')
plt.axis('off')

plt.tight_layout()
plt.show()
import os
import numpy as np
from PIL import Image
from glob import glob
from scipy.ndimage import label, find_objects
import os
import numpy as np
from glob import glob
from PIL import Image
from stardist.models import StarDist2D, Config2D # Import Config2D
from csbdeep.utils import normalize
from sklearn.model_selection import train_test_split # Import train_test_split
import shutil # Import shutil for removing directory

# Đường dẫn tới dataset
image_dir = '/content/drive/MyDrive/source/Jpeg'  # Ảnh gốc
mask_dir = '/content/drive/MyDrive/source/mask'   # Mask phân vùng

def load_images_masks_by_order(image_dir, mask_dir):
    # Load tất cả ảnh và mask theo thứ tự tên file
    image_paths = sorted(glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.tif')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')) + glob(os.path.join(mask_dir, '*.tif')))

    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Số lượng ảnh ({len(image_paths)}) và mask ({len(mask_paths)}) không khớp!")

    images = []
    masks = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        img = Image.open(img_path)
        mask = Image.open(msk_path)

        # Nếu ảnh là RGB, chuyển về grayscale
        if img.mode == 'RGB':
            img = img.convert('L')

        img = np.array(img)
        mask = np.array(mask)

        # Normalize ảnh (0-1, dtype float32)
        img_norm = normalize(img, 1, 99.8)

        images.append(img_norm.astype(np.float32))
        masks.append(mask.astype(np.uint16))

    return np.array(images), np.array(masks)

print("Loading data (by order)...")
X, Y = load_images_masks_by_order(image_dir, mask_dir)
print(f"Loaded {len(X)} images and {len(Y)} masks")

# Thêm chiều kênh nếu cần
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# Split data into training and validation sets
# Using a 80/20 split, stratifying by mask labels to ensure all classes are represented in both sets
# Note: Stratification requires uniform number of labels per sample, which might not be the case for segmentation masks.
# A simpler split without stratification is often sufficient for StarDist training.
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)
# Let's use a simple random split without stratification for robustness
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)


print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")


# Define model directory
model_dir = os.path.join('/content/drive/MyDrive/source')




# Tạo model StarDist
conf = Config2D (
    # Add any custom configuration parameters here if needed
)
model_stardist = StarDist2D(conf, name='my_stardist_model', basedir='/content/drive/MyDrive/source')

print("Starting training...")
# Pass the validation data to the train method
model_stardist.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, steps_per_epoch=100)
print("Training done.")



# Đường dẫn thư mục
image_dir = '/content/drive/MyDrive/source/Jpeg'
mask_dir = '/content/drive/MyDrive/source/mask'
save_dir = '/content/drive/MyDrive/source/dataset'
os.makedirs(save_dir, exist_ok=True)

# Tạo thư mục cho từng loại tế bào
classes = {1: 'RBC', 2: 'WBC', 3: 'Platelet'}
for c in classes.values():
    os.makedirs(os.path.join(save_dir, c), exist_ok=True)

# Lặp qua từng file mask
for mask_path in glob(os.path.join(mask_dir, '*_mask.tif')):
    filename = os.path.basename(mask_path).replace('_mask.tif', '.jpg')
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        continue

    mask = np.array(Image.open(mask_path))
    image = Image.open(image_path)

    for label_id, class_name in classes.items():
        binary_mask = (mask == label_id).astype(np.uint8)
        labeled, num_features = label(binary_mask)
        boxes = find_objects(labeled)

        for i, box in enumerate(boxes):
            if box is None:
                continue

            y_slice, x_slice = box
            cropped = image.crop((x_slice.start, y_slice.start, x_slice.stop, y_slice.stop))

            # Resize nếu muốn
            cropped = cropped.resize((64, 64))  # hoặc 128x128

            save_path = os.path.join(save_dir, class_name, f"{filename.replace('.jpg','')}_{i}.png")
            cropped.save(save_path)

print("✅ Đã tách và lưu xong ảnh từng tế bào!")
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt

mask_path = '/content/drive/MyDrive/source/mask/BloodImage_00000_mask.tif'  # đổi thành file mask thật
mask = np.array(Image.open(mask_path))

plt.imshow(mask, cmap='nipy_spectral')
plt.title("Mask với nhãn")
plt.colorbar()
plt.show()

for label_id, name in {1: 'RBC', 2: 'WBC', 3: 'Platelet'}.items():
    binary = (mask == label_id).astype(np.uint8)
    labeled, n = label(binary)
    print(f"{name}: {n} vùng")
    import os

for cls in ['RBC', 'WBC', 'Platelet']:
    folder = f'/content/drive/MyDrive/source/dataset/{cls}'
    if not os.path.exists(folder):
        print(f"❌ Không tồn tại thư mục {folder}")
    else:
        files = os.listdir(folder)
        print(f"{cls}: {len(files)} ảnh được lưu")
import os
import shutil
import random
from glob import glob

# Thư mục gốc chứa ảnh đã crop theo class
source_dir = '/content/drive/MyDrive/source/dataset'
target_dir = '/content/drive/MyDrive/source/dataset_split'
train_ratio = 0.8

# Tên lớp tế bào
classes = ['RBC', 'WBC', 'Platelet']

for cls in classes:
    src_cls_dir = os.path.join(source_dir, cls)
    train_cls_dir = os.path.join(target_dir, 'train', cls)
    test_cls_dir = os.path.join(target_dir, 'test', cls)

    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    # Lấy tất cả ảnh trong lớp
    images = glob(os.path.join(src_cls_dir, '*.png'))
    random.shuffle(images)

    # Chia tập
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    # Sao chép ảnh
    for img in train_imgs:
        shutil.copy(img, train_cls_dir)
    for img in test_imgs:
        shutil.copy(img, test_cls_dir)

print("✅ Đã chia ảnh vào dataset_split/train và dataset_split/test theo tỷ lệ 80/20.")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đường dẫn thư mục dữ liệu đã chia
data_dir = '/content/drive/MyDrive/source/dataset_split'

# Các biến số
batch_size = 32
num_epochs = 10
num_classes = 3

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Tải dataset
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load ResNet18 pretrained
model = models.resnet18(pretrained=True)

# Sửa lớp FC cuối cùng
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Huấn luyện
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    acc = correct.double() / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Acc: {acc:.4f}")

# Lưu mô hình
torch.save(model.state_dict(), '/content/drive/MyDrive/source/resnet18_blood_cells.pth')

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()

print(f'Test Loss: {test_loss/len(test_loader):.4f}')
print(f'Test Accuracy: {correct/len(test_loader.dataset):.4f}')
