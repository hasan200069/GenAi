import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json
import torch
from torch_geometric.data import Dataset, Data, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

model1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model1.eval()  

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class SceneGraphDataset(Dataset):
    def __init__(self, json_file, image_folder, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = list(self.data.keys())

        self.object_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        all_objects = []
        all_relations = []
        for image_name, image_data in self.data.items():
            for object_id, obj_data in image_data.get('objects', {}).items():
                all_objects.append(object_id)
                if 'relations' in obj_data:
                    for relation_data in obj_data['relations']:
                        all_relations.append(relation_data['name'])

        self.object_encoder.fit(all_objects)
        self.relation_encoder.fit(all_relations)

        print(f"Number of unique relations: {len(self.relation_encoder.classes_)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_data = self.data[image_name]

        relations = []
        object_features = []
        relation_pairs = []

        objects = []
        object_id_map = {}

        for object_id, obj_data in image_data.get('objects', {}).items():
            object_name = obj_data['name']
            x, y, w, h = obj_data['x'], obj_data['y'], obj_data['w'], obj_data['h']
            attrs = obj_data.get('attributes', [])

            object_features.append([x, y, w, h] + 
                                   [self.object_encoder.transform([object_id])[0]] + 
                                   self.encode_attributes(attrs) +
                                   [0] * (16 - 8))
            objects.append(object_id)
            object_id_map[object_id] = len(objects) - 1

            if 'relations' in obj_data:
                for relation_data in obj_data['relations']:
                    relation_name = relation_data['name']
                    related_object = relation_data['object']
                    relations.append(self.relation_encoder.transform([relation_name])[0])
                    object1_idx = object_id_map[object_id]
                    object2_idx = object_id_map.get(related_object)
                    if object2_idx is not None:
                        relation_pairs.append((object1_idx, object2_idx))

        object_features = torch.tensor(object_features, dtype=torch.float32)
        relations = torch.tensor(relations, dtype=torch.long)

        if len(relation_pairs) == 0:
            relation_pairs = torch.empty(0, 2, dtype=torch.long)

        if self.transform:
            image_path = f"{self.image_folder}/{image_name}"
            image = Image.open(image_path+'.jpg').convert("RGB")
            image = self.transform(image)
        self.visualize_bounding_boxes(image_name)    
        data = Data(
            x=object_features,
            edge_index=torch.tensor(relation_pairs, dtype=torch.long).t().contiguous(),
            y=relations,
            image_name=image_name
        )

        return data

    def encode_attributes(self, attributes):
        return [1 if attr in attributes else 0 for attr in ['brown', 'wooden', 'small']]

    def visualize_bounding_boxes(self, image_name):
        image_path = f"{self.image_folder}/{image_name}.jpg"
        img = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([transforms.ToTensor()])
        
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model1(img_tensor)

        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                x_min, y_min, x_max, y_max = box.tolist()
                label_name = COCO_CLASSES[label.item()]
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min - 10), f"{label_name} ({score:.2f})", font=font, fill="red")

        plt.imshow(img)
        plt.title(f"Bounding Boxes and Classified Objects for {image_name}")
        plt.axis('off')
        plt.show()


class SceneGraphModel(nn.Module):
    def __init__(self, num_relation_classes):
        super(SceneGraphModel, self).__init__()
        self.conv1 = SAGEConv(16, 64)
        self.conv2 = SAGEConv(64, 128)

        self.object_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        self.relation_fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, num_relation_classes),
        )

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()

    def forward(self, data):
        object_features = data.x
        relations = data.y
        relation_pairs = data.edge_index

        x = self.conv1(object_features, relation_pairs)
        x = torch.relu(x)

        x = self.conv2(x, relation_pairs)
        x = torch.relu(x)

        relation_outputs = []
        relation_labels = []

        for i in range(relation_pairs.shape[1]):
            object1_idx, object2_idx = relation_pairs[0, i], relation_pairs[1, i]
            object1_features = x[object1_idx]
            object2_features = x[object2_idx]

            combined_features = torch.cat([object1_features, object2_features], dim=-1)

            object_output = self.object_fc(combined_features)

            relation_output = self.relation_fc(object_output)

            relation_outputs.append(relation_output)
            relation_labels.append(relations[i])

        relation_outputs = torch.stack(relation_outputs)
        relation_labels = torch.stack(relation_labels)

        return relation_outputs, relation_labels

    def extract_resnet_features(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.resnet(image)
        return features


def train_model(model, dataloader, epochs=10, plot_interval=20, device='cpu'):
    print("Training Started")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    loss_history = []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(dataloader):
            optimizer.zero_grad()

            data = data.to(device)

            outputs, labels = model(data)
            predicted_relations = torch.argmax(outputs, dim=-1)

            predicted_relation_labels = [dataset.relation_encoder.inverse_transform([pred.item()])[0] for pred in predicted_relations]

            print(f"Sample [{step+1}/{len(dataloader)}]:")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            loss_history.append(loss.item())

            if (step + 1) % plot_interval == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'Training Loss after {step+1} steps')
                plt.legend()
                plt.show()

        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {running_loss/len(dataloader):.4f}")
        

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Final Training Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, dataloader, device='cpu', metrics=None):
    print("Evaluation started")
    model.eval()
    model.to(device)

    all_true_relations = []
    all_predicted_relations = []
    all_probabilities = []
    
    if metrics is None:
        metrics = ['accuracy']

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            outputs, labels = model(data)

            predicted_relations = torch.argmax(outputs, dim=-1)
            all_true_relations.extend(labels.cpu().numpy())
            all_predicted_relations.extend(predicted_relations.cpu().numpy())
            
            all_probabilities.append(torch.softmax(outputs, dim=-1).cpu().numpy())

    all_true_relations = np.array(all_true_relations)
    all_predicted_relations = np.array(all_predicted_relations)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    results = {}
    
    if 'accuracy' in metrics:
        accuracy = accuracy_score(all_true_relations, all_predicted_relations)
        results['accuracy'] = accuracy * 100

    if 'precision' in metrics:
        precision = precision_score(all_true_relations, all_predicted_relations, average='weighted')
        results['precision'] = precision

    if 'recall' in metrics:
        recall = recall_score(all_true_relations, all_predicted_relations, average='weighted')
        results['recall'] = recall

    if 'f1' in metrics:
        f1 = f1_score(all_true_relations, all_predicted_relations, average='weighted')
        results['f1'] = f1

    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(all_true_relations, all_predicted_relations)
        results['confusion_matrix'] = cm

    if 'average_probabilities' in metrics:
        avg_probs = np.mean(all_probabilities, axis=0)
        results['average_probabilities'] = avg_probs

    print("Evaluation Metrics:")
    for metric, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"{metric}: \n{value}")
        else:
            print(f"{metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SceneGraphDataset(json_file="val_sceneGraphs.json", image_folder="images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    num_relation_classes = len(dataset.relation_encoder.classes_)
    model = SceneGraphModel(num_relation_classes=num_relation_classes)

    train_model(model, dataloader, epochs=10)
