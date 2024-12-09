import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import torch
from torch_geometric.data import Dataset, Data, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv

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

            object_features.append([x, y, w, h] + [self.object_encoder.transform([object_id])[0]] + self.encode_attributes(attrs))
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

        data = Data(
            x=object_features,
            edge_index=torch.tensor(relation_pairs, dtype=torch.long).t().contiguous(),
            y=relations,
            image_name=image_name
        )

        return data

    def encode_attributes(self, attributes):
        return [1 if attr in attributes else 0 for attr in ['brown', 'wooden', 'small']]

class SceneGraphModel(nn.Module):
    def __init__(self, num_relation_classes):
        super(SceneGraphModel, self).__init__()
        self.object_fc = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.relation_fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, num_relation_classes),
        )

    def forward(self, data):
        object_features = data.x
        relations = data.y
        relation_pairs = data.edge_index

        relation_outputs = []
        relation_labels = []

        for i in range(relation_pairs.shape[1]):
            object1_idx, object2_idx = relation_pairs[0, i], relation_pairs[1, i]
            object1_features = object_features[object1_idx]
            object2_features = object_features[object2_idx]
            combined_features = torch.cat([object1_features, object2_features], dim=-1)
            object_output = self.object_fc(combined_features)
            relation_output = self.relation_fc(object_output)
            relation_outputs.append(relation_output)
            relation_labels.append(relations[i])

        relation_outputs = torch.stack(relation_outputs)
        relation_labels = torch.stack(relation_labels)

        return relation_outputs, relation_labels

def train_model(model, dataloader, epochs=10, plot_interval=20):
    print("Training Started")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(dataloader):
            optimizer.zero_grad()
            outputs, labels = model(data)
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

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SceneGraphDataset(json_file="val_sceneGraphs.json", image_folder="images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_relation_classes = len(dataset.relation_encoder.classes_)
    model = SceneGraphModel(num_relation_classes=num_relation_classes)

    train_model(model, dataloader, epochs=10)
