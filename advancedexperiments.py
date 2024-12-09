import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Dataset, Data
from torchvision import transforms
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import json
import os
from PIL import Image


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
        data = Data(
            x=object_features,
            edge_index=torch.tensor(relation_pairs, dtype=torch.long).t().contiguous(),
            y=relations,
            image_name=image_name
        )
        return data

    def encode_attributes(self, attributes):
        return [1 if attr in attributes else 0 for attr in ['brown', 'wooden', 'small']]


class SceneGraphModelWithBN(nn.Module):
    def __init__(self, num_relation_classes, use_batch_norm=False):
        super(SceneGraphModelWithBN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = GCNConv(16, 64)
        self.conv2 = GCNConv(64, 128)
        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(64)
            self.batch_norm2 = nn.BatchNorm1d(128)
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

    def forward(self, data):
        object_features = data.x
        relations = data.y
        relation_pairs = data.edge_index
        x = self.conv1(object_features, relation_pairs)
        x = torch.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        x = self.conv2(x, relation_pairs)
        x = torch.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm2(x)
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


def train_model_with_regularization_and_techniques(model, dataloader, learning_rate, steps=100, device='cpu', early_stopping_patience=5, regularization_type=None, regularization_strength=0.0001, dropout_rate=0.5, grad_clip_value=1.0, checkpoint_dir="checkpoints"):
    print(f"Training with learning rate: {learning_rate}, Regularization: {regularization_type}, Gradient Clipping: {grad_clip_value}, Dropout Rate: {dropout_rate}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)

    model.to(device)
    model.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    loss_history = []
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for step, data in enumerate(dataloader):
        if step >= steps:
            break
        
        optimizer.zero_grad()
        data = data.to(device)
        outputs, labels = model(data)
        predicted_relations = torch.argmax(outputs, dim=-1)
        loss = criterion(outputs, labels)
        
        if grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        print(f"Step [{step+1}/{steps}], Loss: {loss.item():.4f}")
        
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at step {step+1}")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss with Regularization and Techniques')
    plt.legend()
    plt.show()


def run_regularization_and_techniques_experiment():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SceneGraphDataset(json_file="val_sceneGraphs.json", image_folder="images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    learning_rate = 0.0005
    regularization_types = ['l2', 'l1', None]
    dropout_rates = [0.5, 0.3]
    grad_clip_values = [1.0, 5.0]
    
    for regularization_type in regularization_types:
        for dropout_rate in dropout_rates:
            for grad_clip_value in grad_clip_values:
                model = SceneGraphModelWithBN(num_relation_classes=len(dataset.relation_encoder.classes_), use_batch_norm=False)
                train_model_with_regularization_and_techniques(
                    model, 
                    dataloader, 
                    learning_rate=learning_rate, 
                    steps=10,
                    early_stopping_patience=5,
                    regularization_type=regularization_type,
                    regularization_strength=0.0001,
                    dropout_rate=dropout_rate,
                    grad_clip_value=grad_clip_value,
                    checkpoint_dir="checkpoints"
                )
    
    print("Experiment Completed!")


if __name__ == "__main__":
    run_regularization_and_techniques_experiment()
