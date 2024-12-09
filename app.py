import tempfile
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from GCN import SceneGraphModel  

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SceneGraphModel(num_relation_classes=295)  
model.load_state_dict(torch.load('GCN.pth'))  
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/generate_scene_graph', methods=['POST'])
def generate_scene_graph():
    try:
        image_file = request.files['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            image_file.save(temp_file_path)
        scene_graph = model.generate_scene_graph(temp_file_path)
        os.remove(temp_file_path)

        return jsonify({"scene_graph": scene_graph})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
