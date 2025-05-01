# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import numpy as np
import folium
from folium.plugins import HeatMap
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define dialect list to match the model's output classes
DIALECTS = [
    'Aleppo', 'Algiers', 'Alexandria', 'Amman', 'Aswan',
    'Baghdad', 'Basra', 'Beirut', 'Benghazi', 'Cairo',
    'Damascus', 'Doha', 'Fes', 'Jeddah', 'Jerusalem',
    'Khartoum', 'Mosul', 'MSA', 'Muscat', 'Rabat',
    'Riyadh', 'Salt', 'Sanaa', 'Sfax', 'Tripoli', 'Tunis'
]

class MADARFeatureLayer(nn.Module):
    """Layer to incorporate MADAR-specific dialect features."""
    def __init__(self, input_dim, lexicon_feature_dim=10):
        super().__init__()
        self.region_embeddings = nn.Embedding(7, lexicon_feature_dim)
        self.concept_weights = nn.Parameter(torch.randn(lexicon_feature_dim, input_dim))
        self.attention = nn.Linear(input_dim, 1)
        
    def forward(self, x, region_ids=None):
        batch_size, seq_len, hidden_dim = x.shape
        attention_scores = self.attention(x).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)
        if region_ids is not None:
            region_embed = self.region_embeddings(region_ids)
            dialect_bias = torch.matmul(region_embed, self.concept_weights)
            dialect_bias = dialect_bias.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + dialect_bias * 0.1
        context_vector = torch.matmul(attention_weights, x).squeeze(1)
        return context_vector

class MADARFeatureLayer(nn.Module):
    """Enhanced layer for MADAR-specific dialect features."""
    def __init__(self, input_dim, lexicon_feature_dim=32):
        super().__init__()
        
        # Embeddings for 7 major dialect regions
        self.region_embeddings = nn.Embedding(7, lexicon_feature_dim)
        
        # Separate weights for feature types
        self.phonological_weights = nn.Parameter(torch.randn(lexicon_feature_dim, input_dim // 3))
        self.lexical_weights = nn.Parameter(torch.randn(lexicon_feature_dim, input_dim // 3))
        self.morphological_weights = nn.Parameter(torch.randn(lexicon_feature_dim, input_dim // 3))
        
        # Attention mechanism
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Region-specific gates
        self.region_gates = nn.Parameter(torch.ones(7, 3) * 0.5)
        
    def forward(self, x, region_ids=None):
        """Apply MADAR-specific attention."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Self-attention mechanism
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        
        # If region info is available, apply dialect-specific biases
        if region_ids is not None:
            region_embed = self.region_embeddings(region_ids)
            
            # Split input for specialized feature processing
            x_chunks = torch.chunk(x, 3, dim=2)
            dialect_features = []
            
            # Process feature types separately with region weights
            for i, (chunk, weights) in enumerate(zip(x_chunks, 
                                                    [self.phonological_weights, 
                                                     self.lexical_weights, 
                                                     self.morphological_weights])):
                # Calculate feature bias
                feature_bias = torch.matmul(region_embed, weights)
                
                # Apply region gates to control feature impact
                gates = F.sigmoid(self.region_gates[region_ids, i]).unsqueeze(-1)
                
                # Expand and add bias
                feature_bias = feature_bias.unsqueeze(1).expand(-1, seq_len, -1)
                chunk_with_bias = chunk + feature_bias * gates
                dialect_features.append(chunk_with_bias)
            
            # Recombine features
            x_with_dialect = torch.cat(dialect_features, dim=2)
            
            # Combine attention and dialect features
            combined = (attended_values + x_with_dialect) / 2
            transformed = self.feature_transform(combined)
        else:
            transformed = self.feature_transform(attended_values)
        
        # Layer normalization for stability
        normalized = self.layer_norm(transformed + x)  # Residual connection
        
        # Pool across sequence to get final representation
        pooled = normalized.mean(dim=1)
        
        return pooled

class BiLSTMCNNAttention(nn.Module):
    """BiLSTM-CNN model with multi-head attention and residuals."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 embedding_matrix=None, dropout=0.3, num_filters=128):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if dropout > 0 else 0
        )
        
        # Multi-scale CNN layers
        self.conv1 = nn.Conv1d(hidden_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, num_filters, kernel_size=7, padding=3)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters * 3, 
            num_heads=8, 
            dropout=dropout if dropout > 0 else 0
        )
        
        # MADAR-specific feature layer
        self.madar_layer = MADARFeatureLayer(num_filters * 3)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(num_filters * 3)
        self.layer_norm2 = nn.LayerNorm(num_filters * 3)
        
        # Output layers
        self.fc1 = nn.Linear(num_filters * 3, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(256)
        
    def forward(self, x, region_ids=None):
        # x: [batch_size, seq_len]
        
        # Embedding
        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # Reshape for CNN
        cnn_in = lstm_out.permute(0, 2, 1)
        # cnn_in: [batch_size, hidden_dim, seq_len]
        
        # Multi-scale CNN
        conv1_out = F.relu(self.conv1(cnn_in))
        conv2_out = F.relu(self.conv2(cnn_in))
        conv3_out = F.relu(self.conv3(cnn_in))
        # conv_out: [batch_size, num_filters, seq_len]
        
        # Max pooling
        pooled1 = F.adaptive_max_pool1d(conv1_out, 1).squeeze(-1)
        pooled2 = F.adaptive_max_pool1d(conv2_out, 1).squeeze(-1)
        pooled3 = F.adaptive_max_pool1d(conv3_out, 1).squeeze(-1)
        # pooled: [batch_size, num_filters]
        
        # Concatenate features
        concat = torch.cat((pooled1, pooled2, pooled3), dim=1)
        # concat: [batch_size, num_filters * 3]
        
        # Self-attention (reshape)
        concat_reshaped = concat.unsqueeze(0)
        attn_out, _ = self.attention(concat_reshaped, concat_reshaped, concat_reshaped)
        attn_out = attn_out.squeeze(0)
        
        # Apply MADAR features if region IDs are available
        if region_ids is not None:
            madar_out = self.madar_layer(attn_out.unsqueeze(1), region_ids)
            attn_out = attn_out + madar_out  # Residual connection
        
        # Residual connection and layer normalization
        attn_out = self.layer_norm1(concat + attn_out)
        
        # Output layers
        fc1_out = self.fc1(attn_out)
        fc1_out = F.relu(self.bn(fc1_out))
        fc1_out = self.dropout(fc1_out)
        
        # Final classification
        out = self.fc2(fc1_out)
        
        return out

def load_models():
    try:
        import torch.serialization
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

        # Load transformer model
        transformer_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT", num_labels=len(DIALECTS))
        checkpoint = torch.load(
            "C:\\Users\\bechi\\Downloads\\best_transformer_model (1).pt",
            map_location=torch.device('cpu'),
            weights_only=False
        )
        transformer_model.load_state_dict(checkpoint["model_state_dict"])
        transformer_model.eval()
        
        vocab_size = 58317
        embedding_dim = 300
        hidden_dim = 128
        output_dim = 26

        # Create model with the same parameters as during training
        bilstm_model = BiLSTMCNNAttention(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            dropout=0,
            num_filters=128,
        )

        # Load compatible weights (ignore missing/extra keys)
        checkpoint = torch.load(
            "C:\\Users\\bechi\\Downloads\\best_bilstm_cnn_model (1).pt", 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        bilstm_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        bilstm_model.eval()
     
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Using simulated predictions for demonstration purposes.")
        bilstm_model = None
        transformer_model = None
    
    return bilstm_model, transformer_model

def predict_dialect(text, bilstm_model=None, transformer_model=None, model_choice='ensemble'):
    """Predict dialect probabilities."""
    # If no models loaded, use random predictions
    if bilstm_model is None and transformer_model is None:
        print("No models available, using random predictions")
        probs = {}
        random_probs = np.random.dirichlet(np.ones(len(DIALECTS))*0.5, size=1)[0]
        for i, dialect in enumerate(DIALECTS):
            probs[dialect] = float(random_probs[i])
        return probs
    
    # Decide which model(s) to use
    use_transformer = (transformer_model is not None) and (model_choice in ['transformer', 'ensemble'])
    use_bilstm = (bilstm_model is not None) and (model_choice in ['bilstm', 'ensemble'])
    
    # Fallback if selected model isn't available
    if model_choice == 'transformer' and transformer_model is None and bilstm_model is not None:
        print("Transformer model not available, using BiLSTM instead")
        use_bilstm = True
    elif model_choice == 'bilstm' and bilstm_model is None and transformer_model is not None:
        print("BiLSTM model not available, using transformer instead")
        use_transformer = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Use transformer if available and selected
    if use_transformer:
        try:
            tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                transformer_model = transformer_model.to(device)
                outputs = transformer_model(**inputs)
            transformer_probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
            for i, dialect in enumerate(DIALECTS):
                results[dialect] = float(transformer_probs[i])
            print("Used transformer model for prediction")
        except Exception as e:
            print(f"Error using transformer model: {e}")
            use_transformer = False
    
    # Use BiLSTM if available and selected
    if use_bilstm:
        try:
            from preprocessing import tokenize_for_bilstm
            tokens = tokenize_for_bilstm(text)
            token_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                bilstm_model = bilstm_model.to(device)
                outputs = bilstm_model(token_tensor)
            bilstm_probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            if results and model_choice == 'ensemble':
                # Average probabilities for ensemble
                for i, dialect in enumerate(DIALECTS):
                    results[dialect] = (results[dialect] + float(bilstm_probs[i])) / 2.0
                print("Used ensemble (transformer + BiLSTM) for prediction")
            else:
                # Use BiLSTM probabilities directly
                for i, dialect in enumerate(DIALECTS):
                    results[dialect] = float(bilstm_probs[i])
                print("Used BiLSTM-CNN model for prediction")
        except Exception as e:
            print(f"Error using BiLSTM model: {e}")
            if not results:
                random_probs = np.random.dirichlet(np.ones(len(DIALECTS))*0.5, size=1)[0]
                for i, dialect in enumerate(DIALECTS):
                    results[dialect] = float(random_probs[i])
    
    # Fallback to random if both models fail
    total = sum(results.values())
    if total > 0:
        # Normalize probabilities
        results = {k: v/total for k, v in results.items()}
        
    return results

# Initialize Flask app
app = Flask(__name__)

# Import preprocessing class
from preprocessing import MADARPreprocessor

# Initialize preprocessor
preprocessor = MADARPreprocessor()
# Models are loaded later in main()
model_bilstm = None
model_transformer = None

# Define dialect city coordinates
dialect_coordinates = {
    'Cairo': {'lat': 30.0444, 'lon': 31.2357, 'region': 'EGY'},
    'Alexandria': {'lat': 31.2001, 'lon': 29.9187, 'region': 'EGY'},
    'Aswan': {'lat': 24.0889, 'lon': 32.8998, 'region': 'EGY'},
    'Beirut': {'lat': 33.8938, 'lon': 35.5018, 'region': 'LEV'},
    'Damascus': {'lat': 33.5138, 'lon': 36.2765, 'region': 'LEV'},
    'Aleppo': {'lat': 36.2021, 'lon': 37.1343, 'region': 'LEV'},
    'Jerusalem': {'lat': 31.7683, 'lon': 35.2137, 'region': 'LEV'},
    'Amman': {'lat': 31.9454, 'lon': 35.9284, 'region': 'LEV'},
    'Salt': {'lat': 32.0387, 'lon': 35.7272, 'region': 'LEV'},
    'Doha': {'lat': 25.2854, 'lon': 51.5310, 'region': 'GLF'},
    'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'region': 'GLF'},
    'Jeddah': {'lat': 21.4858, 'lon': 39.1925, 'region': 'GLF'},
    'Muscat': {'lat': 23.5880, 'lon': 58.3829, 'region': 'GLF'},
    'Rabat': {'lat': 33.9716, 'lon': -6.8498, 'region': 'MGR'},
    'Fes': {'lat': 34.0181, 'lon': -5.0078, 'region': 'MGR'},
    'Tripoli': {'lat': 32.8872, 'lon': 13.1913, 'region': 'MGR'},
    'Tunis': {'lat': 36.8065, 'lon': 10.1815, 'region': 'MGR'},
    'Sfax': {'lat': 34.7398, 'lon': 10.7600, 'region': 'MGR'},
    'Algiers': {'lat': 36.7538, 'lon': 3.0588, 'region': 'MGR'},
    'Baghdad': {'lat': 33.3152, 'lon': 44.3661, 'region': 'IRQ'},
    'Basra': {'lat': 30.5085, 'lon': 47.7804, 'region': 'IRQ'},
    'Mosul': {'lat': 36.3350, 'lon': 43.1189, 'region': 'IRQ'},
    'Khartoum': {'lat': 15.5007, 'lon': 32.5599, 'region': 'SDN'},
    'Sanaa': {'lat': 15.3694, 'lon': 44.1910, 'region': 'YEM'},
    'Benghazi': {'lat': 32.1267, 'lon': 20.0830, 'region': 'MGR'}
}

# Region colors for visualization
region_colors = {
    'EGY': '#FF5733',  # Egyptian - Orange-Red
    'LEV': '#33FF57',  # Levantine - Green
  'GLF': '#3357FF',  # Gulf - Blue
'MGR': '#FFCC33',  # Maghrebi - Yellow
    'IRQ': '#FF33CC',  # Iraqi - Pink
   'SDN': '#33CCFF',  # Sudanese - Light Blue
     'YEM': '#CC33FF'   # Yemeni - Purple
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    map_type = request.form.get('map_type', 'heat')
    model_choice = request.form.get('model_choice', 'ensemble')
    
    # Preprocess text
    preprocessed_text = preprocessor.preprocess(text, embedding_model='ARABERT')
    
    # Make predictions
    dialect_probs = predict_dialect(
        preprocessed_text, 
        model_bilstm, 
        model_transformer, 
        model_choice=model_choice
    )
    
    # Generate map
    if map_type == 'heat':
        map_html = generate_heatmap(dialect_probs)
    else:
        map_html = generate_pointmap(dialect_probs)
    
    # Get sorted predictions
    all_dialects = sorted(dialect_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Determine which model was actually used (or if there was an error)
    model_used = model_choice
    if model_choice == 'transformer':
        if model_transformer is None:
            model_used = 'error (transformer model not available)'
    elif model_choice == 'bilstm':
        if model_bilstm is None:
            model_used = 'error (bilstm model not available)'
    elif model_choice == 'ensemble':
        if model_bilstm is None or model_transformer is None:
            if model_bilstm is None and model_transformer is None:
                model_used = 'error (no models available for ensemble)'
            elif model_bilstm is None:
                model_used = 'error (bilstm model required for ensemble)'
            else:
                model_used = 'error (transformer model required for ensemble)'
    else:
        model_used = 'error (invalid model selection)'
    
    return jsonify({
        'map_html': map_html,
        'top_predictions': [
            {'dialect': dialect, 'probability': f"{prob*100:.2f}%"} 
            for dialect, prob in all_dialects[:10]
        ],
        'model_used': model_used
    })

def generate_heatmap(dialect_probs):
    """Generate a heatmap of dialect probabilities."""
    # Create map centered for wide view
    m = folium.Map(
        location=[20.0, 0.0],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    # Sort dialects by probability
    sorted_dialects = sorted(dialect_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare heatmap data (only show >1% probability)
    heat_data = []
    for dialect, prob in sorted_dialects:
        if dialect in dialect_coordinates and prob > 0.01:
            coords = dialect_coordinates[dialect]
            weight = float(prob) * 20
            heat_data.append([float(coords['lat']), float(coords['lon']), weight])

    # Add heatmap layer
    if heat_data:
        HeatMap(
            heat_data,
            radius=40,
            blur=25,
            gradient={"0.4": 'blue', "0.65": 'lime', "0.8": 'yellow', "1.0": 'red'},
            max_opacity=0.8
        ).add_to(m)
    
    # Add markers for top dialects
    for dialect, prob in sorted_dialects:
        if dialect in dialect_coordinates:
            coords = dialect_coordinates[dialect]
            folium.Marker(
                location=[float(coords['lat']), float(coords['lon'])],
                popup=f"{dialect}: {float(prob)*100:.2f}%",
                tooltip=dialect
            ).add_to(m)

    # Save to temporary file
    map_path = os.path.join('static', 'temp_heatmap.html')
    os.makedirs('static', exist_ok=True)
    m.save(map_path)
    
    # Return URL
    return f'/static/temp_heatmap.html'

def generate_pointmap(dialect_probs):
    """Generate a point map with sized circles."""
    # Create map
    m = folium.Map(
        location=[20.0, 0.0],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    # Sort dialects by probability
    sorted_dialects = sorted(dialect_probs.items(), key=lambda x: x[1], reverse=True)

    # Add points for dialects > 0.5% probability
    for dialect, prob in sorted_dialects:
        if dialect in dialect_coordinates and prob > 0.005:
            coords = dialect_coordinates[dialect]
            region = str(coords.get('region', 'Unknown'))
            color = region_colors.get(region, '#000000')
            
            # Scale radius based on probability
            radius = float(prob) * 60
            
            # Add circle marker
            folium.CircleMarker(
                location=[float(coords['lat']), float(coords['lon'])],
                radius=radius,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"{dialect}: {float(prob)*100:.2f}%",
                tooltip=dialect
            ).add_to(m)
    
    # Save to temporary file
    map_path = os.path.join('static', 'temp_pointmap.html')
    os.makedirs('static', exist_ok=True)
    m.save(map_path)
    
    # Return URL
    return f'/static/temp_pointmap.html'

def main():
    # Load models before starting the app
    global model_bilstm, model_transformer
    model_bilstm, model_transformer = load_models()
    
    # Check if models loaded successfully
    if model_bilstm is None and model_transformer is None:
        print("Warning: No models available. App will use random predictions.")
    
    os.makedirs('templates', exist_ok=True)
    # Start Flask app
    app.run(debug=True)

if __name__ == '__main__':
    main()