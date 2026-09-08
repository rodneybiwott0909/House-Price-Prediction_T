from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

print("=" * 60)
print("INITIALIZING FLASK APPLICATION")
print("=" * 60)

# Load the trained model
with open('./model/model_filename.pkl', 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully")

# Load the feature names
with open('./model/feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)
print("Feature names loaded successfully")


print("=" * 60)
print("PREDICTION ENDPOINT")
print("=" * 60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded!'
            }), 500
            
        # validate all features present
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing feature names: {missing}',
                'required': feature_names
            }), 400
            
        # Validate all values are number
        features = []
        for f in feature_names:
            try:
                features.append(float(data[f]))
            except (ValueError, TypeError):
                return jsonify({
                'success': False,
                'error': f'Invalid value for {f}. Must be a number',
            }), 400
                
        # predict
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
                'success': True,
                'input_features': {f: data[f] for f in feature_names},
                'predicted_price': round(prediction, 2),
                'predicted_price_formatted': f" Ksh. {prediction:,.2f}"
            }), 200
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    

if __name__ == '__main__':    
    app.run(debug=True)

    
                
            



