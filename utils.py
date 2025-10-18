# Install required packages
!pip install transformers torchvision pillow scikit-learn pandas numpy requests plotly seaborn

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io
import os

# Check GPU availability
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

class ProductPricePredictor:
    def _init_(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.text_tokenizer = None
        self.image_processor = None
        self.scaler = StandardScaler()
        self.multi_modal_model = None
        self.gb_model = None
        self.feature_columns = []
        self.is_trained = False
        
    def load_train_data(self, file_path=None, uploaded_file=None):
        """Load training data from file or uploaded content"""
        print("Loading training data...")
        if uploaded_file is not None:
            # Use uploaded file
            self.train_df = pd.read_csv(io.BytesIO(uploaded_file))
        elif file_path is not None:
            # Use file path
            self.train_df = pd.read_csv(file_path)
        else:
            raise ValueError("No training data provided")
        
        print(f"Training dataset loaded: {self.train_df.shape}")
        print(f"Columns: {self.train_df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['catalog_content', 'price']
        for col in required_columns:
            if col not in self.train_df.columns:
                raise ValueError(f"Required column '{col}' not found in training data")
        
        return self.train_df

    def load_test_data(self, file_path=None, uploaded_file=None):
        """Load test data from file or uploaded content"""
        print("Loading test data...")
        if uploaded_file is not None:
            # Use uploaded file
            self.test_df = pd.read_csv(io.BytesIO(uploaded_file))
        elif file_path is not None:
            # Use file path
            self.test_df = pd.read_csv(file_path)
        else:
            raise ValueError("No test data provided")
        
        print(f"Test dataset loaded: {self.test_df.shape}")
        print(f"Columns: {self.test_df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['catalog_content']
        for col in required_columns:
            if col not in self.test_df.columns:
                raise ValueError(f"Required column '{col}' not found in test data")
        
        return self.test_df

    def advanced_feature_engineering(self, df, is_training=True):
        """Extract comprehensive features from catalog content"""
        print("Performing advanced feature engineering...")
        
        def extract_features_from_catalog(catalog_text):
            features = {}
            
            if pd.isna(catalog_text):
                catalog_text = ""
            
            # Extract item name
            item_name_match = re.search(r'Item Name:\s*([^\n]+)', str(catalog_text))
            features['item_name'] = item_name_match.group(1).strip() if item_name_match else ""
            
            # Extract value
            value_match = re.search(r'Value:\s*([\d.]+)', str(catalog_text))
            features['value'] = float(value_match.group(1)) if value_match else 0.0
            
            # Extract unit
            unit_match = re.search(r'Unit:\s*([^\n]+)', str(catalog_text))
            features['unit'] = unit_match.group(1).strip() if unit_match else ""
            
            # Extract bullet points count
            bullet_points = re.findall(r'Bullet Point\s*\d+:', str(catalog_text))
            features['bullet_points_count'] = len(bullet_points)
            
            # Extract product description presence
            features['has_product_description'] = 1 if 'Product Description:' in str(catalog_text) else 0
            
            # Text length features
            features['catalog_length'] = len(str(catalog_text))
            features['item_name_length'] = len(features['item_name'])
            
            # Keyword features
            text_lower = str(catalog_text).lower()
            features['has_organic'] = 1 if 'organic' in text_lower else 0
            features['has_gluten_free'] = 1 if 'gluten-free' in text_lower or 'gluten free' in text_lower else 0
            features['has_vegan'] = 1 if 'vegan' in text_lower else 0
            features['has_natural'] = 1 if 'natural' in text_lower else 0
            features['has_kosher'] = 1 if 'kosher' in text_lower else 0
            features['has_non_gmo'] = 1 if 'non-gmo' in text_lower or 'non gmo' in text_lower else 0
            
            # Product category keywords
            categories = ['coffee', 'tea', 'chocolate', 'candy', 'snack', 'sauce', 'spice', 
                         'bean', 'grain', 'supplement', 'protein', 'beverage', 'oil', 'baking']
            for category in categories:
                features[f'category_{category}'] = 1 if category in text_lower else 0
            
            return features

        def extract_brand(item_name):
            common_brands = ['Starbucks', 'Trader Joe', 'Goya', 'Smuckers', 'Bush', 
                            'Planter', 'Arizona', 'Hershey', 'Keebler', 'Quaker',
                            'Kraft', 'Campbell', 'Nature', 'Organic', 'Simple Mills']
            item_name_str = str(item_name).lower()
            for brand in common_brands:
                if brand.lower() in item_name_str:
                    return brand
            return 'Other'

        def extract_package_info(text):
            pack_match = re.search(r'pack of\s*(\d+)', str(text).lower())
            count_match = re.search(r'(\d+)\s*count', str(text).lower())
            pack_size = int(pack_match.group(1)) if pack_match else 1
            item_count = int(count_match.group(1)) if count_match else 1
            return pack_size, item_count

        # Apply feature extraction
        catalog_features = df['catalog_content'].apply(extract_features_from_catalog)
        features_df = pd.DataFrame(catalog_features.tolist())
        
        # Combine with original data
        featured_df = pd.concat([df, features_df], axis=1)
        
        # Advanced features
        featured_df['brand'] = featured_df['item_name'].apply(extract_brand)
        featured_df['pack_size'] = featured_df['catalog_content'].apply(lambda x: extract_package_info(x)[0])
        featured_df['item_count'] = featured_df['catalog_content'].apply(lambda x: extract_package_info(x)[1])
        
        # Only calculate price_per_unit for training data (where price is available)
        if 'price' in featured_df.columns:
            featured_df['price_per_unit'] = featured_df['price'] / featured_df['value'].replace(0, 1)
        
        # Fill NaN values
        featured_df = featured_df.fillna(0)
        
        print(f"Feature engineering completed. Total features: {len(features_df.columns) + 4}")
        return featured_df

    class MultiModalProductDataset(Dataset):
        def _init_(self, df, text_model_name='microsoft/deberta-v3-base', 
                     image_model_name='microsoft/resnet-50', max_length=512, is_training=True):
            self.df = df.reset_index(drop=True)
            self.max_length = max_length
            self.is_training = is_training
            
            # Text processor
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            
            # Image processor - use simpler transform for Colab
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Numerical features
            self.numerical_features = [
                'value', 'bullet_points_count', 'has_product_description',
                'catalog_length', 'item_name_length', 'has_organic', 
                'has_gluten_free', 'has_vegan', 'has_natural', 'has_kosher', 
                'has_non_gmo', 'pack_size', 'item_count'
            ]
            
            # Initialize scaler only for training
            self.scaler = StandardScaler()
            numerical_data = self.df[self.numerical_features].fillna(0)
            if len(numerical_data) > 0 and is_training:
                self.scaler.fit(numerical_data)
        
        def _len_(self):
            return len(self.df)
        
        def _getitem_(self, idx):
            row = self.df.iloc[idx]
            
            # Text features
            text = f"{row.get('item_name', '')} {row['catalog_content']}"
            text_encoding = self.text_tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Image features - handle download errors gracefully
            try:
                response = requests.get(row['image_link'], timeout=5)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image_tensor = self.transform(image)
            except:
                # Create blank image if download fails
                image_tensor = torch.zeros(3, 224, 224)
            
            # Numerical features
            numerical_data = []
            for feature in self.numerical_features:
                numerical_data.append(row.get(feature, 0))
            
            numerical_data = np.array(numerical_data, dtype=float)
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                numerical_tensor = torch.FloatTensor(
                    self.scaler.transform([numerical_data])[0]
                )
            else:
                numerical_tensor = torch.FloatTensor(numerical_data)
            
            # Target (only for training)
            if self.is_training and 'price' in row:
                price = torch.FloatTensor([row['price']])
                return {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0),
                    'image': image_tensor,
                    'numerical': numerical_tensor,
                    'price': price
                }
            else:
                return {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0),
                    'image': image_tensor,
                    'numerical': numerical_tensor,
                    'sample_id': row.get('sample_id', idx)
                }

    class AdvancedMultiModalModel(nn.Module):
        def _init_(self, text_model_name='microsoft/deberta-v3-base',
                     image_model_name='microsoft/resnet-50',
                     numerical_dim=13, hidden_dim=256, dropout=0.3):
            super()._init_()
            
            # Text encoder - use smaller model for Colab
            self.text_encoder = AutoModel.from_pretrained('microsoft/deberta-v3-small')
            text_hidden_size = self.text_encoder.config.hidden_size
            self.text_projection = nn.Linear(text_hidden_size, hidden_dim)
            
            # Image encoder - use pretrained ResNet
            self.image_encoder = models.resnet18(pretrained=True)
            self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, hidden_dim)
            
            # Numerical feature processor
            self.numerical_encoder = nn.Sequential(
                nn.Linear(numerical_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, hidden_dim),
                nn.ReLU()
            )
            
            # Multi-modal fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
            
            # Regression head
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, input_ids, attention_mask, image, numerical):
            # Text encoding
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            text_features = self.text_projection(text_features)
            
            # Image encoding
            image_features = self.image_encoder(image)
            
            # Numerical encoding
            numerical_features = self.numerical_encoder(numerical)
            
            # Concatenate all features
            combined_features = torch.cat([text_features, image_features, numerical_features], dim=1)
            fused_features = self.fusion_layer(combined_features)
            
            # Final prediction
            price_pred = self.regressor(fused_features)
            
            return price_pred.squeeze(-1)

    def train_multi_modal_model(self, df, epochs=10, batch_size=4):
        """Train the multi-modal deep learning model"""
        print("Training multi-modal model...")
        
        # Use smaller subset for Colab
        sample_df = df.head(min(30, len(df))).copy()
        
        # Create dataset and dataloader
        dataset = self.MultiModalProductDataset(sample_df, is_training=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.multi_modal_model = self.AdvancedMultiModalModel().to(self.device)
        
        # Loss and optimizer
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.multi_modal_model.parameters(), lr=1e-4)
        
        # Training loop
        self.multi_modal_model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch in dataloader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                numerical = batch['numerical'].to(self.device)
                price = batch['price'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                pred_price = self.multi_modal_model(input_ids, attention_mask, image, numerical)
                loss = criterion(pred_price, price.squeeze())
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multi_modal_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)
                
                if epoch % 2 == 0:
                    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        print("Multi-modal model training completed!")
        
        # Plot training loss
        if losses:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title('Multi-modal Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        
        return losses

    def train_gradient_boosting(self, df):
        """Train traditional Gradient Boosting model"""
        print("Training Gradient Boosting model...")
        
        # Prepare features
        feature_columns = [
            'value', 'bullet_points_count', 'has_product_description',
            'catalog_length', 'item_name_length', 'has_organic', 'has_gluten_free',
            'has_vegan', 'has_natural', 'has_kosher', 'has_non_gmo',
            'pack_size', 'item_count', 'price_per_unit'
        ] + [col for col in df.columns if col.startswith('category_')]
        
        self.feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[self.feature_columns].fillna(0)
        y = df['price']
        
        # Train model
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.gb_model.fit(X, y)
        
        # Evaluate on training data
        y_pred = self.gb_model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"Gradient Boosting Training Results:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        # Plot feature importance
        if hasattr(plt, 'figure'):
            plt.subplot(1, 2, 2)
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.gb_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)  # Top 10 features
            
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Top 10 Feature Importance (Gradient Boosting)')
            plt.tight_layout()
            plt.show()
        
        return self.gb_model

    class LLMAnalyzer:
        """Simulated LLM analysis for product pricing"""
        def _init_(self):
            self.premium_keywords = ['organic', 'premium', 'gourmet', 'artisanal', 'imported', 
                                   'handcrafted', 'specialty', 'luxury', 'authentic']
            self.budget_keywords = ['value', 'economy', 'budget', 'affordable', 'discount', 'cheap']
            
        def analyze_product(self, catalog_content, item_name):
            """Analyze product using keyword-based heuristic (simulating LLM)"""
            text = f"{item_name} {catalog_content}".lower()
            
            # Calculate premium score
            premium_score = sum(1 for keyword in self.premium_keywords if keyword in text)
            budget_score = sum(1 for keyword in self.budget_keywords if keyword in text)
            
            # Base price calculation based on product type
            if 'coffee' in text:
                base_price = 12 + (premium_score * 6) - (budget_score * 3)
            elif 'chocolate' in text or 'candy' in text:
                base_price = 10 + (premium_score * 5) - (budget_score * 2)
            elif 'tea' in text:
                base_price = 8 + (premium_score * 4) - (budget_score * 2)
            elif 'protein' in text or 'supplement' in text:
                base_price = 15 + (premium_score * 8) - (budget_score * 3)
            elif 'oil' in text:
                base_price = 10 + (premium_score * 5) - (budget_score * 2)
            else:
                base_price = 15 + (premium_score * 7) - (budget_score * 3)
            
            # Adjust based on health/specialty claims
            if 'organic' in text:
                base_price *= 1.25
            if 'gluten-free' in text:
                base_price *= 1.15
            if 'vegan' in text:
                base_price *= 1.1
            if 'natural' in text:
                base_price *= 1.05
            
            confidence = min(0.9, 0.5 + (premium_score + budget_score) * 0.08)
            
            return {
                'suggested_price': max(1, base_price),
                'confidence': confidence,
                'premium_score': premium_score,
                'budget_score': budget_score
            }

    def predict_on_test_data(self, test_df):
        """Generate predictions for test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Generating predictions for test data...")
        
        # Feature engineering for test data
        test_featured_df = self.advanced_feature_engineering(test_df, is_training=False)
        
        predictions = []
        llm_analyzer = self.LLMAnalyzer()
        
        for idx, row in test_featured_df.iterrows():
            sample_id = row.get('sample_id', idx)
            
            # Get LLM analysis
            llm_result = llm_analyzer.analyze_product(
                row['catalog_content'],
                row.get('item_name', '')
            )
            
            # Get Gradient Boosting prediction
            if self.gb_model is not None:
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(row.get(col, 0))
                
                try:
                    gb_pred = self.gb_model.predict([feature_vector])[0]
                except:
                    gb_pred = llm_result['suggested_price']
            else:
                gb_pred = llm_result['suggested_price']
            
            # For multi-modal model prediction (simplified for Colab)
            multi_modal_pred = (llm_result['suggested_price'] + gb_pred) / 2
            
            # Ensemble weights
            weights = {
                'multi_modal': 0.4,
                'gradient_boosting': 0.35,
                'llm': 0.25
            }
            
            final_price = (
                weights['multi_modal'] * multi_modal_pred +
                weights['gradient_boosting'] * gb_pred +
                weights['llm'] * llm_result['suggested_price']
            )
            
            # Ensure price is reasonable
            final_price = max(0.5, final_price)  # Minimum $0.50
            
            predictions.append({
                'sample_id': sample_id,
                'price': final_price
            })
            
            if idx % 50 == 0:  # Print progress every 50 samples
                print(f"Processed {idx+1}/{len(test_featured_df)} samples...")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        print(f"Predictions generated for {len(predictions_df)} samples")
        return predictions_df

    def train_models(self, train_df):
        """Train all models on the training data"""
        print("Starting model training...")
        
        # Feature engineering for training data
        train_featured_df = self.advanced_feature_engineering(train_df, is_training=True)
        
        # Train Gradient Boosting
        self.train_gradient_boosting(train_featured_df)
        
        # Train multi-modal model if GPU available
        if torch.cuda.is_available():
            print("\n🚀 GPU detected! Training multi-modal model...")
            self.train_multi_modal_model(train_featured_df, epochs=10, batch_size=4)
        else:
            print("\n⏩ Skipping multi-modal model (CPU-only mode)")
        
        self.is_trained = True
        print("✅ All models trained successfully!")
        
        return train_featured_df

    def upload_and_train(self):
        """Interactive method for Colab - upload train file and train"""
        print("📁 Upload your TRAIN CSV file")
        print("Required columns: catalog_content, price")
        print("Optional columns: sample_id, image_link")
        
        uploaded = files.upload()
        
        if uploaded:
            file_name = list(uploaded.keys())[0]
            print(f"Train file '{file_name}' uploaded successfully!")
            
            # Load training data
            train_df = self.load_train_data(uploaded_file=uploaded[file_name])
            
            # Train models
            self.train_models(train_df)
            
            print("\n✅ Training completed! You can now upload test data for predictions.")
            return True
        else:
            print("❌ No file uploaded.")
            return False

    def upload_and_predict(self):
        """Interactive method for Colab - upload test file and generate predictions"""
        if not self.is_trained:
            print("❌ Please train the model first before making predictions")
            return None
        
        print("📁 Upload your TEST CSV file")
        print("Required columns: catalog_content")
        print("Optional columns: sample_id, image_link")
        
        uploaded = files.upload()
        
        if uploaded:
            file_name = list(uploaded.keys())[0]
            print(f"Test file '{file_name}' uploaded successfully!")
            
            # Load test data
            test_df = self.load_test_data(uploaded_file=uploaded[file_name])
            
            # Generate predictions
            predictions_df = self.predict_on_test_data(test_df)
            
            # Save predictions to CSV
            output_filename = 'predictions.csv'
            predictions_df.to_csv(output_filename, index=False)
            
            # Download the predictions
            files.download(output_filename)
            
            print(f"✅ Predictions saved to {output_filename} and downloaded!")
            print(f"📊 Output format: {len(predictions_df)} rows with columns: {predictions_df.columns.tolist()}")
            
            # Show sample predictions
            print("\n📋 Sample predictions:")
            print(predictions_df.head(10))
            
            return predictions_df
        else:
            print("❌ No file uploaded.")
            return None

    def evaluate_on_train(self):
        """Evaluate model performance on training data"""
        if not self.is_trained or not hasattr(self, 'train_df'):
            print("❌ Model not trained or training data not available")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION ON TRAINING DATA")
        print("="*50)
        
        # Feature engineering for training data
        train_featured_df = self.advanced_feature_engineering(self.train_df, is_training=True)
        
        # Make predictions on training data
        X_train = train_featured_df[self.feature_columns].fillna(0)
        y_train = train_featured_df['price']
        y_pred = self.gb_model.predict(X_train)
        
        # Calculate metrics
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"Gradient Boosting Performance on Training Data:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_train, y_pred, alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Training: Actual vs Predicted\nR² = {r2:.3f}')
        
        # Price distribution
        plt.subplot(1, 3, 2)
        plt.hist(y_train, bins=20, alpha=0.7, edgecolor='black', label='Actual')
        plt.hist(y_pred, bins=20, alpha=0.7, edgecolor='black', label='Predicted')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        plt.legend()
        
        # Residuals
        plt.subplot(1, 3, 3)
        residuals = y_train - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

# Main execution for Colab
def main():
    print("🛍  E-COMMERCE PRODUCT PRICE PREDICTION SYSTEM")
    print("="*60)
    print("This system uses separate train and test files")
    print("Output: predictions.csv with sample_id and price columns")
    print("="*60)
    
    # Initialize predictor
    predictor = ProductPricePredictor()
    
    trained = False
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. 📁 Upload TRAIN data and train models")
        print("2. 📁 Upload TEST data and generate predictions")
        print("3. 📊 Evaluate model on training data")
        print("4. 🚪 Exit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            # Upload and train
            trained = predictor.upload_and_train()
            
        elif choice == '2':
            # Upload and predict
            if trained:
                predictions = predictor.upload_and_predict()
            else:
                print("❌ Please train the model first (option 1)")
                
        elif choice == '3':
            # Evaluate
            if trained:
                predictor.evaluate_on_train()
            else:
                print("❌ Please train the model first (option 1)")
                
        elif choice == '4':
            print("👋 Thank you for using the Product Price Prediction System!")
            break
            
        else:
            print("❌ Invalid option. Please choose 1-4.")

# Run the system
if __name__ == "_main_":
    predictor = main()