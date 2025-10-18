# Install required packages
!pip install -q transformers torchvision pillow scikit-learn pandas numpy requests sentence-transformers

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

from google.colab import files
import io
from tqdm import tqdm

print("Checking GPU availability...")
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU - this may be slower")

class AdvancedProductPricePredictor:
    def _init_(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")

        # Initialize all models
        self.sentence_model = None
        self.scaler = StandardScaler()
        self.multi_modal_model = None
        self.gb_model = None
        self.graph_model = None
        self.feature_columns = []
        self.numerical_columns = []
        self.is_trained = False
        self.train_df = None
        self.test_df = None

    def load_train_data(self, file_path=None, uploaded_file=None):
        """Load training data"""
        print("\n📊 Loading training data...")
        if uploaded_file is not None:
            self.train_df = pd.read_csv(io.BytesIO(uploaded_file))
        elif file_path is not None:
            self.train_df = pd.read_csv(file_path)
        else:
            raise ValueError("No training data provided")

        print(f"✓ Training dataset loaded: {self.train_df.shape[0]} rows, {self.train_df.shape[1]} columns")
        print(f"✓ Columns: {', '.join(self.train_df.columns.tolist())}")
        return self.train_df

    def load_test_data(self, file_path=None, uploaded_file=None):
        """Load test data"""
        print("\n📊 Loading test data...")
        if uploaded_file is not None:
            self.test_df = pd.read_csv(io.BytesIO(uploaded_file))
        elif file_path is not None:
            self.test_df = pd.read_csv(file_path)
        else:
            raise ValueError("No test data provided")

        print(f"✓ Test dataset loaded: {self.test_df.shape[0]} rows, {self.test_df.shape[1]} columns")
        return self.test_df

    def advanced_feature_engineering(self, df, is_training=True):
        """Enhanced feature engineering"""
        print("\n🔧 Performing feature engineering...")

        def extract_features_from_catalog(catalog_text):
            features = {}

            if pd.isna(catalog_text):
                catalog_text = ""

            # Basic text features
            item_name_match = re.search(r'Item Name:\s*([^\n]+)', str(catalog_text))
            features['item_name'] = item_name_match.group(1).strip() if item_name_match else ""

            value_match = re.search(r'Value:\s*([\d.]+)', str(catalog_text))
            features['value'] = float(value_match.group(1)) if value_match else 0.0

            unit_match = re.search(r'Unit:\s*([^\n]+)', str(catalog_text))
            features['unit'] = unit_match.group(1).strip() if unit_match else ""

            # Advanced text features
            bullet_points = re.findall(r'Bullet Point\s*\d+:', str(catalog_text))
            features['bullet_points_count'] = len(bullet_points)

            features['has_product_description'] = 1 if 'Product Description:' in str(catalog_text) else 0
            features['catalog_length'] = len(str(catalog_text))
            features['item_name_length'] = len(features['item_name'])

            # Premium indicators
            text_lower = str(catalog_text).lower()
            premium_indicators = {
                'has_organic': ['organic'],
                'has_gluten_free': ['gluten-free', 'gluten free'],
                'has_vegan': ['vegan'],
                'has_natural': ['natural'],
                'has_kosher': ['kosher'],
                'has_non_gmo': ['non-gmo', 'non gmo'],
                'has_premium': ['premium', 'gourmet', 'artisanal', 'luxury'],
                'has_imported': ['imported', 'import'],
                'has_handcrafted': ['handcrafted', 'handmade']
            }

            for feature, keywords in premium_indicators.items():
                features[feature] = 1 if any(keyword in text_lower for keyword in keywords) else 0

            # Product categories
            categories = {
                'coffee': ['coffee', 'espresso', 'cappuccino'],
                'tea': ['tea', 'chai', 'herbal tea'],
                'chocolate': ['chocolate', 'cocoa'],
                'snack': ['snack', 'chips', 'crackers'],
                'supplement': ['supplement', 'vitamin', 'protein'],
                'beverage': ['beverage', 'drink', 'juice'],
                'oil': ['oil', 'olive oil', 'coconut oil'],
                'baking': ['baking', 'flour', 'sugar'],
                'spice': ['spice', 'herb', 'seasoning'],
                'canned': ['canned', 'jar', 'bottle']
            }

            for category, keywords in categories.items():
                features[f'category_{category}'] = 1 if any(keyword in text_lower for keyword in keywords) else 0

            # Scores
            features['premium_score'] = sum([features.get(f'has_{ind}', 0) for ind in ['organic', 'premium', 'imported', 'handcrafted']])
            features['health_score'] = sum([features.get(f'has_{ind}', 0) for ind in ['gluten_free', 'vegan', 'natural', 'non_gmo']])

            return features

        def extract_package_info(text):
            pack_match = re.search(r'pack of\s*(\d+)', str(text).lower())
            count_match = re.search(r'(\d+)\s*count', str(text).lower())
            weight_match = re.search(r'(\d+\.?\d*)\s*(oz|ounce|lb|pound|fl oz)', str(text).lower())

            pack_size = int(pack_match.group(1)) if pack_match else 1
            item_count = int(count_match.group(1)) if count_match else 1
            package_weight = float(weight_match.group(1)) if weight_match else 0

            return pack_size, item_count, package_weight

        # Apply feature extraction with progress
        print("  → Extracting catalog features...")
        catalog_features = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing", disable=len(df) < 100):
            catalog_features.append(extract_features_from_catalog(row['catalog_content']))

        features_df = pd.DataFrame(catalog_features)

        # Combine with original data
        featured_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

        # Package features
        print("  → Extracting package information...")
        package_info = featured_df['catalog_content'].apply(extract_package_info)
        featured_df['pack_size'], featured_df['item_count'], featured_df['package_weight'] = zip(*package_info)

        # Fill NaN values
        featured_df = featured_df.fillna(0)

        print(f"✓ Feature engineering completed: {featured_df.shape[1]} features created")
        return featured_df

    # Simplified Multi-modal Model (CPU-friendly)
    class AdvancedMultiModalModel(nn.Module):
        def _init_(self, text_dim=384, numerical_dim=30, hidden_dim=256, dropout=0.3):
            super()._init_()

            # Text encoder
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Numerical feature processor
            self.numerical_encoder = nn.Sequential(
                nn.Linear(numerical_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Regression head
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, text_features, numerical_features):
            # Encode each modality
            text_emb = self.text_encoder(text_features)
            numerical_emb = self.numerical_encoder(numerical_features)

            # Concatenate and fuse
            combined = torch.cat([text_emb, numerical_emb], dim=1)
            fused_features = self.fusion(combined)

            # Regression
            price_pred = self.regressor(fused_features)

            return price_pred.squeeze(-1)

    # Simplified Dataset Class (no images for speed)
    class MultiModalDataset(Dataset):
        def _init_(self, df, text_model, numerical_columns, scaler=None, is_training=True):
            self.df = df.reset_index(drop=True)
            self.is_training = is_training
            self.text_model = text_model
            self.numerical_columns = numerical_columns
            self.scaler = scaler if scaler is not None else StandardScaler()

            # Fit scaler on training data
            if is_training:
                numerical_data = self.df[self.numerical_columns].fillna(0).values
                self.scaler.fit(numerical_data)

            # Pre-compute text embeddings to avoid repeated encoding
            print("  → Pre-computing text embeddings...")
            self.text_embeddings = []
            batch_texts = []
            for idx in tqdm(range(len(self.df)), desc="  Encoding text", disable=len(self.df) < 100):
                row = self.df.iloc[idx]
                text = f"{row.get('item_name', '')} {row['catalog_content']}"
                batch_texts.append(text)

                # Process in batches of 32
                if len(batch_texts) == 32 or idx == len(self.df) - 1:
                    with torch.no_grad():
                        embeddings = self.text_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                        self.text_embeddings.extend(embeddings)
                    batch_texts = []

        def _len_(self):
            return len(self.df)

        def _getitem_(self, idx):
            row = self.df.iloc[idx]

            # Get pre-computed text features
            text_features = self.text_embeddings[idx]

            # Numerical features
            numerical_data = row[self.numerical_columns].fillna(0).values.astype(float)
            numerical_tensor = torch.FloatTensor(self.scaler.transform([numerical_data])[0])

            if self.is_training and 'price' in row:
                price = torch.FloatTensor([row['price']])
                return text_features, numerical_tensor, price
            else:
                return text_features, numerical_tensor, row.get('sample_id', idx)

    def train_advanced_models(self, train_df, epochs=10, batch_size=16):
        """Train all models"""
        print("\n🚀 Training models...")

        # Feature engineering
        train_featured_df = self.advanced_feature_engineering(train_df, is_training=True)

        # Initialize sentence transformer
        print("\n🤖 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model = self.sentence_model.to(self.device)
        print("✓ Model loaded")

        # Define numerical columns
        self.numerical_columns = [
            'value', 'bullet_points_count', 'has_product_description',
            'catalog_length', 'item_name_length', 'has_organic',
            'has_gluten_free', 'has_vegan', 'has_natural', 'has_kosher',
            'has_non_gmo', 'pack_size', 'item_count', 'package_weight',
            'premium_score', 'health_score'
        ] + [col for col in train_featured_df.columns if col.startswith('category_')]

        print(f"✓ Using {len(self.numerical_columns)} numerical features")

        # Create dataset
        print("\n📦 Creating dataset...")
        dataset = self.MultiModalDataset(
            train_featured_df,
            self.sentence_model,
            self.numerical_columns,
            is_training=True
        )
        self.scaler = dataset.scaler

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        print(f"✓ Dataset created: {len(dataset)} samples, batch size: {batch_size}")

        # Initialize neural network model
        print("\n🧠 Initializing neural network...")
        self.multi_modal_model = self.AdvancedMultiModalModel(
            text_dim=384,
            numerical_dim=len(self.numerical_columns)
        ).to(self.device)
        print("✓ Model initialized")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.multi_modal_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        # Loss function
        criterion = nn.HuberLoss()

        # Training loop
        print(f"\n🏋 Training for {epochs} epochs...")
        self.multi_modal_model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                text_features, numerical_features, prices = batch

                # Move to device
                text_features = text_features.to(self.device)
                numerical_features = numerical_features.to(self.device)
                prices = prices.to(self.device).squeeze()

                # Forward pass
                optimizer.zero_grad()
                price_preds = self.multi_modal_model(text_features, numerical_features)

                # Loss
                loss = criterion(price_preds, prices)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multi_modal_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch+1} complete - Avg Loss: {avg_loss:.4f}")

        print("✓ Neural network training complete")

        # Train Gradient Boosting
        self.train_gradient_boosting(train_featured_df)

        self.is_trained = True
        print("\n✅ All models trained successfully!")

        return train_featured_df

    def train_gradient_boosting(self, df):
        """Train Gradient Boosting"""
        print("\n🌲 Training Gradient Boosting...")

        self.feature_columns = [col for col in self.numerical_columns if col in df.columns]

        X = df[self.feature_columns].fillna(0)
        y = df['price']

        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        print("  → Fitting model...")
        self.gb_model.fit(X, y)

        y_pred = self.gb_model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"✓ Gradient Boosting trained - MAE: ${mae:.2f}, R²: {r2:.4f}")

        return self.gb_model

    def ensemble_prediction(self, test_df):
        """Ensemble prediction"""
        print("\n🔮 Generating predictions...")

        # Feature engineering
        test_featured_df = self.advanced_feature_engineering(test_df, is_training=False)

        # Create dataset
        print("\n📦 Preparing test dataset...")
        test_dataset = self.MultiModalDataset(
            test_featured_df,
            self.sentence_model,
            self.numerical_columns,
            scaler=self.scaler,
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        self.multi_modal_model.eval()

        # Neural network predictions
        print("\n🧠 Getting neural network predictions...")
        nn_predictions = []
        all_sample_ids = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="  Predicting"):
                text_features, numerical_features, sample_ids = batch

                text_features = text_features.to(self.device)
                numerical_features = numerical_features.to(self.device)

                preds = self.multi_modal_model(text_features, numerical_features)
                nn_predictions.extend(preds.cpu().numpy())
                all_sample_ids.extend(sample_ids if isinstance(sample_ids, list) else sample_ids.tolist())

        # Gradient Boosting predictions
        print("\n🌲 Getting Gradient Boosting predictions...")
        X_test = test_featured_df[self.feature_columns].fillna(0)
        gb_predictions = self.gb_model.predict(X_test)

        # Ensemble
        print("\n⚖ Combining predictions...")
        final_predictions = []
        for i in range(len(nn_predictions)):
            # Weighted average: 60% NN, 40% GB
            final_price = 0.6 * nn_predictions[i] + 0.4 * gb_predictions[i]
            final_price = max(0.5, final_price)  # Minimum price

            final_predictions.append({
                'sample_id': all_sample_ids[i],
                'price': round(final_price, 2)
            })

        predictions_df = pd.DataFrame(final_predictions)
        print(f"✓ Generated {len(predictions_df)} predictions")

        # Statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"  Mean price: ${predictions_df['price'].mean():.2f}")
        print(f"  Median price: ${predictions_df['price'].median():.2f}")
        print(f"  Min price: ${predictions_df['price'].min():.2f}")
        print(f"  Max price: ${predictions_df['price'].max():.2f}")

        return predictions_df

    def upload_and_train(self):
        """Interactive training interface"""
        print("\n" + "="*60)
        print("📁 UPLOAD TRAINING DATA")
        print("="*60)
        print("Required columns: catalog_content, price")
        print("Optional columns: sample_id, image_link")
        print()

        uploaded = files.upload()

        if uploaded:
            file_name = list(uploaded.keys())[0]
            print(f"\n✓ File '{file_name}' uploaded!")

            train_df = self.load_train_data(uploaded_file=uploaded[file_name])
            self.train_advanced_models(train_df)

            return True
        else:
            print("❌ No file uploaded")
            return False

    def upload_and_predict(self):
        """Interactive prediction interface"""
        if not self.is_trained:
            print("❌ Please train the model first")
            return None

        print("\n" + "="*60)
        print("📁 UPLOAD TEST DATA")
        print("="*60)
        print("Required columns: catalog_content")
        print("Optional columns: sample_id, image_link")
        print()

        uploaded = files.upload()

        if uploaded:
            file_name = list(uploaded.keys())[0]
            print(f"\n✓ File '{file_name}' uploaded!")

            test_df = self.load_test_data(uploaded_file=uploaded[file_name])
            predictions_df = self.ensemble_prediction(test_df)

            output_filename = 'predictions.csv'
            predictions_df.to_csv(output_filename, index=False)

            print(f"\n💾 Saving predictions...")
            files.download(output_filename)

            print(f"✅ Predictions saved to '{output_filename}'")
            print(f"\n📋 First 10 predictions:")
            print(predictions_df.head(10).to_string(index=False))

            return predictions_df
        else:
            print("❌ No file uploaded")
            return None

# Main execution
def main():
    print("\n" + "="*60)
    print("🚀 PRODUCT PRICE PREDICTION SYSTEM")
    print("="*60)
    print("Advanced AI-powered price prediction")
    print("Output: predictions.csv with sample_id and price")
    print("="*60)

    predictor = AdvancedProductPricePredictor()
    trained = False

    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. 📁 Upload TRAIN data and train models")
        print("2. 📁 Upload TEST data and generate predictions")
        print("3. ℹ  Show model information")
        print("4. 🚪 Exit")
        print("="*60)

        choice = input("\nChoose option (1-4): ").strip()

        if choice == '1':
            trained = predictor.upload_and_train()

        elif choice == '2':
            if trained:
                predictor.upload_and_predict()
            else:
                print("\n❌ Please train the model first (option 1)")

        elif choice == '3':
            if trained:
                print("\n" + "="*60)
                print("🤖 MODEL INFORMATION")
                print("="*60)
                print("Architecture:")
                print("  • Multi-modal Neural Network (Text + Numerical)")
                print("  • Gradient Boosting Regressor")
                print("  • Ensemble: 60% NN + 40% GB")
                print("\nFeatures:")
                print(f"  • {len(predictor.numerical_columns)} engineered features")
                print("  • Sentence transformer embeddings (384-dim)")
                print("  • Advanced text processing")
                print("="*60)
            else:
                print("\n❌ Models not trained yet")

        elif choice == '4':
            print("\n👋 Thank you for using the Price Prediction System!")
            print("="*60)
            break

        else:
            print("\n❌ Invalid option. Please choose 1-4")

main()