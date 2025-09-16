# test_models.py - Run this to verify your models work
import joblib
import pandas as pd
import numpy as np


import joblib
import pandas as pd
import numpy as np

def inspect_model_features():
    """Inspect what features your trained models expect"""
    print("🔍 Inspecting model feature requirements...")
    
    try:
        # Load feature preprocessor to see what it expects
        feature_preprocessor = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_prediction_preprocessor.pkl")
        feature_names = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_prediction_feature_names.pkl")
        
        print(f"📋 Expected feature names: {feature_names}")
        
        # Try to get feature names from the preprocessor
        if hasattr(feature_preprocessor, 'feature_names_in_'):
            print(f"📋 Preprocessor expects: {list(feature_preprocessor.feature_names_in_)}")
        
        # Check clustering preprocessor too
        cluster_preprocessor = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_clustering_preproc.pkl")
        if hasattr(cluster_preprocessor, 'feature_names_in_'):
            print(f"📋 Clustering preprocessor expects: {list(cluster_preprocessor.feature_names_in_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting models: {e}")
        return False

# Updated test with complete feature set
def test_models_fixed():
    print("🔍 Testing MedOptix Models with Complete Features...")
    
    try:
        # Test 1: Load all models
        print("\n1️⃣ Loading All Models...")
        preprocessor = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_clustering_preproc.pkl")
        pca = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_clustering_pca.pkl")
        kmeans = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_clustering_kmeans.pkl")
        dropout_model = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_prediction_model.pkl")
        feature_preprocessor = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_prediction_preprocessor.pkl")
        feature_names = joblib.load("C:\Users\kaoth\OneDrive\Desktop\Data Science Lectures\Machine Learning\Amdari Internship\medoptix-ai-internship\models\medoptix_prediction_feature_names.pkl")
        print("✅ All models loaded successfully")
        
        # Test 2: Create sample with ALL expected features
        print("\n2️⃣ Creating Complete Sample Patient...")
        sample_patient = {
            'age': 45.0,
            'gender': 'Female', 
            'bmi': 28.5,
            'smoker': 'No',
            'chronic_cond': 'No',
            'injury_type': 'knee',
            'referral_source': 'GP',
            'insurance_type': 'NHS',
            'n_sessions': 5,
            'avg_session_duration': 45.0,
            'first_week': 1,  # ← ADD MISSING FEATURE
            'last_week': 5,   # ← ADD MISSING FEATURE
            'mean_pain': 6.0,
            'mean_pain_delta': -1.0,
            'home_adherence_mean': 0.8,
            'satisfaction_mean': 3.5
        }
        
        print(f"📋 Sample patient features: {list(sample_patient.keys())}")
        
        # Test 3: Predict cluster
        print("\n3️⃣ Testing Cluster Prediction...")
        df = pd.DataFrame([sample_patient])
        
        # Get expected clustering features from preprocessor
        if hasattr(preprocessor, 'feature_names_in_'):
            cluster_features = list(preprocessor.feature_names_in_)
        else:
            # Fallback to common features
            cluster_features = ['age', 'bmi', 'n_sessions', 'avg_session_duration', 
                               'first_week', 'last_week', 'mean_pain', 'mean_pain_delta', 
                               'gender', 'smoker', 'chronic_cond', 'injury_type']
        
        print(f"📋 Using cluster features: {cluster_features}")
        
        # Filter available features
        available_features = [f for f in cluster_features if f in df.columns]
        df_cluster = df[available_features]
        
        print(f"📋 Available cluster features: {available_features}")
        
        X_processed = preprocessor.transform(df_cluster)
        X_pca = pca.transform(X_processed)
        cluster = kmeans.predict(X_pca)[0]
        print(f"✅ Predicted cluster: {cluster}")
        
        # Test 4: Predict dropout
        print("\n4️⃣ Testing Dropout Prediction...")
        
        # Add cluster to sample
        sample_with_cluster = sample_patient.copy()
        sample_with_cluster['cluster'] = cluster
        df_pred = pd.DataFrame([sample_with_cluster])
        
        # Get expected prediction features from preprocessor
        if hasattr(feature_preprocessor, 'feature_names_in_'):
            expected_features = list(feature_preprocessor.feature_names_in_)
            print(f"📋 Expected prediction features: {expected_features}")
        else:
            # Use all except cluster for preprocessing
            expected_features = [col for col in df_pred.columns if col != 'cluster']
        
        # Filter to only features the model expects
        available_pred_features = [f for f in expected_features if f in df_pred.columns]
        X_features = df_pred[available_pred_features]
        
        print(f"📋 Using prediction features: {available_pred_features}")
        
        # Apply feature preprocessing
        X_processed_pred = feature_preprocessor.transform(X_features)
        
        # Create DataFrame with feature names
        feature_names_no_cluster = [name for name in feature_names if name != 'cluster']
        X_df = pd.DataFrame(X_processed_pred, columns=feature_names_no_cluster)
        
        # Add cluster back
        X_df['cluster'] = cluster
        
        # Predict dropout
        dropout_prob = dropout_model.predict_proba(X_df.values)[0, 1]
        print(f"✅ Predicted dropout probability: {dropout_prob:.3f}")
        
        print("\n🎉 ALL TESTS PASSED! Models working with complete features.")
        return True, sample_patient
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return False, None

if __name__ == "__main__":
    # First inspect what the model expects
    inspect_model_features()
    print("\n" + "="*50)
    
    # Then test with complete features
    success, sample = test_models_fixed()
    if success:
        print("\n✅ Ready to proceed with API development!")
        print(f"📋 Use this complete sample structure: {sample}")
    else:
        print("\n🔧 Need to investigate further...")