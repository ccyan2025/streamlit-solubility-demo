import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import MolToImage
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Drug Solubility Prediction System", layout="wide")
st.title("🧪 Drug Solubility Prediction Model (LightGBM)")
st.markdown("Input drug molecule, solvent molecule and temperature to predict solubility (log10 mol/L).")

# Initialize session state for storing prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'solubility_value' not in st.session_state:
    st.session_state.solubility_value = None
if 'show_evaluation' not in st.session_state:
    st.session_state.show_evaluation = False

# Load LightGBM model and related files
@st.cache_resource
def load_models():
    try:
        # Load LightGBM model
        lightgbm_model = joblib.load('lightgbm_model.pkl')
        
        # Load temperature scaler
        scaler_temp = joblib.load('temperature_scaler.pkl')
        
        # Load feature information
        feature_info = joblib.load('feature_info.pkl')
        
        return lightgbm_model, scaler_temp, feature_info
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None

lightgbm_model, scaler_temp, feature_info = load_models()

# Define feature extraction function (consistent with training)
def smiles_to_descriptors(smiles, prefix):
    if pd.isna(smiles) or smiles == '':
        return {
            f'{prefix}_MolWt': 0.0,
            f'{prefix}_LogP': 0.0,
            f'{prefix}_TPSA': 0.0,
            f'{prefix}_HBA': 0,
            f'{prefix}_HBD': 0
        }
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            f'{prefix}_MolWt': 0.0,
            f'{prefix}_LogP': 0.0,
            f'{prefix}_TPSA': 0.0,
            f'{prefix}_HBA': 0,
            f'{prefix}_HBD': 0
        }
    return {
        f'{prefix}_MolWt': float(Descriptors.MolWt(mol)),
        f'{prefix}_LogP': float(Descriptors.MolLogP(mol)),
        f'{prefix}_TPSA': float(Descriptors.TPSA(mol)),
        f'{prefix}_HBA': int(Descriptors.NumHAcceptors(mol)),
        f'{prefix}_HBD': int(Descriptors.NumHDonors(mol))
    }

# Generate interaction features (consistent with training)
def generate_interaction_features(drug_features, solvent_features):
    features = {}
    
    # Original features
    features.update(drug_features)
    features.update(solvent_features)
    
    # Interaction features (same as training)
    if all(k in features for k in ['Drug_MolWt', 'Solvent_MolWt']):
        features['MolWt_ratio'] = features['Drug_MolWt'] / (features['Solvent_MolWt'] if features['Solvent_MolWt'] != 0 else 1e-6)
    
    if all(k in features for k in ['Drug_LogP', 'Solvent_LogP']):
        features['LogP_diff'] = features['Drug_LogP'] - features['Solvent_LogP']
    
    if all(k in features for k in ['Drug_TPSA', 'Solvent_TPSA']):
        features['TPSA_ratio'] = features['Drug_TPSA'] / (features['Solvent_TPSA'] if features['Solvent_TPSA'] != 0 else 1e-6)
    
    if all(k in features for k in ['Drug_HBA', 'Solvent_HBA']):
        features['HBA_sum'] = features['Drug_HBA'] + features['Solvent_HBA']
    
    if all(k in features for k in ['Drug_HBD', 'Solvent_HBD']):
        features['HBD_sum'] = features['Drug_HBD'] + features['Solvent_HBD']
    
    return features

# User input section
col1, col2 = st.columns(2)
with col1:
    drug_smiles = st.text_input("💊 Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")  # Example: Aspirin
    solvent_smiles = st.text_input("🧪 Solvent SMILES", value="CCO")  # Example: Ethanol
with col2:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=200.0, value=25.0, step=0.1)

# Prediction button
if st.button("🚀 Predict Solubility", use_container_width=True) and lightgbm_model is not None:
    try:
        # Extract features
        drug_features = smiles_to_descriptors(drug_smiles, "Drug")
        solvent_features = smiles_to_descriptors(solvent_smiles, "Solvent")
        
        # Generate interaction features
        all_features = generate_interaction_features(drug_features, solvent_features)
        
        # Temperature normalization
        if scaler_temp:
            temp_norm = scaler_temp.transform([[temperature]])[0][0]
            all_features['Temperature_norm'] = temp_norm
        
        # Build feature vector (according to training feature order)
        if feature_info and 'selected_features' in feature_info:
            selected_features = feature_info['selected_features']
            
            # Create feature vector in training order
            feature_vector = []
            for feat in selected_features:
                if feat in all_features:
                    feature_vector.append(all_features[feat])
                else:
                    # If feature doesn't exist, fill with 0 (or use training median, simplified here)
                    feature_vector.append(0.0)
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Predict using LightGBM model
            solubility_log10 = lightgbm_model.predict(feature_array)[0]
            
            # Store results in session state
            st.session_state.solubility_value = solubility_log10
            st.session_state.prediction_made = True
            st.session_state.all_features = all_features
            st.session_state.selected_features = selected_features
            st.session_state.drug_smiles = drug_smiles
            st.session_state.solvent_smiles = solvent_smiles
            
            # Output results
            st.success(f"✅ Predicted solubility (log10 mol/L): **{solubility_log10:.4f}**")
            st.metric("Predicted Value", f"{solubility_log10:.4f}")

            # Molecular structure visualization with smaller images
            st.subheader("🖼️ Molecular Structure Visualization")
            col3, col4 = st.columns(2)
            with col3:
                mol_drug = Chem.MolFromSmiles(drug_smiles)
                if mol_drug:
                    # Create smaller image
                    img_drug = MolToImage(mol_drug, size=(300, 200))
                    st.image(img_drug, caption="Drug Molecular Structure", use_column_width=True)
            with col4:
                mol_solvent = Chem.MolFromSmiles(solvent_smiles)
                if mol_solvent:
                    # Create smaller image
                    img_solvent = MolToImage(mol_solvent, size=(300, 200))
                    st.image(img_solvent, caption="Solvent Molecular Structure", use_column_width=True)

            # Feature importance visualization with smaller figure
            if hasattr(lightgbm_model, 'feature_importances_'):
                st.subheader("📊 Feature Importance (Top 10)")
                feature_importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': lightgbm_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                # Create smaller figure
                fig, ax = plt.subplots(figsize=(8, 5))  # Reduced size
                ax.barh(range(len(feature_importance_df)), 
                       feature_importance_df['Importance'].values,
                       color='skyblue')
                ax.set_yticks(range(len(feature_importance_df)))
                ax.set_yticklabels(feature_importance_df['Feature'].values, fontsize=9)  # Smaller font
                ax.set_xlabel('Feature Importance', fontsize=10)
                ax.set_title('Top 10 Feature Importance', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            # Display input feature values
            st.subheader("📋 Input Feature Values")
            feature_display_df = pd.DataFrame({
                'Feature': list(all_features.keys()),
                'Value': list(all_features.values())
            })
            st.dataframe(feature_display_df, use_container_width=True)
            
        else:
            st.error("❌ Failed to load feature information")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Please ensure the SMILES format is correct, e.g., Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`, Ethanol: `CCO`")

# Sidebar information
with st.sidebar:
    st.header("📚 Instructions")
    st.markdown("""
    1. Input drug and solvent SMILES strings on the left.
    2. Input temperature (unit: °C).
    3. Click the "Predict Solubility" button.
    4. View prediction results and visualization charts.
    
    **Example SMILES:**
    - Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
    - Ethanol: `CCO`
    - Water: `O`
    """)
    
    st.header("📊 Model Information")
    st.markdown("""
    - **Model Type:** LightGBM (Gradient Boosting Tree)
    - **Optimization Method:** Bayesian Hyperparameter Optimization
    - **Feature Engineering:** Molecular Descriptors + Interaction Features
    - **Evaluation Metrics:** MAE, MSE, R²
    """)
    
    # View Model Evaluation Report button
    if st.button("View Model Evaluation Report"):
        st.session_state.show_evaluation = True
    
    # Display evaluation report if button was clicked
    if st.session_state.show_evaluation:
        try:
            st.subheader("Model Evaluation Report")
            metrics_df = pd.read_csv('model_metrics.csv')
            st.dataframe(metrics_df, use_container_width=True)
            
            # Display training and test set performance comparison with smaller figure
            st.subheader("Model Performance Comparison")
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))  # Reduced size
            datasets = ['Training', 'Test']
            metrics = ['R2', 'MAE', 'MSE']
            
            for i, metric in enumerate(metrics):
                values = metrics_df[metric].values
                ax[i].bar(datasets, values, color=['skyblue', 'lightgreen'])
                ax[i].set_title(f'{metric} Comparison', fontsize=10)
                ax[i].set_ylabel(metric, fontsize=9)
                ax[i].tick_params(axis='x', labelsize=9)
                ax[i].tick_params(axis='y', labelsize=9)
                
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Close button for evaluation report
            if st.button("Close Report"):
                st.session_state.show_evaluation = False
                st.rerun()
                
        except Exception as e:
            st.warning(f"Model evaluation file not found or error: {e}")

# Display previous prediction results if they exist
if st.session_state.prediction_made and not st.session_state.show_evaluation:
    st.divider()
    st.subheader("Previous Prediction Results")
    st.info(f"Predicted solubility: **{st.session_state.solubility_value:.4f}** (log10 mol/L)")

# Footer information
st.markdown("---")
st.caption("💡 This system uses LightGBM machine learning model to predict drug solubility in solvents. Results are for reference only.")
