import numpy as np
from scipy.io import loadmat
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Primary SNOMED-CT codes that directly indicate ischemia
PRIMARY_ISCHEMIA_CODES = {
    '413444003',  # Acute ischemic heart disease
    '22298006',   # Myocardial ischemia
    '57054005',   # Acute myocardial ischemia
}

# Secondary codes that may indicate ischemia when combined
SECONDARY_ISCHEMIA_CODES = {
    '164861001',  # ST segment depression - common in ischemia
    '164865005',  # ST segment elevation - indicates possible STEMI
    '251238007',  # T wave abnormal - may indicate ischemia
    '164867002',  # ST segment abnormal
}

# Chronic conditions - not acute ischemia
CHRONIC_CODES = {
    '413844008',  # Chronic ischemic heart disease
    '426783006',  # Chronic ischemic heart disease
}

def read_hea_file(hea_path):
    """
    Read the header file to get diagnostic information
    
    Args:
        hea_path (str): Path to the .hea file
        
    Returns:
        bool: True if acute ischemia is present, False otherwise
    """
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#Dx:'):
                    # Extract diagnosis codes
                    dx_codes = set(line.split(':')[1].strip().split(','))
                    
                    # If any primary ischemia code is present, it's definitely ischemia
                    if dx_codes & PRIMARY_ISCHEMIA_CODES:
                        return True
                    
                    # If it's a chronic condition only, it's not acute ischemia
                    if dx_codes & CHRONIC_CODES and not (dx_codes & PRIMARY_ISCHEMIA_CODES):
                        return False
                    
                    # If there are at least two secondary indicators, consider it ischemia
                    secondary_matches = dx_codes & SECONDARY_ISCHEMIA_CODES
                    if len(secondary_matches) >= 2:
                        return True
                    
                    return False
        return False  # No diagnosis codes found
    except Exception as e:
        print(f"Error reading header file {hea_path}: {str(e)}")
        return None

def extract_features_from_mat(mat_file_path):
    """
    Load and extract features from a .mat ECG file
    
    Args:
        mat_file_path (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Load .mat file
        mat_data = loadmat(mat_file_path)
        # Get ECG signal data
        signal = mat_data.get('val', mat_data[list(mat_data.keys())[-1]])
        signal = signal.flatten()  # Ensure 1D array
        
        # Extract features
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'max': np.max(signal),
            'min': np.min(signal),
            'rms': np.sqrt(np.mean(np.square(signal))),  # Root mean square
            'peak_to_peak': np.ptp(signal),  # Peak-to-peak amplitude
        }
        return features
    except Exception as e:
        print(f"Error processing {mat_file_path}: {str(e)}")
        return None

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess the entire dataset from the directory
    
    Args:
        data_dir (str): Directory containing .mat and .hea files
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    features_list = []
    labels = []
    processed = 0
    ischemia_count = 0
    
    # Get all .mat files
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    total_files = len(mat_files)
    
    print(f"Found {total_files} .mat files")
    
    for mat_file in sorted(mat_files):
        base_name = mat_file[:-4]  # Remove .mat extension
        mat_path = os.path.join(data_dir, mat_file)
        hea_path = os.path.join(data_dir, base_name + '.hea')
        
        if os.path.exists(hea_path):
            features = extract_features_from_mat(mat_path)
            label = read_hea_file(hea_path)
            
            if features is not None and label is not None:
                feature_values = [
                    features['mean'],
                    features['std'],
                    features['max'],
                    features['min'],
                    features['rms'],
                    features['peak_to_peak']
                ]
                features_list.append(feature_values)
                labels.append(label)
                processed += 1
                if label:
                    ischemia_count += 1
        
        # Print progress every 100 files
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_files} files")
    
    if not features_list:
        raise ValueError("No valid data files found in the specified directory")
    
    print(f"\nDataset statistics:")
    print(f"Total processed files: {processed}")
    print(f"Ischemia cases: {ischemia_count} ({ischemia_count/processed*100:.1f}%)")
    print(f"Non-ischemia cases: {processed-ischemia_count} ({(processed-ischemia_count)/processed*100:.1f}%)")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test