import numpy as np

# Sample MediaPipe landmarks (replace with actual data)
mediapipe_landmarks = {
    'NOSE': {'x': 0.11046108603477478, 'y': 0.3069313168525696, 'z': -0.014904981479048729},
    'LEFT_EYE_INNER': {'x': 0.10104776173830032, 'y': 0.26885882019996643, 'z': -0.046609412878751755},
    'LEFT_EYE': {'x': 0.10263994336128235, 'y': 0.26447880268096924, 'z': -0.046895384788513184},
    'LEFT_EYE_OUTER': {'x': 0.10415074229240417, 'y': 0.26204055547714233, 'z': -0.0467832088470459},
    'RIGHT_EYE_INNER': {'x': 0.10145360231399536, 'y': 0.2656291425228119, 'z': -0.0022475349251180887},
    'RIGHT_EYE': {'x': 0.10362464189529419, 'y': 0.25904062390327454, 'z': -0.0027874158695340157},
    'RIGHT_EYE_OUTER': {'x': 0.10572028160095215, 'y': 0.2537047863006592, 'z': -0.0028583912644535303},
    'LEFT_EAR': {'x': 0.12195783853530884, 'y': 0.22563229501247406, 'z': -0.11729385703802109},
    'RIGHT_EAR': {'x': 0.12240374088287354, 'y': 0.2185983806848526, 'z': 0.08517616242170334},
    'MOUTH_LEFT': {'x': 0.13384681940078735, 'y': 0.3071644604206085, 'z': -0.027846667915582657},
    'MOUTH_RIGHT': {'x': 0.13492918014526367, 'y': 0.3035433292388916, 'z': 0.027374913915991783},
    'LEFT_SHOULDER': {'x': 0.21746686100959778, 'y': 0.28164878487586975, 'z': -0.19637450575828552},
    'RIGHT_SHOULDER': {'x': 0.2455911785364151, 'y': 0.24570633471012115, 'z': 0.26038289070129395},
    'LEFT_ELBOW': {'x': 0.23817335069179535, 'y': 0.6109218597412109, 'z': -0.2287784367799759},
    'RIGHT_ELBOW': {'x': 0.26053157448768616, 'y': 0.5461099147796631, 'z': 0.2653427720069885},
    'LEFT_WRIST': {'x': 0.2521745562553406, 'y': 0.9182374477386475, 'z': -0.20580998063087463},
    'RIGHT_WRIST': {'x': 0.26382407546043396, 'y': 0.8087633848190308, 'z': 0.1102309599518776},
    'LEFT_PINKY': {'x': 0.21302594244480133, 'y': 0.9551529884338379, 'z': -0.26273301243782043},
    'RIGHT_PINKY': {'x': 0.23212982714176178, 'y': 0.8091593980789185, 'z': 0.09311166405677795},
    'LEFT_INDEX': {'x': 0.21116946637630463, 'y': 0.9156287312507629, 'z': -0.23825229704380035},
    'RIGHT_INDEX': {'x': 0.2291271984577179, 'y': 0.8211706280708313, 'z': 0.04174191877245903},
    'LEFT_THUMB': {'x': 0.22665055096149445, 'y': 0.9106804728507996, 'z': -0.19509513676166534},
    'RIGHT_THUMB': {'x': 0.24241501092910767, 'y': 0.8272628784179688, 'z': 0.08197057247161865},
    'LEFT_HIP': {'x': 0.4864996671676636, 'y': 0.4098797142505646, 'z': -0.13847169280052185},
    'RIGHT_HIP': {'x': 0.49559324979782104, 'y': 0.36535096168518066, 'z': 0.1383330225944519},
    'LEFT_KNEE': {'x': 0.7036011815071106, 'y': 0.5659564137458801, 'z': -0.12866583466529846},
    'RIGHT_KNEE': {'x': 0.7071273326873779, 'y': 0.5258067846298218, 'z': 0.13314145803451538},
    'LEFT_ANKLE': {'x': 0.9333381652832031, 'y': 0.6592122316360474, 'z': -0.13290944695472717},
    'RIGHT_ANKLE': {'x': 0.916006326675415, 'y': 0.626302182674408, 'z': 0.18809330463409424},
    'LEFT_HEEL': {'x': 0.9830852746963501, 'y': 0.6391748785972595, 'z': -0.14310868084430695},
    'RIGHT_HEEL': {'x': 0.966896653175354, 'y': 0.6107717156410217, 'z': 0.1879265159368515},
    'LEFT_FOOT_INDEX': {'x': 0.9498238563537598, 'y': 0.9097353219985962, 'z': -0.2876703143119812},
    'RIGHT_FOOT_INDEX': {'x': 0.9275399446487427, 'y': 0.8473045229911804, 'z': 0.076536625623703}
}

def flatten_landmarks(landmarks):
    """Flatten landmarks into a single row of features (x, y, z for each landmark)."""
    flat_list = []
    
    # Corrected full order of 33 landmarks as per the landmarks.csv
    landmark_order = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
    
    # Flatten each landmark's x, y, z coordinates into the row
    for landmark in landmark_order:
        flat_list.append(landmarks[landmark]['x'])
        flat_list.append(landmarks[landmark]['y'])
        flat_list.append(landmarks[landmark]['z'])
    
    return flat_list

# Generate the flattened feature vector
feature_vector = flatten_landmarks(mediapipe_landmarks)

# Print out the feature vector (in the same order as the landmarks.csv)
print(f"Feature vector: {feature_vector}")

# Count number of entries in the feature vector
num_entries = len(feature_vector)
print(f"Number of entries in the feature vector: {num_entries}")
