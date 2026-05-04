# Hand Gesture Feature Representation

## 🔥 Big Picture

Each **row = one hand (gesture)**

We transform raw 3D joint coordinates into a feature vector that captures:

> **shape, structure, and relative finger geometry**

—not absolute position in space.

---

## 🔹 Step 0: Raw Data Structure

Each hand consists of:

- 1 palm point  
- 5 fingers × 4 joints = 20 points  
# Hand Gesture Feature Representation

## 🔥 Big Picture

Each **row = one hand (gesture)**

We transform raw 3D joint coordinates into a feature vector that captures:

> **shape, structure, and relative finger geometry**

—not absolute position in space.

---

## 🔹 Step 0: Raw Data Structure

Each hand consists of:

- 1 palm point  
- 5 fingers × 4 joints = 20 points  

👉 Total = **21 points in 3D**

$$
\{p_0, p_1, ..., p_{20}\}, \quad p_i \in \mathbb{R}^3
$$

---

## 🔹 Step 1: Normalize the Hand

### ✅ Centering (Translation Invariance)

$$
p_i' = p_i - p_{\text{palm}}
$$

- Moves palm to the origin  
- Removes dependence on hand position  

---

### ✅ Scaling (Size Invariance)

$$
p_i'' = \frac{p_i'}{\|p'\|}
$$

- Normalizes overall hand size  
- Makes gestures comparable across users  

---

## 🔹 Final Normalized Points

$$
\{p_0'', p_1'', ..., p_{20}''\}
$$

---

## 🔹 Feature 1: Radial Distances (21 features)

Distance of each joint from the palm:

$$
r_i = \|p_i''\|
$$

### 👉 Captures:
- How extended each finger is  
- Overall hand openness  

**Examples:**
- Fist → small values  
- Open hand → large values  

---

## 🔹 Feature 2: Fingertip Distances (10 features)

Compute pairwise distances between the **5 fingertips**:

$$
d_{ij} = \|f_i - f_j\|
$$

Number of features:

$$
\binom{5}{2} = 10
$$

### 👉 Captures:
- Finger spacing  
- Gesture width  

**Examples:**
- Peace sign → large distance between two fingers  
- Closed hand → small distances  

---

## 🔹 Feature 3: Fingertip Angles (10 features)

Angle between fingertip vectors (from palm):

$$
\theta_{ij} = \cos^{-1}\left(
\frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}
\right)
$$

### 👉 Captures:
- Directional relationships between fingers  
- Geometry independent of rotation  

**Examples:**
- Fingers aligned → small angles  
- Fingers spread → large angles  

---

## 🔹 Feature 4: Finger Lengths (5 features)

For each finger:

$$
L_k = \sum \|p_{k,j+1} - p_{k,j}\|
$$

### 👉 Captures:
- Whether a finger is extended or curled  

**Examples:**
- Bent finger → shorter length  
- Straight finger → longer length  

---

## 🔹 Final Feature Vector

All features are concatenated:

$$
\text{Feature vector} =
[r_1, ..., r_{21},\ d_{ij},\ \theta_{ij},\ L_k]
$$

---

## 🔢 Total Feature Count

| Feature Type         | Count |
|---------------------|------:|
| Radial distances    | 21    |
| Fingertip distances | 10    |
| Fingertip angles    | 10    |
| Finger lengths      | 5     |
| **Total**           | **46** |

---

## 🔹 Why This Works

Instead of modeling:

> “Where is each point in space?”

We model:

> **“What shape does this hand form?”**

---

## 🔹 What Each Feature Captures

| Feature Type         | Meaning                         |
|---------------------|---------------------------------|
| Radial distances    | Open vs closed hand             |
| Fingertip distances | Finger spacing                  |
| Fingertip angles    | Hand geometry / orientation     |
| Finger lengths      | Finger bending                  |

---

## 🔹 Key Invariances

These features are:

- ✅ Translation invariant  
- ✅ Scale invariant  
- ✅ Mostly rotation invariant  
- ✅ Based on relationships (not raw position)  

---

## 🔹 One-Line Summary

> Raw coordinates → **Geometry-based representation of hand shape**
👉 Total = **21 points in 3D**

$$
\{p_0, p_1, ..., p_{20}\}, \quad p_i \in \mathbb{R}^3
$$

---

## 🔹 Step 1: Normalize the Hand

### ✅ Centering (Translation Invariance)

$$
p_i' = p_i - p_{\text{palm}}
$$

- Moves palm to the origin  
- Removes dependence on hand position  

---

### ✅ Scaling (Size Invariance)

$$
p_i'' = \frac{p_i'}{\|p'\|}
$$

- Normalizes overall hand size  
- Makes gestures comparable across users  

---

## 🔹 Final Normalized Points

$$
\{p_0'', p_1'', ..., p_{20}''\}
$$

---

## 🔹 Feature 1: Radial Distances (21 features)

Distance of each joint from the palm:

$$
r_i = \|p_i''\|
$$

### 👉 Captures:
- How extended each finger is  
- Overall hand openness  

**Examples:**
- Fist → small values  
- Open hand → large values  

---

## 🔹 Feature 2: Fingertip Distances (10 features)

Compute pairwise distances between the **5 fingertips**:

$$
d_{ij} = \|f_i - f_j\|
$$

Number of features:

$$
\binom{5}{2} = 10
$$

### 👉 Captures:
- Finger spacing  
- Gesture width  

**Examples:**
- Peace sign → large distance between two fingers  
- Closed hand → small distances  

---

## 🔹 Feature 3: Fingertip Angles (10 features)

Angle between fingertip vectors (from palm):

$$
\theta_{ij} = \cos^{-1}\left(
\frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}
\right)
$$

### 👉 Captures:
- Directional relationships between fingers  
- Geometry independent of rotation  

**Examples:**
- Fingers aligned → small angles  
- Fingers spread → large angles  

---

## 🔹 Feature 4: Finger Lengths (5 features)

For each finger:

$$
L_k = \sum \|p_{k,j+1} - p_{k,j}\|
$$

### 👉 Captures:
- Whether a finger is extended or curled  

**Examples:**
- Bent finger → shorter length  
- Straight finger → longer length  

---

## 🔹 Final Feature Vector

All features are concatenated:

$$
\text{Feature vector} =
[r_1, ..., r_{21},\ d_{ij},\ \theta_{ij},\ L_k]
$$

---

## 🔢 Total Feature Count

| Feature Type         | Count |
|---------------------|------:|
| Radial distances    | 21    |
| Fingertip distances | 10    |
| Fingertip angles    | 10    |
| Finger lengths      | 5     |
| **Total**           | **46** |

---

## 🔹 Why This Works

Instead of modeling:

> “Where is each point in space?”

We model:

> **“What shape does this hand form?”**

---

## 🔹 What Each Feature Captures

| Feature Type         | Meaning                         |
|---------------------|---------------------------------|
| Radial distances    | Open vs closed hand             |
| Fingertip distances | Finger spacing                  |
| Fingertip angles    | Hand geometry / orientation     |
| Finger lengths      | Finger bending                  |

---

## 🔹 Key Invariances

These features are:

- ✅ Translation invariant  
- ✅ Scale invariant  
- ✅ Mostly rotation invariant  
- ✅ Based on relationships (not raw position)  

---

## 🔹 One-Line Summary

> Raw coordinates → **Geometry-based representation of hand shape**