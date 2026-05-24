# 🖼️ Digital Image Processing — DIP Project

A desktop GUI application built with Python that covers all fundamental Digital Image Processing operations, developed as a university project at **Menoufia University**.

---

## ✨ Features

### 1. 📌 Point Operations
- **Addition** — Adds a constant value (+60) to all pixels
- **Subtraction** — Subtracts a constant value (−60) from all pixels
- **Division** — Divides all pixel values by 2
- **Complement** — Inverts the image (255 − pixel)

### 2. 🎨 Color Image Operations
- **Change Red Lighting** — Boosts the red channel intensity (+80)
- **Swap Channels (R ↔ G)** — Swaps the Red and Green channels
- **Eliminate Red** — Removes the red channel completely

### 3. 📊 Image Histogram
- **Histogram Stretching** — Enhances contrast by stretching pixel range
- **Histogram Equalization** — Equalizes the histogram for uniform distribution
- **Show Histogram** — Displays original vs equalized histogram chart

### 4. 🔲 Neighborhood Processing
**Linear Filters:**
- Average Filter (5×5)
- Laplacian Filter

**Non-Linear Filters:**
- Maximum Filter (5×5)
- Minimum Filter (5×5)
- Median Filter (5×5)
- Mode Filter — Most frequent values (5×5)

### 5. 🔧 Image Restoration
**Salt & Pepper Noise — Restoration methods:**
- Average Filter
- Median Filter
- Outlier Method

**Gaussian Noise — Restoration methods:**
- Image Averaging (n=10)
- Average Filter

### 6. ✂️ Image Segmentation
- **Basic Global Thresholding** — Manual threshold with live slider
- **Automatic Thresholding** — Otsu's method (auto threshold)
- **Adaptive Thresholding** — Local threshold based on neighborhood

### 7. 🔍 Edge Detection
- **Sobel Horizontal** — Detects horizontal edges
- **Sobel Vertical** — Detects vertical edges
- **Sobel Combined** — Combined edge magnitude

### 8. 🔷 Mathematical Morphology
- **Dilation** (5×5)
- **Erosion** (5×5)
- **Opening** (Erosion → Dilation)
- **Internal Boundary Extraction**
- **External Boundary Extraction**
- **Morphological Gradient**

---

## 🖥️ GUI
- All operations collected in **one unified desktop application**
- Each operation displays **Original** and **Result** images side by side
- Live **threshold slider** for segmentation operations
- Sidebar navigation with hover and active state highlighting

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `Python 3.x` | Core language |
| `OpenCV` | Image processing operations |
| `Tkinter` | GUI framework |
| `NumPy` | Array and pixel manipulation |
| `Pillow (PIL)` | Image display in GUI |
| `Matplotlib` | Histogram visualization |
| `SciPy` | Mode filter implementation |

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install opencv-python numpy pillow matplotlib scipy
```

**2. Run the application:**
```bash
python dip_project.py
```

---

## 📸 How to Use

1. Click **"Load Image"** to open any image (JPG, PNG, BMP, TIFF)
2. Select any operation from the sidebar
3. The **Original** image appears on the left, **Result** on the right
4. For segmentation, use the **live slider** to adjust the threshold in real time

---

## 📁 Project Structure
```
DIP-Project/
│
├── dip_project.py      # Main application
└── README.md           # Project documentation
```

---

## 🎓 Academic Info

- **University:** Menoufia University
- **Course:** Digital Image Processing (DIP)
- **Language:** Python

---

## 👨‍💻 Author

**Ahmed Hazem**  
Computer Science Student — Menoufia University  
[LinkedIn](#) • [GitHub](#)
