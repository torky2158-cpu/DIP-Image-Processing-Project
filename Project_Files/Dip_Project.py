import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'bg': '#1e1e1e', 'fg': '#ffffff', 'btn': '#3c3c3c',
    'btn_hov': '#505050', 'accent': '#007acc', 'success': '#6a9955'
}

class DIPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DIP Project - Digital Image Processing")
        self.root.geometry("1300x750")
        self.root.configure(bg=COLORS['bg'])
        
        self.original = None
        self.result = None
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top bar
        top_bar = tk.Frame(main_frame, bg=COLORS['bg'])
        top_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.load_btn = tk.Button(top_bar, text="📁 Load Image", command=self.load_image,
                                  bg=COLORS['btn'], fg=COLORS['fg'], font=('Arial', 10, 'bold'),
                                  padx=20, pady=8, relief=tk.FLAT)
        self.load_btn.pack(side=tk.LEFT)
        
        self.save_btn = tk.Button(top_bar, text="💾 Save Result", command=self.save_result,
                                  bg=COLORS['btn'], fg=COLORS['fg'], font=('Arial', 10, 'bold'),
                                  padx=20, pady=8, relief=tk.FLAT)
        self.save_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        self.status = tk.Label(top_bar, text="No image loaded", bg=COLORS['bg'], 
                               fg='#888888', font=('Arial', 9))
        self.status.pack(side=tk.LEFT, padx=20)
        
        # Image display area
        images_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Original
        left_panel = tk.Frame(images_frame, bg=COLORS['bg'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(left_panel, text="ORIGINAL IMAGE", bg=COLORS['bg'], fg=COLORS['accent'],
                font=('Arial', 11, 'bold')).pack()
        self.canvas_orig = tk.Canvas(left_panel, bg='#0d0d0d', highlightthickness=1,
                                      highlightbackground='#404040')
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right panel - Result
        right_panel = tk.Frame(images_frame, bg=COLORS['bg'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(right_panel, text="RESULT IMAGE", bg=COLORS['bg'], fg=COLORS['success'],
                font=('Arial', 11, 'bold')).pack()
        self.canvas_res = tk.Canvas(right_panel, bg='#0d0d0d', highlightthickness=1,
                                     highlightbackground='#404040')
        self.canvas_res.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Categories (organized as requested)
        categories_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        categories_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create Notebook (tabbed interface)
        notebook = ttk.Notebook(categories_frame)
        notebook.pack(fill=tk.X)
        
        style = ttk.Style()
        style.theme_use('clam')  # مهم عشان يطبق الألوان صح
        
        style.configure(
            'TNotebook',
            background=COLORS['bg'],
            borderwidth=0
        )
        
        style.configure(
            'TNotebook.Tab',
            background='#2d2d2d',
            foreground='#ffffff',
            padding=[12, 6],
            font=('Arial', 9, 'bold')
        )
        
        style.map(
            'TNotebook.Tab',
            background=[
                ('selected', COLORS['accent']),
                ('active', '#404040')
            ],
            foreground=[
                ('selected', '#ffffff'),
                ('active', '#ffffff')
            ]
        )

        
        # 1. POINT OPERATIONS
        frame1 = self.create_category_frame(notebook)
        self.add_button(frame1, "1. Addition (+50)", self.op_addition, row=0, col=0)
        self.add_button(frame1, "2. Addition (Image + Image)", self.op_add_image, row=0, col=1)
        self.add_button(frame1, "3. Subtraction (-60)", self.op_subtraction, row=0, col=2)
        self.add_button(frame1, "4. Division (÷2)", self.op_division, row=0, col=3)
        self.add_button(frame1, "5. Complement", self.op_complement, row=1, col=0)
        notebook.add(frame1, text="1. Point Operations")
        
        # 2. COLOR IMAGE OPERATIONS
        frame2 = self.create_category_frame(notebook)
        self.add_button(frame2, "5a. Change Red Lighting", self.op_change_red, row=0, col=0)
        self.add_button(frame2, "6a. Swap R ↔ G", self.op_swap_rg, row=0, col=1)
        self.add_button(frame2, "7a. Eliminate Red", self.op_elim_red, row=0, col=2)
        notebook.add(frame2, text="2. Color Operations")
        
        # 3. IMAGE HISTOGRAM
        frame3 = self.create_category_frame(notebook)
        self.add_button(frame3, "8. Histogram Stretching", self.op_hist_stretch, row=0, col=0)
        self.add_button(frame3, "9. Histogram Equalization", self.op_hist_equal, row=0, col=1)
        self.add_button(frame3, "Show Histogram", self.op_show_hist, row=0, col=2)
        notebook.add(frame3, text="3. Histogram")
        
        # 4. NEIGHBORHOOD PROCESSING
        frame4 = self.create_category_frame(notebook)
        self.add_button(frame4, "1a. Average Filter", self.op_average, row=0, col=0)
        self.add_button(frame4, "1b. Laplacian Filter", self.op_laplacian, row=0, col=1)
        self.add_button(frame4, "2a. Maximum Filter", self.op_maximum, row=1, col=0)
        self.add_button(frame4, "2b. Minimum Filter", self.op_minimum, row=1, col=1)
        self.add_button(frame4, "2c. Median Filter", self.op_median, row=1, col=2)
        self.add_button(frame4, "2d. Mode Filter", self.op_mode, row=2, col=0)
        notebook.add(frame4, text="4. Neighborhood Filters")
        
        # 5. IMAGE RESTORATION
        frame5 = self.create_category_frame(notebook)
        self.add_button(frame5, "Add Salt & Pepper", self.op_add_sp, row=0, col=0)
        self.add_button(frame5, "Add Gaussian Noise", self.op_add_gauss, row=0, col=1)
        self.add_button(frame5, "Restore: Average", self.op_restore_average, row=1, col=0)
        self.add_button(frame5, "Restore: Median", self.op_restore_median, row=1, col=1)
        self.add_button(frame5, "Restore: Outlier Method", self.op_restore_outlier, row=1, col=2)
        notebook.add(frame5, text="5. Image Restoration")
        
        # 6. IMAGE SEGMENTATION
        frame6 = self.create_category_frame(notebook)
        self.add_button(frame6, "a) Basic Global Threshold", self.op_global_thresh, row=0, col=0)
        self.add_button(frame6, "b) Automatic (Otsu)", self.op_otsu_thresh, row=0, col=1)
        self.add_button(frame6, "c) Adaptive Threshold", self.op_adaptive_thresh, row=0, col=2)
        notebook.add(frame6, text="6. Segmentation")
        
        # 7. EDGE DETECTION
        frame7 = self.create_category_frame(notebook)
        self.add_button(frame7, "Sobel Detector (X)", self.op_sobel_x, row=0, col=0)
        self.add_button(frame7, "Sobel Detector (Y)", self.op_sobel_y, row=0, col=1)
        self.add_button(frame7, "Sobel Combined", self.op_sobel_combined, row=0, col=2)
        notebook.add(frame7, text="7. Edge Detection")
        
        # 8. MATHEMATICAL MORPHOLOGY
        frame8 = self.create_category_frame(notebook)
        self.add_button(frame8, "Dilation", self.op_dilate, row=0, col=0)
        self.add_button(frame8, "Erosion", self.op_erode, row=0, col=1)
        self.add_button(frame8, "Opening", self.op_opening, row=0, col=2)
        notebook.add(frame8, text="8. Morphology")
        
        # 9. BOUNDARY EXTRACTION
        frame9 = self.create_category_frame(notebook)
        self.add_button(frame9, "a) Internal Boundary", self.op_boundary_internal, row=0, col=0)
        self.add_button(frame9, "b) External Boundary", self.op_boundary_external, row=0, col=1)
        self.add_button(frame9, "c) Morphological Gradient", self.op_boundary_gradient, row=0, col=2)
        notebook.add(frame9, text="9. Boundary Extraction")
        
        # Slider for threshold
        self.slider_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        self.threshold_slider = tk.Scale(self.slider_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         bg=COLORS['bg'], fg=COLORS['fg'], highlightthickness=0,
                                         length=300)
        self.threshold_slider.pack()
        self.slider_label = tk.Label(self.slider_frame, text="Threshold: 127", bg=COLORS['bg'], fg=COLORS['fg'])
        self.slider_label.pack()
        
        # Bind resize events
        self.canvas_orig.bind("<Configure>", lambda e: self.show_image(self.canvas_orig, self.original))
        self.canvas_res.bind("<Configure>", lambda e: self.show_image(self.canvas_res, self.result))
    
    def create_category_frame(self, parent):
        frame = tk.Frame(parent, bg=COLORS['bg'])
        return frame
    
    def add_button(self, parent, text, command, row, col):
        btn = tk.Button(parent, text=text, command=command,
                       bg=COLORS['btn'], fg=COLORS['fg'], font=('Arial', 9),
                       padx=10, pady=6, relief=tk.FLAT, cursor='hand2')
        btn.grid(row=row, column=col, padx=5, pady=3, sticky='ew')
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=COLORS['btn_hov']))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=COLORS['btn']))
        parent.columnconfigure(col, weight=1)
    
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if path:
            img = self.load_image_file(path)
            if img is not None:
                self.original = img
                self.result = img.copy()
                self.show_image(self.canvas_orig, img)
                self.show_image(self.canvas_res, img)
                h, w = img.shape[:2]
                self.status.config(text=f"{os.path.basename(path)} ({w}x{h})")
            else:
                messagebox.showerror("Cannot read file", "Cannot read this file. Please choose another image.")

    def load_image_file(self, path):
        try:
            pil_img = Image.open(path)
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)[:, :, ::-1]
            return img
        except Exception:
            try:
                with open(path, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                return img
            except Exception:
                return None
    
    def show_image(self, canvas, img):
        if img is None: return
        canvas.update_idletasks()
        if canvas.winfo_width() < 2: return
        
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = rgb.shape[:2]
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(rgb, (nw, nh))
        
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        canvas.delete("all")
        canvas._photo = photo
        canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor="nw", image=photo)
    
    def update_result(self, img, op_name):
        self.result = img
        self.show_image(self.canvas_res, img)
        self.status.config(text=f"{op_name} - {self.status.cget('text').split(' - ')[0] if ' - ' in self.status.cget('text') else self.status.cget('text')}")
    
    def to_gray(self, img=None):
        src = img if img is not None else self.original
        if src is None: return None
        if len(src.shape) == 2: return src
        return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # ==================== 1. POINT OPERATIONS ====================
    def op_addition(self):
        if self.original is None: return
        result = cv2.add(self.original, 50)
        self.update_result(result, "Addition (+50)")

    def op_add_image(self):
        if self.original is None: return
        path = filedialog.askopenfilename(title="Select second image to add", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if path:
            img2 = self.load_image_file(path)
            if img2 is not None:
                h, w = self.original.shape[:2]
                img2 = cv2.resize(img2, (w, h))
                result = cv2.add(self.original, img2)
                self.update_result(result, "Addition (Image + Image)")
            else:
                messagebox.showerror("Error", "Cannot read second image")

    def op_subtraction(self):
        if self.original is not None:
            self.update_result(cv2.subtract(self.original, 60), "Subtraction (-60)")
    
    def op_division(self):
        if self.original is not None:
            self.update_result((self.original / 2).astype(np.uint8), "Division (÷2)")
    
    def op_complement(self):
        if self.original is not None:
            self.update_result(cv2.bitwise_not(self.original), "Complement")
    
    # ==================== 2. COLOR OPERATIONS ====================
    def op_change_red(self):
        if self.original is not None and len(self.original.shape) == 3:
            res = self.original.copy()
            res[:,:,2] = np.clip(res[:,:,2] + 80, 0, 255)
            self.update_result(res, "Change Red Lighting")
    
    def op_swap_rg(self):
        if self.original is not None and len(self.original.shape) == 3:
            res = self.original.copy()
            res[:,:,[1,2]] = res[:,:,[2,1]]
            self.update_result(res, "Swap R ↔ G")
    
    def op_elim_red(self):
        if self.original is not None and len(self.original.shape) == 3:
            res = self.original.copy()
            res[:,:,2] = 0
            self.update_result(res, "Eliminate Red")
    
    # ==================== 3. HISTOGRAM ====================
    def op_hist_stretch(self):
        if self.original is not None:
            gray = self.to_gray()
            mn, mx = gray.min(), gray.max()
            if mx > mn:
                stretched = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
                self.update_result(stretched, "Histogram Stretching")
    
    def op_hist_equal(self):
        if self.original is not None:
            self.update_result(cv2.equalizeHist(self.to_gray()), "Histogram Equalization")
    
    def op_show_hist(self):
        if self.original is not None:
            gray = self.to_gray()
            plt.figure(figsize=(8, 5), facecolor='#1e1e1e')
            plt.hist(gray.ravel(), bins=256, range=(0,255), color='#007acc', alpha=0.7)
            plt.title('Image Histogram', color='white', fontsize=12)
            plt.xlabel('Pixel Intensity', color='white')
            plt.ylabel('Frequency', color='white')
            plt.gca().set_facecolor('#2d2d2d')
            plt.tick_params(colors='white')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    # ==================== 4. NEIGHBORHOOD FILTERS ====================
    def op_average(self):
        if self.original is not None:
            self.update_result(cv2.blur(self.original, (5,5)), "Average Filter")
    
    def op_laplacian(self):
        if self.original is not None:
            lap = cv2.Laplacian(self.to_gray(), cv2.CV_64F)
            self.update_result(np.abs(lap).astype(np.uint8), "Laplacian Filter")
    
    def op_maximum(self):
        if self.original is not None:
            kernel = np.ones((5,5), np.uint8)
            self.update_result(cv2.dilate(self.original, kernel), "Maximum Filter")
    
    def op_minimum(self):
        if self.original is not None:
            kernel = np.ones((5,5), np.uint8)
            self.update_result(cv2.erode(self.original, kernel), "Minimum Filter")
    
    def op_median(self):
        if self.original is not None:
            self.update_result(cv2.medianBlur(self.original, 5), "Median Filter")
    
    def op_mode(self):
        if self.original is None:
            return
        gray = self.to_gray()
        if gray is None:
            return
        result = generic_filter(
            gray,
            lambda x: np.bincount(x.astype(np.uint8)).argmax(),
            size=3
        )
        self.update_result(result.astype(np.uint8), "Mode Filter")
    
    # ==================== 5. IMAGE RESTORATION ====================
    def op_add_sp(self):
        if self.original is not None:
            noisy = self.original.copy()
            n = int(self.original.size * 0.03)
            coords = [np.random.randint(0, s, n) for s in self.original.shape[:2]]
            noisy[coords[0], coords[1]] = 255
            coords = [np.random.randint(0, s, n) for s in self.original.shape[:2]]
            noisy[coords[0], coords[1]] = 0
            self.update_result(noisy, "Salt & Pepper Added")
    
    def op_add_gauss(self):
        if self.original is not None:
            noise = np.random.normal(0, 25, self.original.shape)
            noisy = np.clip(self.original.astype(float) + noise, 0, 255).astype(np.uint8)
            self.update_result(noisy, "Gaussian Noise Added")
    
    def op_restore_average(self):
        if self.result is not None:
            self.update_result(cv2.blur(self.result, (3,3)), "Restored: Average")
    
    def op_restore_median(self):
        if self.result is not None:
            self.update_result(cv2.medianBlur(self.result, 3), "Restored: Median")
    
    def op_restore_outlier(self):
        if self.result is not None:
            src = self.result
            if len(src.shape) == 3:
                src = self.to_gray(src)
            mean = cv2.blur(src, (3,3)).astype(float)
            mask = np.abs(src.astype(float) - mean) > 40
            res = src.copy().astype(float)
            res[mask] = mean[mask]
            self.update_result(res.astype(np.uint8), "Restored: Outlier Method")
    
    # ==================== 6. SEGMENTATION ====================
    def op_global_thresh(self):
        if self.original is not None:
            self.slider_frame.pack(fill=tk.X, pady=5)
            def apply_thresh(val):
                val = int(val)
                self.slider_label.config(text=f"Threshold: {val}")
                _, thresh = cv2.threshold(self.to_gray(), val, 255, cv2.THRESH_BINARY)
                self.update_result(thresh, f"Global Threshold (T={val})")
            self.threshold_slider.config(command=apply_thresh)
            apply_thresh(127)
    
    def op_otsu_thresh(self):
        if self.original is not None:
            _, thresh = cv2.threshold(self.to_gray(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.update_result(thresh, "Automatic (Otsu) Threshold")
            self.slider_frame.pack_forget()
    
    def op_adaptive_thresh(self):
        if self.original is not None:
            thresh = cv2.adaptiveThreshold(self.to_gray(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            self.update_result(thresh, "Adaptive Threshold")
            self.slider_frame.pack_forget()
    
    # ==================== 7. EDGE DETECTION ====================
    def op_sobel_x(self):
        if self.original is not None:
            sobel = cv2.Sobel(self.to_gray(), cv2.CV_64F, 1, 0, ksize=3)
            self.update_result(np.abs(sobel).astype(np.uint8), "Sobel (Horizontal)")
    
    def op_sobel_y(self):
        if self.original is not None:
            sobel = cv2.Sobel(self.to_gray(), cv2.CV_64F, 0, 1, ksize=3)
            self.update_result(np.abs(sobel).astype(np.uint8), "Sobel (Vertical)")
    
    def op_sobel_combined(self):
        if self.original is not None:
            gray = self.to_gray()
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sx**2 + sy**2)
            self.update_result(np.clip(mag, 0, 255).astype(np.uint8), "Sobel Combined")
    
    # ==================== 8. MORPHOLOGY ====================
    def op_dilate(self):
        if self.original is not None:
            kernel = np.ones((5,5), np.uint8)
            self.update_result(cv2.dilate(self.original, kernel), "Dilation")
    
    def op_erode(self):
        if self.original is not None:
            kernel = np.ones((5,5), np.uint8)
            self.update_result(cv2.erode(self.original, kernel), "Erosion")
    
    def op_opening(self):
        if self.original is not None:
            kernel = np.ones((5,5), np.uint8)
            self.update_result(cv2.morphologyEx(self.original, cv2.MORPH_OPEN, kernel), "Opening")
    
    # ==================== 9. BOUNDARY EXTRACTION ====================
    def op_boundary_internal(self):
        if self.original is not None:
            gray = self.to_gray()
            kernel = np.ones((3,3), np.uint8)
            eroded = cv2.erode(gray, kernel)
            boundary = cv2.subtract(gray, eroded)
            self.update_result(boundary, "Internal Boundary")
    
    def op_boundary_external(self):
        if self.original is not None:
            gray = self.to_gray()
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(gray, kernel)
            boundary = cv2.subtract(dilated, gray)
            self.update_result(boundary, "External Boundary")
    
    def op_boundary_gradient(self):
        if self.original is not None:
            kernel = np.ones((3,3), np.uint8)
            gradient = cv2.morphologyEx(self.to_gray(), cv2.MORPH_GRADIENT, kernel)
            self.update_result(gradient, "Morphological Gradient")

    def save_result(self):
        if self.result is None:
            messagebox.showwarning("No Result", "No filtered image available to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff")
            ],
            title="Save filtered image"
        )
        if path:
            # Try OpenCV first (fast), but on Windows cv2.imwrite may fail for Unicode paths.
            try:
                ok = cv2.imwrite(path, self.result)
            except Exception:
                ok = False

            if ok:
                messagebox.showinfo("Saved", f"Filtered image saved to:\n{path}")
                return

            # Fallback to PIL which handles Unicode paths reliably
            try:
                arr = self.result
                if arr is None:
                    raise ValueError("No image array to save")
                if arr.ndim == 2:
                    img_pil = Image.fromarray(arr)
                else:
                    # Convert BGR (OpenCV) to RGB for PIL
                    img_pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                img_pil.save(path)
                messagebox.showinfo("Saved", f"Filtered image saved to:\n{path}")
                return
            except Exception as e:
                messagebox.showerror("Save Failed", f"Unable to save the filtered image.\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DIPApp(root)
    root.mainloop()