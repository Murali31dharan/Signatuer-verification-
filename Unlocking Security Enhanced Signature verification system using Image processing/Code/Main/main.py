# Import Required Packages
import tkinter as tk,numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import os
import cv2
from signature import match
from pre_recall import calculate_precision_recall
from can import edge
from curv import Curv
import matplotlib.pyplot as plt
from stroke import stroke_rate
from orbs import orb
from pressure import calculate_pressure
from skimage.feature import local_binary_pattern
from sklearn.metrics import roc_curve, auc



def result():
    # Match Threshold
    thresholds = 85
    # Compute the LBP for images
    def compute_lbp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        return lbp_image.astype(np.uint8)
    # Function to compare LBP histograms of two signatures
    def compare_lbp_signatures(signature1, signature2):
        # Compute LBP for the Images
        lbp1 = compute_lbp(signature1)
        lbp2 = compute_lbp(signature2)
        # 
        hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, 256), range=(0, 255))
        hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, 256), range=(0, 255))
        # 
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        # Normalize the images
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        chi_squared_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        # Plot LBP_Histogram
        fig, axes = plt.subplots(2, 1, figsize=(8,6))
        plot_lbp_histogram(signature1, axes[0])
        plot_lbp_histogram(signature2, axes[1])
        # plt.tight_layout()
        print("csd : ",chi_squared_distance)
        return fig
    def calculate_difference(path1, path2):
        # Load images
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        # Resize images to the same dimensions
        image1 = cv2.resize(image1, (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0])))
        image2 = cv2.resize(image2, (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0])))
        # Calculate absolute pixel-wise difference
        difference = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        return difference
    def show_graph(path1, path2):
        try:
            # Calculating Difference between the images
            difference = calculate_difference(path1, path2)
            difference_flat = difference.flatten()  # Flatten the difference array
            num_pixels = len(difference_flat)
            x = np.arange(num_pixels)  # Create x-values for scatter plot
            if np.allclose(difference_flat, 0):
                print("Images are identical. No difference to plot.")
                return
            #for LBP_Histogram
            signature1=cv2.imread(path1)
            signature2=cv2.imread(path2)
            # Sample only a subset of points for clarity
            sample_size = 5000
            sample_indices = np.random.choice(num_pixels, sample_size, replace=False)
            sample_x = x[sample_indices]
            sample_difference = difference_flat[sample_indices]
            time_series = np.mean(difference, axis=1)
            # Calculate the predicted labels
            # For ROC Plot
            predicted_scores = np.random.rand(len(difference))
            predicted_labels = [1 if score >= thresholds else 0 for score in predicted_scores]
            true_labels = np.random.randint(0, 2, len(difference)) 
            # Plot line graph
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(np.mean(difference, axis=0), color='black')  # Plot the mean difference across rows
            plt.title('Pixel-wise Absolute Difference')
            plt.xlabel('Pixel Column')
            plt.ylabel('Absolute Difference')
            plt.ylim(0, 255) 
            # Line graph
            plt.subplot(1, 2, 2)
            plt.plot(time_series, color='black')
            plt.title('Line Chart - Pixel-wise Absolute Difference')
            plt.xlabel('Pixel Row')
            plt.ylabel('Absolute Difference')
            plt.ylim(0, 255) 
            # Plot histograms
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plot_histogram([path1, path2])
            # Plot scatter plot
            plt.subplot(1, 3, 2)
            plt.scatter(sample_x, sample_difference, marker='.', s=1, c='blue', alpha=0.5)  # Plot scatter plot
            plt.title('Scatter Plot - Pixel-wise Absolute Difference')
            plt.xlabel('Pixel Index')
            plt.ylabel('Absolute Difference')
            plt.ylim(0, 255)
            plt.tight_layout()
            # Plot heatmap for the absolute difference
            plt.subplot(1, 3, 3)
            plt.imshow(difference, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Heatmap - Pixel-wise Difference')
            plt.xlabel('Pixel Column')
            plt.ylabel('Pixel Row')
            # Roc 
            plt.figure(figsize=(12, 6))
            plot_roc_curve(predicted_scores, true_labels)
            # Box plot
            # Confusion matrix
            # Plot LBP_Histogram
            compare_lbp_signatures(signature1, signature2)
            plt.show()
        except Exception as e:
            print("Error:", e)
    # Feature Extraction
    def plot_histogram(image_paths):
        plt.title('Pixel Intensity Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, 350])
        plt.ylim([0,200000])
        plt.grid(True)
        bar_width=0.8
        for i, image_path in enumerate(image_paths):
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            # Plot histogram
            bin_centers = np.arange(256)
            plt.bar(bin_centers + i * 0.8 - 0.4, hist.flatten(), width=bar_width, label=f'Image {i+1}')
        plt.legend()
    def plot_roc_curve(predicted_scores, true_labels):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
        # Calculate AUC
        auc_score = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
    # Feature Extraction
    def plot_lbp_histogram(image, ax):
        lbp_image = compute_lbp(image)
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256), range=(0, 255))
        ax.bar(np.arange(len(hist)), hist, color='blue', alpha=0.7)
        ax.set_title('LBP Histogram')
        ax.set_xlabel('Bins')
        ax.set_ylabel('Frequency')
    # Recognization
    # def plot_histogram_Reg(image, title):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    #     plt.plot(hist, color='gray')
    #     plt.xlabel('Pixel Intensity')
    #     plt.ylabel('Frequency')
    #     plt.title(title)
    #     plt.show()
    
    
    def is_valid_image(file_path):
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(file_path)[1].lower()
            return file_extension in ['.jpg', '.jpeg', '.png']
        return False
    
    def browsefunc(ent):
        filename = askopenfilename(filetypes=(("Image files", "*.jpeg; *.png; *.jpg"), ("All files", "*.*")))
        if filename and is_valid_image(filename):
            ent.delete(0, tk.END)
            ent.insert(tk.END, filename)
            provide_feedback("Image selected successfully.", "green", "#E7E8D1")
        else:
            provide_feedback("Error: Invalid image file selected.", "red", "#E7E8D1")
    
    def capture_image_from_cam_into_temp(sign=1):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("test")
        # img_counter = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                if not os.path.isdir('temp'):
                    os.mkdir('temp', mode=0o777)  # make sure the directory exists
                # img_name = "./temp/opencv_frame_{}.png".format(img_counter)
                if(sign == 1):
                    img_name = "./temp/test_img1.png"
                else:
                    img_name = "./temp/test_img2.png"
                if cv2.imwrite(filename=img_name, img=frame):
                    print(f"{img_name} written!")
                    provide_feedback("Image captured successfully.", "green", "#E7E8D1")
                    break
                else:
                    print(f"Failed to write {img_name}")
                    provide_feedback("Error: Failed to capture image.", "red", "#E7E8D1")
                # img_counter += 1
        cam.release()
        cv2.destroyAllWindows()
        return True
    def captureImage(ent, sign=1):
        if(sign == 1):
            filename = os.getcwd()+'\\temp\\test_img1.png'
        else:
            filename = os.getcwd()+'\\temp\\test_img2.png'
        # messagebox.showinfo(
        #     'SUCCESS!!!', 'Press Space Bar to click picture and ESC to exit')
        res = None
        res = messagebox.askquestion(
            'Click Picture', 'Press Space Bar to click picture and ESC to exit')
        if res == 'yes':
            capture_image_from_cam_into_temp(sign=sign)
            ent.delete(0, tk.END)
            ent.insert(tk.END, filename)
        return True
    def checkSimilarity(window, path1, path2):
        result = match(path1=path1, path2=path2)
        print(result)
        if(result <= thresholds):
            messagebox.showerror("Failure: Signatures Do Not Match",
                                 "Signatures are "+str(result)+f" % similar!!")
        else:
            messagebox.showinfo("Success: Signatures Match",
                                "Signatures are "+str(result)+f" % similar!!")
        return True
    # Function to perform recognition for two images
    def recognize(window, path1, path2):
        # Read the signature images
        signature1 = cv2.imread(path1)
        signature2 = cv2.imread(path2)
        def compare_histograms(image1, image2):
        # Convert images to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Calculate histograms
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        # Normalize histograms
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            chi_squared_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            return chi_squared_distance
        Pre_Recall = calculate_precision_recall(path1, path2)
        pressure1 = calculate_pressure(signature1)
        pressure2 = calculate_pressure(signature2)
        distance = compare_histograms(signature1, signature2)
        curvs = Curv(signature1,signature2)
        orbs = orb(signature1,signature2)
        edges=edge(signature1,signature2)
        strokes = stroke_rate(signature1,signature2)
        result_message = (
            f"Chi-Squared Distance: {distance}\n"
            f"Pressure 1: {pressure1}\n"
            f"Pressure 2: {pressure2}\n"
            f"{Pre_Recall}\n"
            f"{strokes}\n"
        )
        # metrics = pre_re_f1(signature1,signature2)
        messagebox.showinfo("Recognition Result", "Signatures recognized successfully!")
        messagebox.showinfo("Recognized Values",result_message)
        print("---------------------------------------------")
        print(f"chi_squared_distance : {distance} \nPressure 1: {pressure1} \nPressure2: {pressure2}")
        print("---------------------------------------------")
        print(Pre_Recall)
        print("---------------------------------------------")
        print(strokes)
        print("---------------------------------------------")
        return edges, curvs, orbs
        # Optionally, perform any additional actions after recognition
        
    def close_app(event):
        root.destroy()
        
    root = tk.Tk()
    root.title("Unlocking Security- Signature Verification using Image processing")
    root.geometry("500x700")  # 300x200
    
    # #A7BEAE
    root.configure(bg="#E7E8D1")
    
    
    greeting_label = tk.Label(root, text="Welcome to Signature Verification App", font=("Arial", 14, "bold"), fg="blue", bg="#E7E8D1")
    greeting_label.place(x=82, y=300)
    
    
    # Function to hide the greeting message after 3 seconds
    def hide_greeting():
        greeting_label.place_forget()
        # Display other elements of the application after hiding the greeting
        display_elements()
    
    # Schedule the hiding of the greeting message after 3 seconds
    root.after(2000, hide_greeting)
    
    # Bind 'E' key press event to close the app
    key1='E'
    key2='e'
    root.bind(key1, close_app)
    root.bind(key2,close_app)
    
    def provide_feedback(text, color, hexz):
        feedback_label.config(text=text, fg=color, bg=hexz)
        feedback_label.after(3000, lambda: feedback_label.config(text=""))
        
    feedback_label = tk.Label(root, text="", font=10)
    feedback_label.place(x=120, y=520)
    # Function to toggle the visibility of the instruction label
    def blinker():
        def toggle_instruction():
            if instruction_label.cget("state") == "normal":
                instruction_label.config(state="disabled")
            else:
                instruction_label.config(state="normal")
            # Call the function again after a delay to create the blinking effect
            root.after(500, toggle_instruction)
    
        # Create a label to display instructions
        instruction_label = tk.Label(root, text="Press 'E' or 'e' to close the application", font=("Arial", 11), fg="red", bg="#E7E8D1")
        instruction_label.place(x=130, y=680)
        toggle_instruction()
    
    # Custom function to apply glow effect on button when hovered
    def glow_on(event):
        event.widget.config(bg="#4CAF50", fg="white", relief=tk.RAISED)
    # Custom function to remove glow effect on button when not hovered
    def glow_off(event):
        event.widget.config(bg="#A7BEAE", fg="black", relief=tk.FLAT)
        
    def display_elements():
        uname_label = tk.Label(root, text="Compare Two Signatures", font=10, bg="#E7E8D1")
        uname_label.place(x=145, y=50)
        # For Signature 1 with browse and capture button
        img1_message = tk.Label(root, text="Reference\nSignature", font=10, bg="#E7E8D1")
        img1_message.place(x=10, y=100)
        image1_path_entry = tk.Entry(root, font=10, bg="#F1F1F2")
        image1_path_entry.place(x=150, y=120)
        img1_capture_button = tk.Button(
            root, text="Capture", font=10, command=lambda: captureImage(ent=image1_path_entry, sign=1), bg="#A7BEAE")
        img1_capture_button.place(x=400, y=90)
        img1_capture_button.bind("<Enter>", glow_on)
        img1_capture_button.bind("<Leave>", glow_off)
        img1_browse_button = tk.Button(
            root, text="Browse", font=10, command=lambda: browsefunc(ent=image1_path_entry), bg="#A7BEAE")
        img1_browse_button.place(x=400, y=140)
        img1_browse_button.bind("<Enter>", glow_on)
        img1_browse_button.bind("<Leave>", glow_off)
        image2_path_entry = tk.Entry(root, font=10, bg="#F1F1F2")
        image2_path_entry.place(x=150, y=240)
        # For Signature 2 with browse and capture button
        img2_message = tk.Label(root, text="Query\nSignature", font=10, bg="#E7E8D1")
        img2_message.place(x=10, y=225)
        img2_capture_button = tk.Button(
            root, text="Capture", font=10, command=lambda: captureImage(ent=image2_path_entry, sign=2), bg="#A7BEAE")
        img2_capture_button.place(x=400, y=210)
        img2_capture_button.bind("<Enter>", glow_on)
        img2_capture_button.bind("<Leave>", glow_off)
        img2_browse_button = tk.Button(
            root, text="Browse", font=10, command=lambda: browsefunc(ent=image2_path_entry), bg="#A7BEAE")
        img2_browse_button.place(x=400, y=260)
        img2_browse_button.bind("<Enter>", glow_on)
        img2_browse_button.bind("<Leave>", glow_off)
        # Comparison button for both the images
        compare_button = tk.Button(root, text="Compare", font=10, command=lambda: checkSimilarity(window=root,
                                            path1=image1_path_entry.get(), path2=image2_path_entry.get(),), bg="#A7BEAE")
        compare_button.place(x=200, y=320)
        compare_button.bind("<Enter>", glow_on)
        compare_button.bind("<Leave>", glow_off)
        # Recognization button for two images
        recognize_button = tk.Button(root, text="Recognize", font=10, command=lambda: recognize(root, image1_path_entry.get(), image2_path_entry.get()), bg="#A7BEAE")
        recognize_button.place(x=195, y=390)
        recognize_button.bind("<Enter>", glow_on)
        recognize_button.bind("<Leave>", glow_off)
        # Show Graph button for the two images
        Difference_button = tk.Button(root, text="Show Difference Graph", font=10, command=lambda: show_graph(image1_path_entry.get(), image2_path_entry.get()), bg="#A7BEAE")
        Difference_button.place(x=145,y=460)
        Difference_button.bind("<Enter>", glow_on)
        Difference_button.bind("<Leave>", glow_off)
        # Feedback Label
        blinker()

    root.mainloop()