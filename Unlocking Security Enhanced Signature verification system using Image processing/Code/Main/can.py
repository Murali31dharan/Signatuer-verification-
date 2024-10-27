import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Function to perform edge detection
def edge(image1,image2):
    def edge_detection(pic):
        # Convert image to grayscale
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)  # Adjust threshold values as needed

        return edges


    # Perform edge detection
    edges1 = edge_detection(image1)
    edges2 = edge_detection(image2)

    # Create a Tkinter window
    root = tk.Tk()
    root.title('Signature and Edge Comparison')

    # Plot images and their edges for comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Original signature 1
    axes[0, 0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Signature 1')
    axes[0, 0].axis('off')

    # Edge-detected signature 1
    axes[0, 1].imshow(edges1, cmap='gray')
    axes[0, 1].set_title('Edges of Signature 1')
    axes[0, 1].axis('off')

    # Original signature 2
    axes[1, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Signature 2')
    axes[1, 0].axis('off')

    # Edge-detected signature 2
    axes[1, 1].imshow(edges2, cmap='gray')
    axes[1, 1].set_title('Edges of Signature 2')
    axes[1, 1].axis('off')


    # Create a Tkinter canvas to embed the plot
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()