import csv
import os
import cv2
import torch
import numpy as np
import glob
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.getcwd(), script_dir + '/ros/src/yolov7-ros/src'))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

# Configuration
is_ripe = True
polar = False
fixedView = False

def find_latest_folder(base_dir):
    """Finds the most recently modified subdirectory in given base directory."""
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return os.path.join(base_dir, max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d))))

def run_detection():
    # Device setup
    device =  select_device('0' if torch.cuda.is_available() else 'cpu') 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'ros/src/yolov7-ros/weights/apples_weights.pt')
    print(model_path)
    model = attempt_load(model_path, map_location=device).eval().to(device)
    class_names = model.names
    print(f"Model loaded with {len(class_names)} classes: {class_names}")

    # Path configuration
    base_dir = 'datasets/'
    base_dir = os.path.join(script_dir, base_dir)
    base_dir = os.path.join(base_dir, "polar" if polar else "cartesian")
    base_dir = os.path.join(base_dir, "fixed_view" if fixedView else "variable_view")
    base_dir = os.path.join(base_dir, "ripe" if is_ripe else "raw")
    
    latest_folder = find_latest_folder(base_dir)
    print(f"Processing images in: {latest_folder}")

    # Input/output paths
    input_csv_path = glob.glob(os.path.join(latest_folder, "*.csv"))[0]
    output_image_dir = os.path.join(latest_folder, "detections")
    os.makedirs(output_image_dir, exist_ok=True)
    output_csv_path = os.path.join(latest_folder, 'MeansAndCountsDatasetCNN.csv')

    # Read input CSV
    rows, image_paths = [], []
    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            rows.append(row)
            image_paths.append(row[-1])

    # Batch processing function
    def process_batch(image_paths, batch_size=8):
        all_detections, all_original_images = [], []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images, original_images, ratio_pads = [], [], []
            
            for path in batch_paths:
                img = cv2.imread(path)
                if img is None: continue
                original_img = img.copy()
                img_resized, ratio, pad = letterbox(img, 640, auto=False)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                img_tensor = torch.from_numpy(img_resized).float().to(device) / 255.0
                batch_images.append(img_tensor)
                original_images.append(original_img)
                ratio_pads.append((ratio, pad))
            
            if not batch_images: continue
            with torch.no_grad():
                pred = model(torch.stack(batch_images, 0))[0]
            detections = non_max_suppression(pred, 0.6, 0.7)
            
            for j, det in enumerate(detections):
                orig_img = original_images[j]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(batch_images[0].shape[1:], det[:, :4], orig_img.shape, ratio_pads[j])
                all_detections.append(det)
                all_original_images.append(orig_img)
        return all_detections, all_original_images

    # Process images and save results
    detections, original_images = process_batch(image_paths)
    results = []
    for det, img, path in zip(detections, original_images, image_paths):
        class_conf = {cls: [] for cls in range(len(class_names))}
        if det is not None:
            for *xyxy, conf, cls in det:
                cls_id = int(cls)
                class_conf[cls_id].append(conf.item())
                plot_one_box(xyxy, img, label=f"{class_names[cls_id]} {conf:.2f}",color=(0, 255, 0), line_thickness=2)
        
        output_path = os.path.join(output_image_dir, os.path.basename(path))
        cv2.imwrite(output_path, img)
        results.append((
            [np.median(class_conf[cls]) if class_conf[cls] else 0.5 for cls in range(len(class_names))],
            [len(class_conf[cls]) for cls in range(len(class_names))]
        ))

    # Write detection results CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['x', 'y', 'yaw'] + [f"{n}_mean" for n in class_names] + [f"{n}_count" for n in class_names]
        writer.writerow(header)
        for row, (means, counts) in zip(rows, results):
            new_row = row[:3] + means + counts
            writer.writerow(new_row)
    
    print("Detection completed successfully.")
    return latest_folder

def process_and_visualize(latest_folder):
    input_csv = os.path.join(latest_folder, 'MeansAndCountsDatasetCNN.csv')
    output_csv = os.path.join(latest_folder, 'SurrogateDatasetCNN.csv')

    # Read data and calculate weighted scores
    data = {'x': [], 'y': [], 'yaw': [], 'classes': {}}
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        class_names = [col[:-5] for col in header if col.endswith('_mean')]
        for cls in class_names: data['classes'][cls] = {'means': [], 'counts': []}
        
        for row in reader:
            data['x'].append(float(row[0]))
            data['y'].append(float(row[1]))
            data['yaw'].append(float(row[2]))
            for i, cls in enumerate(class_names):
                    mean_col = f"{cls}_mean"
                    count_col = f"{cls}_count"
                    mean_idx = header.index(mean_col)
                    count_idx = header.index(count_col)
                    data['classes'][cls]['means'].append(float(row[mean_idx]))
                    data['classes'][cls]['counts'].append(float(row[count_idx]))

    def weight_value(n_elements, mean_score, midpoint=5, steepness=10):
        """
        Calculates a weighted score for each class.
        
        The weighting smoothly transitions the score between the raw mean_score and a default of 0.5,
        based on the number of elements.
        """
        return np.ceil(100 *(mean_score - 0.5) * (0.5 + 0.5 * np.tanh(steepness * (n_elements - midpoint)))) /100 + 0.5

    # Calculate weighted scores
    weighted = {}
    for cls in class_names:
        means = np.array(data['classes'][cls]['means'])
        counts = np.array(data['classes'][cls]['counts'])
        weighted[cls] = weight_value(counts, means)

    # Write filtered CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'yaw'] + [f"{n}_score" for n in class_names])
        for i in range(len(data['x'])):
            row = [data['x'][i], data['y'][i], data['yaw'][i]] + [weighted[cls][i] for cls in class_names]
            writer.writerow(row)

    # Create visualizations
    fig_cones = make_subplots(rows=1, cols=len(class_names), specs=[[{'type': 'scene'}]*len(class_names)])
    fig_surface = make_subplots(rows=1, cols=len(class_names), specs=[[{'type': 'scene'}]*len(class_names)])
    
    for idx, cls in enumerate(class_names):
        scores = weighted[cls]
        yaw = np.array(data['yaw'])
        u, v = np.cos(yaw)*scores, np.sin(yaw)*scores
        
        # Cone plot
        fig_cones.add_trace(go.Cone(
            x=data['x'], y=data['y'], z=scores,
            u=u, v=v, w=np.zeros_like(scores),
            name=cls, showscale=False
        ), 1, idx+1)
        
        # Surface plot
        fig_surface.add_trace(go.Scatter3d(
            x=data['x'], y=data['y'], z=scores,
            mode='markers', marker=dict(size=5, color=scores, colorscale='Viridis'),
            name=cls
        ), 1, idx+1)

    # Show plots
    fig_cones.update_layout(title_text="Class Confidence Distributions - Cone Plots")
    fig_surface.update_layout(title_text="Class Confidence Distributions - Surface Plots")
    pio.show(fig_cones)
    pio.show(fig_surface)
    print("Visualization completed successfully.")

if __name__ == "__main__":
    latest_folder = run_detection()
    process_and_visualize(latest_folder)