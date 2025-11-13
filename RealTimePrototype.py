import cv2
import numpy as np
import pandas as pd
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 1. Objective Function (Otsu's Method) ---
# We want to MAXIMIZE the between-class variance.
# Our optimizer will minimize, so we return the negative variance.
def multilevel_otsu_fitness(thresholds, hist):
    """
    Calculates the fitness (negative between-class variance) for a set of thresholds.
    Based on
    """
    thresholds = sorted(thresholds)
    thresholds = [0] + [int(t) for t in thresholds] + [255]
    total_pixels = np.sum(hist)
    
    if total_pixels == 0:
        return 0

    # Calculate global mean
    global_mean = np.dot(np.arange(256), hist) / total_pixels
    
    variance = 0.0
    for i in range(len(thresholds) - 1):
        start_thresh = thresholds[i]
        end_thresh = thresholds[i+1]
        
        class_pixels_indices = np.arange(start_thresh, end_thresh)
        if class_pixels_indices.size == 0:
            continue
            
        class_hist = hist[start_thresh:end_thresh]
        
        # Calculate weight (w_k)
        class_pixels = np.sum(class_hist)
        weight = class_pixels / total_pixels
        
        if class_pixels == 0:
            continue
            
        # Calculate mean (mu_k)
        mean = np.dot(class_pixels_indices, class_hist) / class_pixels
        
        # Add to between-class variance
        variance += weight * ((mean - global_mean) ** 2)

    # Since we want to maximize variance, we return the negative for a minimizer.
    return -variance

# --- 2. Swarm Intelligence Optimizers ---

class StandardPSO:
    """
    A standard Particle Swarm Optimizer for baseline comparison.
    """
    def __init__(self, n_particles, n_dimensions, max_iter, lb, ub, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_dim = n_dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.w, self.c1, self.c2 = w, c1, c2
        self.positions = None
        self.velocities = None
        self.gbest_pos = None

    def optimize(self, fitness_func, hist, warm_start_pos=None):
        # 1. Initialize population
        if warm_start_pos is not None:
            # Warm-start: Center new population around the previous best
            self.positions = np.random.normal(loc=warm_start_pos, scale=20, 
                                             size=(self.n_particles, self.n_dim))
            # Add some new random particles for exploration
            rand_indices = np.random.choice(self.n_particles, self.n_particles // 2, replace=False)
            self.positions[rand_indices] = np.random.uniform(self.lb, self.ub, 
                                                             (self.n_particles // 2, self.n_dim))
            self.positions = np.clip(self.positions, self.lb, self.ub)
        else:
            # Cold start
            self.positions = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dim))
            
        self.velocities = np.zeros((self.n_particles, self.n_dim))
        
        # 2. Evaluate initial population
        pbest_val = np.array([fitness_func(p, hist) for p in self.positions])
        pbest_pos = np.copy(self.positions)
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = np.copy(pbest_pos[gbest_idx])
        gbest_val = pbest_val[gbest_idx]

        # 3. Run iterations
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity (standard PSO equation)
                r1, r2 = np.random.rand(self.n_dim), np.random.rand(self.n_dim)
                cognitive_vel = self.c1 * r1 * (pbest_pos[i] - self.positions[i])
                social_vel = self.c2 * r2 * (gbest_pos - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_vel + social_vel
                
                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # Evaluate fitness
                current_val = fitness_func(self.positions[i], hist)
                
                # Update pbest
                if current_val < pbest_val[i]:
                    pbest_val[i] = current_val
                    pbest_pos[i] = np.copy(self.positions[i])
                    
                    # Update gbest
                    if current_val < gbest_val:
                        gbest_val = current_val
                        gbest_pos = np.copy(self.positions[i])
        
        self.gbest_pos = gbest_pos
        return sorted(self.gbest_pos)


class HPSOSCAOptimizer:
    """
    Implementation of the HPSO-SCA hybrid algorithm.
    Based on
    This implementation uses SCA to update the global best (gbest) particle's
    position to enhance global exploration, while the rest of the swarm
    uses standard PSO for local exploitation.
    """
    def __init__(self, n_particles, n_dimensions, max_iter, lb, ub, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_dim = n_dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.w, self.c1, self.c2 = w, c1, c2
        self.positions = None
        self.velocities = None
        self.gbest_pos = None

    def optimize(self, fitness_func, hist, warm_start_pos=None):
        # 1. Initialize population
        if warm_start_pos is not None:
            # Warm-start
            self.positions = np.random.normal(loc=warm_start_pos, scale=20, 
                                             size=(self.n_particles, self.n_dim))
            rand_indices = np.random.choice(self.n_particles, self.n_particles // 2, replace=False)
            self.positions[rand_indices] = np.random.uniform(self.lb, self.ub, 
                                                             (self.n_particles // 2, self.n_dim))
            self.positions = np.clip(self.positions, self.lb, self.ub)
        else:
            # Cold start
            self.positions = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dim))
            
        self.velocities = np.zeros((self.n_particles, self.n_dim))
        
        # 2. Evaluate initial population
        pbest_val = np.array([fitness_func(p, hist) for p in self.positions])
        pbest_pos = np.copy(self.positions)
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = np.copy(pbest_pos[gbest_idx])
        gbest_val = pbest_val[gbest_idx]

        # 3. Run iterations
        for t in range(self.max_iter):
            
            # --- SCA Part (Global Exploration for gbest) ---
            # SCA updates the position of the best particle based on sine/cosine
            # This helps the "gbest" explore globally
            r1 = 2 * np.pi * np.random.rand() # r1
            r2 = 2 * np.random.rand()       # r2 (amplitude)
            r3 = np.random.rand()           # r3 (prob switch)
            r4 = np.random.uniform(-2, 2)   # r4 (weight for current pos)

            sca_new_pos = np.zeros(self.n_dim)
            if r3 < 0.5:
                # Sine update
                sca_new_pos = self.positions[gbest_idx] + r2 * np.sin(r1) * np.abs(r4 * gbest_pos - self.positions[gbest_idx])
            else:
                # Cosine update
                sca_new_pos = self.positions[gbest_idx] + r2 * np.cos(r1) * np.abs(r4 * gbest_pos - self.positions[gbest_idx])
            
            sca_new_pos = np.clip(sca_new_pos, self.lb, self.ub)
            sca_new_val = fitness_func(sca_new_pos, hist)
            
            # Greedy selection: update gbest if SCA found a better position
            if sca_new_val < gbest_val:
                gbest_val = sca_new_val
                gbest_pos = np.copy(sca_new_pos)
            
            # --- PSO Part (Local Exploitation for all particles) ---
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(self.n_dim), np.random.rand(self.n_dim)
                cognitive_vel = self.c1 * r1 * (pbest_pos[i] - self.positions[i])
                social_vel = self.c2 * r2 * (gbest_pos - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_vel + social_vel
                
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                current_val = fitness_func(self.positions[i], hist)
                
                if current_val < pbest_val[i]:
                    pbest_val[i] = current_val
                    pbest_pos[i] = np.copy(self.positions[i])
                    
                    if current_val < gbest_val:
                        gbest_val = current_val
                        gbest_pos = np.copy(self.positions[i])
        
        self.gbest_pos = gbest_pos
        return sorted(self.gbest_pos)

# --- 3. Baseline Algorithm (Traditional Otsu) ---
def run_baseline_otsu(gray_frame):
    """
    Runs standard Otsu's method from OpenCV for baseline comparison.
   
    """
    thresh_val, segmented_mask = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [thresh_val], segmented_mask

# --- 4. Helper Functions ---
def load_ground_truth_masks(mask_folder_path):
    """
    Loads all mask images from a folder into a dictionary.
    Ensures they are grayscale (0-255).
    """
    print(f"Loading ground truth masks from {mask_folder_path}...")
    masks = {}
    if not os.path.exists(mask_folder_path):
        print(f"Warning: Mask folder not found at {mask_folder_path}. Proceeding without metrics.")
        return None
        
    mask_files = sorted([f for f in os.listdir(mask_folder_path) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    for i, fname in enumerate(mask_files):
        mask = cv2.imread(os.path.join(mask_folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask file {fname}")
            continue
        # Ensure mask is binary (0 or 255) for fair comparison
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks[i] = mask
        
    print(f"Loaded {len(masks)} ground truth masks.")
    return masks

def apply_thresholds_to_frame(gray_frame, thresholds, n_thresholds):
    """
    Applies the final thresholds to create a binary mask.
    This example just takes the highest-level object.
    """
    thresholds = sorted(thresholds)
    
    # For a 1-threshold (2-class) problem:
    if n_thresholds == 1:
        thresh_low = int(thresholds[0])
    # For a 2-threshold (3-class) problem, we take the top class:
    elif n_thresholds == 2:
        thresh_low = int(thresholds[1])
    else:
        # Default to highest threshold for simplicity
        thresh_low = int(thresholds[-1])
        
    _, segmented_mask = cv2.threshold(gray_frame, thresh_low, 255, cv2.THRESH_BINARY)
    return segmented_mask

def calculate_metrics(gt_mask, segmented_mask):
    """
    Calculates PSNR and SSIM between the ground truth and the result.
   
    """
    if gt_mask is None:
        return {'psnr': None, 'ssim': None}
        
    # Ensure masks are the same shape
    if gt_mask.shape != segmented_mask.shape:
        h, w = gt_mask.shape
        segmented_mask = cv2.resize(segmented_mask, (w, h))
        
    # Data range is 0-255
    current_psnr = psnr(gt_mask, segmented_mask, data_range=255)
    current_ssim = ssim(gt_mask, segmented_mask, data_range=255)
    
    return {'psnr': current_psnr, 'ssim': current_ssim}

# --- 5. Main Experiment Runner ---
def run_experiment(video_path, mask_path, algorithms_to_run, n_thresholds):
    """
    Main batch-processing loop to run all algorithms on the video
    and save the results to CSV files.
    """
    
    gt_masks = load_ground_truth_masks(mask_path)
    
    for alg_config in algorithms_to_run:
        alg_name = alg_config['name']
        alg_instance = alg_config['instance']
        is_warm_start = alg_config['warm_start']
        
        print(f"\n--- Running Experiment: {alg_name} ---")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            continue
            
        frame_num = 0
        results_log = []
        last_gbest_pos = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get the corresponding ground truth mask
            current_gt_mask = gt_masks.get(frame_num, None)
            
            start_time = time.time()
            
            if alg_name == "Traditional_Otsu":
                thresholds, segmented_mask = run_baseline_otsu(gray_frame)
            else:
                # This is an SI algorithm
                hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).flatten()
                
                start_sol = last_gbest_pos if is_warm_start else None
                thresholds = alg_instance.optimize(multilevel_otsu_fitness, hist, start_sol)
                
                if is_warm_start:
                    last_gbest_pos = thresholds # Save for next frame
                    
                segmented_mask = apply_thresholds_to_frame(gray_frame, thresholds, n_thresholds)
            
            end_time = time.time()
            proc_time = end_time - start_time
            if proc_time == 0.0:
            # Set FPS to a very high number (or 0) if time was too small to measure
                fps = np.inf 
            else:
                fps = 1.0 / proc_time
            
            # Calculate quality metrics
            metrics = calculate_metrics(current_gt_mask, segmented_mask)
            
            # Log the data
            log_entry = {
                'frame': frame_num,
                'algorithm': alg_name,
                'fps': fps,
                'proc_time_ms': proc_time * 1000,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'thresholds': thresholds
            }
            results_log.append(log_entry)
            
            if frame_num % 50 == 0:
                print(f"  Frame {frame_num}: FPS={fps:.2f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}")
                
            frame_num += 1
            
        cap.release()
        
        # Save results to CSV
        output_filename = f"results_{alg_name}.csv"
        df = pd.DataFrame(results_log)
        df.to_csv(output_filename, index=False)
        print(f"--- Experiment {alg_name} complete. Results saved to {output_filename} ---")


# --- 6. Main execution block ---
if __name__ == "__main__":
    
    # --- *** YOUR ACTION REQUIRED *** ---
    # 1. Provide the path to your video file
    VIDEO_FILE_PATH = "D:\MTech Notes\Sem1\Soft Computing\Project(SI)\Project\Video_Generation_Complete.mp4" # e.g., "my_drone_footage.mp4"
    
    # 2. Provide the path to the FOLDER containing your ground-truth masks
    #    Masks should be named like: 0000.png, 0001.png, 0002.png ...
    MASK_FOLDER_PATH = "D:\MTech Notes\Sem1\Soft Computing\Project(SI)\Project\Mask2" # e.g., "my_video_masks/"
    
    # 3. Set the number of thresholds you want to find (e.g., 1, 2, or 3)
    NUM_THRESHOLDS = 1
    
    # --- End of Configuration ---

    # Check if paths are set
    if VIDEO_FILE_PATH == "path/to/your/video.mp4":
        print("="*50)
        print("ERROR: Please set 'VIDEO_FILE_PATH' and 'MASK_FOLDER_PATH'")
        print("in the __main__ block at the bottom of the script.")
        print("="*50)
    else:
        # Define optimizer parameters
        N_PARTICLES = 30
        MAX_ITER = 20 # Keep iterations low for speed
        LOWER_BOUND = 0
        UPPER_BOUND = 255
        
        # Define all algorithms to test
        algorithms_to_run = [
            {
                "name": "HPSOSCA_WarmStart",
                "instance": HPSOSCAOptimizer(N_PARTICLES, NUM_THRESHOLDS, MAX_ITER, LOWER_BOUND, UPPER_BOUND),
                "warm_start": True
            },
            {
                "name": "HPSOSCA_ColdStart",
                "instance": HPSOSCAOptimizer(N_PARTICLES, NUM_THRESHOLDS, MAX_ITER, LOWER_BOUND, UPPER_BOUND),
                "warm_start": False
            },
            {
                "name": "StandardPSO_WarmStart",
                "instance": StandardPSO(N_PARTICLES, NUM_THRESHOLDS, MAX_ITER, LOWER_BOUND, UPPER_BOUND),
                "warm_start": True
            },
            {
                "name": "Traditional_Otsu",
                "instance": None, # Not an optimizer
                "warm_start": False
            }
        ]

        # Run the full experiment
        run_experiment(VIDEO_FILE_PATH, MASK_FOLDER_PATH, algorithms_to_run, NUM_THRESHOLDS)


#Code for comparing the outputs:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- 1. Configuration ---
# 
# List the exact names of the CSV files generated by your experiment script
CSV_FILES = [
    "results_HPSOSCA_WarmStart.csv",
    "results_HPSOSCA_ColdStart.csv",
    "results_StandardPSO_WarmStart.csv", # Add this if you ran it
    "results_Traditional_Otsu.csv"
]

# Optional: Define nicer names for the plot legends
ALGORITHM_NAMES_MAP = {
    "HPSOSCA_WarmStart": "Proposed (HPSOSCA Warm)",
    "HPSOSCA_ColdStart": "HPSOSCA Cold",
    "StandardPSO_WarmStart": "PSO Warm",
    "Traditional_Otsu": "Otsu Baseline"
}

# Folder to save the plots
PLOT_OUTPUT_FOLDER = "comparison_plots"

# --- End of Configuration ---


def load_and_combine_data(csv_files):
    """Loads multiple CSV files into a single pandas DataFrame."""
    all_data = []
    for file in csv_files:
        if os.path.exists(file):
            print(f"Loading {file}...")
            df = pd.read_csv(file)
            # Add algorithm name based on filename if not present (optional)
            if 'algorithm' not in df.columns:
                 # Extract name from filename like "results_ALGORITHM_NAME.csv"
                alg_name = file.replace("results_", "").replace(".csv", "")
                df['algorithm'] = alg_name
            all_data.append(df)
        else:
            print(f"Warning: CSV file not found - {file}")
            
    if not all_data:
        print("Error: No data loaded. Cannot proceed.")
        return None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Apply nicer names if map is provided
    if ALGORITHM_NAMES_MAP:
        combined_df['algorithm_display'] = combined_df['algorithm'].map(ALGORITHM_NAMES_MAP)
    else:
        combined_df['algorithm_display'] = combined_df['algorithm']
        
    # Handle potential infinite FPS values from the fix
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print("\nData loaded and combined successfully.")
    print(combined_df.head())
    print(f"\nAlgorithms found: {combined_df['algorithm_display'].unique()}")
    
    return combined_df

def plot_metrics_over_frames(df, output_folder):
    """Plots PSNR and SSIM over frame numbers for each algorithm."""
    if df is None or df.empty:
        print("No data to plot metrics over frames.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    # Plot PSNR over frames
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='frame', y='psnr', hue='algorithm_display', marker='o', markersize=4, linestyle='--')
    plt.title('PSNR Over Frames per Algorithm')
    plt.xlabel('Frame Number')
    plt.ylabel('PSNR (dB)')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "psnr_over_frames.png"), dpi=300)
    print("Saved psnr_over_frames.png")
    plt.close()

    # Plot SSIM over frames
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='frame', y='ssim', hue='algorithm_display', marker='o', markersize=4, linestyle='--')
    plt.title('SSIM Over Frames per Algorithm')
    plt.xlabel('Frame Number')
    plt.ylabel('SSIM')
    plt.ylim(0, 1.05) # SSIM is typically between 0 and 1
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "ssim_over_frames.png"), dpi=300)
    print("Saved ssim_over_frames.png")
    plt.close()

def plot_average_comparisons(df, output_folder):
    """Plots bar charts comparing average FPS, PSNR, and SSIM."""
    if df is None or df.empty:
        print("No data to plot average comparisons.")
        return
        
    # Calculate average metrics, ignoring NaN values
    avg_metrics = df.groupby('algorithm_display')[['fps', 'psnr', 'ssim']].mean().reset_index()
    
    # Sort for consistent plotting order (optional)
    avg_metrics = avg_metrics.sort_values(by='algorithm_display')
    
    print("\nAverage Metrics per Algorithm:")
    print(avg_metrics)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Average FPS
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_metrics['algorithm_display'], avg_metrics['fps'], color=sns.color_palette("viridis", len(avg_metrics)))
    plt.title('Average Frames Per Second (FPS) Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Average FPS')
    plt.xticks(rotation=15, ha='right')
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval):
             plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') # va: vertical alignment
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_fps_comparison.png"), dpi=300)
    print("Saved average_fps_comparison.png")
    plt.close()

    # Plot Average PSNR
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_metrics['algorithm_display'], avg_metrics['psnr'], color=sns.color_palette("viridis", len(avg_metrics)))
    plt.title('Average Peak Signal-to-Noise Ratio (PSNR) Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Average PSNR (dB)')
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval):
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_psnr_comparison.png"), dpi=300)
    print("Saved average_psnr_comparison.png")
    plt.close()

    # Plot Average SSIM
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_metrics['algorithm_display'], avg_metrics['ssim'], color=sns.color_palette("viridis", len(avg_metrics)))
    plt.title('Average Structural Similarity Index (SSIM) Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Average SSIM')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval):
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_ssim_comparison.png"), dpi=300)
    print("Saved average_ssim_comparison.png")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Create output directory for plots
    if not os.path.exists(PLOT_OUTPUT_FOLDER):
        os.makedirs(PLOT_OUTPUT_FOLDER)
        print(f"Created directory for plots: {PLOT_OUTPUT_FOLDER}")
        
    # 1. Load and combine data from all CSV files
    combined_data = load_and_combine_data(CSV_FILES)
    
    if combined_data is not None:
        # 2. Plot metrics over frames
        plot_metrics_over_frames(combined_data, PLOT_OUTPUT_FOLDER)
        
        # 3. Plot average comparisons
        plot_average_comparisons(combined_data, PLOT_OUTPUT_FOLDER)
        
        print(f"\nPlotting complete. Check the '{PLOT_OUTPUT_FOLDER}' folder.")