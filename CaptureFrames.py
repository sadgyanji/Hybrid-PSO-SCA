#Video to Frames
import cv2
import os

# --- 1. Configuration ---
# 
# *** ACTION REQUIRED: Change these two variables ***
#
# Put the path to your video file here
VIDEO_PATH = VIDEO_PATH = "D:\\MTech Notes\\Sem1\\Soft Computing\\Project(SI)\\Project\\output\\masked_video.mp4"
  
#
# Name of the folder where the frames will be saved
OUTPUT_FOLDER = "D:\MTech Notes\Sem1\Soft Computing\Project(SI)\Project\Mask2"
#
# --- End of Configuration ---


def export_frames(video_path, output_folder):
    """
    Reads a video file and saves every frame as an image in an output folder.
    """
    
    # 2. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    else:
        print(f"Directory already exists: {output_folder}. Files may be overwritten.")

    # 3. Open the video file
    cap = cv2.VideoCapture(video_path)

    # 4. Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 5. Loop through the video and save frames
    frame_count = 0
    while True:
        # Read one frame from the video capture object
        # 'ret' is a boolean: True if the frame was read, False if at the end of the video
        ret, frame = cap.read()

        if not ret:
            # End of video
            print("Successfully reached the end of the video.")
            break

        # 6. Create the output filename
        # We use 5-digit padding (e.g., "00001", "00002")
        # This is CRITICAL for keeping your frames in the correct order.
        frame_filename = f"frame_{frame_count:05d}.png"
        output_path = os.path.join(output_folder, frame_filename)

        # 7. Save the frame to the output folder
        # Using PNG is recommended as it's a lossless format.
        cv2.imwrite(output_path, frame)
        
        # Print progress to the console
        if frame_count % 30 == 0: # Print a message every 30 frames
             print(f"Saved {frame_filename}...")

        frame_count += 1

    # 8. Clean up and release the video file
    cap.release()
    print(f"\nDone. Successfully exported {frame_count} frames to the '{output_folder}' folder.")

# --- Run the script ---
if __name__ == "__main__":
    if VIDEO_PATH == "path/to/your/video.mp4":
        print("="*50)
        print("ERROR: Please update the 'VIDEO_PATH' variable in the script")
        print("to point to your actual video file.")
        print("="*50)
    else:
        export_frames(VIDEO_PATH, OUTPUT_FOLDER)