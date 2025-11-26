import cv2
import sys
import numpy as np
import os

def main():
    """
    Captures video from a webcam or file and generates a binary
    foreground mask (white foreground, black background).
    
    The masked video is saved to 'output/masked_video.mp4'.
    """
    
    # --- 1. Set Video Source ---
    # Use 0 for the default webcam
    #video_source = 0
    
    # Or, uncomment the line below to use a video file
    video_source = "D:\\MTech Notes\\Sem1\\Soft Computing\\Project(SI)\\Project\\Video_Generation_Complete.mp4"

    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'.")
            if video_source == 0:
                print("Is your webcam connected and enabled?")
            else:
                print("Does the video file exist at that path?")
            return
    except Exception as e:
        print(f"An error occurred while trying to open the video source: {e}")
        return

    # --- 1.5. Set Up Output Video Writer ---
    output_folder = "output"
    output_filename = os.path.join(output_folder, "masked_video.mp4")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # We will initialize the writer object on the first frame
    writer = None

    # --- 2. Create Background Subtractor ---
    # We use MOG2 (Mixture of Gaussians), which is a robust algorithm
    # that adapts to changing lighting.
    #
    # We set detectShadows=False to get a clean binary mask (0 for bg, 255 for fg)
    # as requested. If this was True, shadows would be marked with a gray value (e.g., 127).
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    print("Processing video... Press 'q' to quit the windows.")
    print(f"Will save masked video to: {output_filename}")

    while True:
        # --- 3. Read a Frame ---
        ret, frame = cap.read()

        # If the frame was not read correctly (e.g., end of video file)
        if not ret or frame is None:
            if video_source == 0:
                print("Error reading from webcam.")
            else:
                print("End of video file reached.")
            break

        # --- 3.5. Initialize VideoWriter on the first frame ---
        if writer is None:
            try:
                # Get frame dimensions
                height, width = frame.shape[:2]
                frame_size = (width, height)
                
                # Get FPS from the source
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    print("Warning: Could not get FPS from video source. Defaulting to 20.0 FPS.")
                    fps = 20.0
                
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                # Note: We set isColor=True. We must convert the 1-channel mask
                # to a 3-channel BGR image before writing.
                writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)
                
                if not writer.isOpened():
                    print(f"Error: Could not open video writer for path: {output_filename}")
                    raise IOError("Failed to open VideoWriter")
            
            except Exception as e:
                print(f"Error initializing VideoWriter: {e}")
                # Stop processing if we can't write the video
                break

        # --- 4. Compute the Foreground Mask ---
        # The apply() method updates the background model with the current frame
        # and returns the foreground mask.
        #
        # Because we set detectShadows=False, this mask will only
        # contain 0 (background) and 255 (foreground).
        fgMask = backSub.apply(frame)

        # --- 4.5. Prepare and Save the Masked Frame ---
        if writer is not None:
            # Convert the 1-channel grayscale mask to a 3-channel BGR image
            # so it can be saved in a standard video format.
            fgMask_bgr = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
            
            # Write the frame
            writer.write(fgMask_bgr)

        # --- 5. Display the Results ---
        # Show the original, unprocessed frame
        cv2.imshow("Original Frame", frame)
        
        # Show the final mask (BG=black, FG=white)
        cv2.imshow("Foreground Mask", fgMask)

        # --- 6. Handle User Input ---
        # Wait 30ms for a key press. If 'q' is pressed, exit the loop.
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("'q' pressed. Exiting.")
            break

    # --- 7. Cleanup ---
    print("Releasing resources.")
    if writer is not None:
        writer.release()
        print(f"Masked video saved to {output_filename}")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # A quick check to ensure OpenCV is installed
    try:
        cv2.getVersionString()
    except AttributeError:
        print("Error: OpenCV (cv2) library not found.")
        print("Please install it by running: pip install opencv-python")
        sys.exit(1)
        
    main()

