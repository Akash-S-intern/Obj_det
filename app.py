import os
import shutil
import zipfile
import cv2
import matplotlib.pyplot as plt

def clone_and_setup_yolov5():
    # Clone YOLOv5 repository and install dependencies if not already set up
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5')
    os.system('pip install -r yolov5/requirements.txt')

def train_model():
    # Define the correct path for the data.yaml file based on the folder structure
    data_yaml_path = "yolov5/content/data.yaml"
    
    # Ensure the file exists before proceeding with training
    if not os.path.exists(data_yaml_path):
        print(f"Error: '{data_yaml_path}' does not exist.")
        return

    # Make sure the yolov5s.pt weights file is also present
    weights_path = "yolov5s.pt"
    if not os.path.exists(weights_path):
        print(f"Error: '{weights_path}' weights file not found.")
        return

    # Train the model (adjust the epochs, batch size, or other parameters as needed)
    os.system(f'python yolov5/train.py --img 640 --batch 16 --epochs 50 --data {data_yaml_path} --weights {weights_path}')

def detect_objects():
    # Define the path for test image and results directory
    test_image_path = "yolov5/content/test.jpg"
    results_dir = "yolov5/content/detection_results"

    # Ensure test image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image '{test_image_path}' not found.")
        return

    # Check for the latest training run directory dynamically
    runs_dir = "yolov5/runs/train"
    if not os.path.exists(runs_dir):
        print(f"Error: '{runs_dir}' directory not found. Ensure training has been completed.")
        return
    
    # Retrieve the latest run directory for weights
    last_run = sorted([d for d in os.listdir(runs_dir) if d.startswith('exp')], reverse=True)
    if len(last_run) == 0:
        print(f"Error: No experiment folders found in '{runs_dir}'.")
        return
    
    weights_path = f'{runs_dir}/{last_run[0]}/weights/best.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found. Ensure training completed successfully.")
        return

    # Perform object detection
    os.system(f'python yolov5/detect.py --weights {weights_path} --source {test_image_path} '
              f'--iou-thres 0.2 --conf-thres 0.2 --save-txt --save-conf --project {results_dir} --name results --exist-ok')

def view_results():
    # Find the latest result folder dynamically
    results_dir = "yolov5/content/detection_results/results"
    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' not found.")
        return

    result_images = [f for f in os.listdir(results_dir) if f.endswith('.jpg')]

    if len(result_images) == 0:
        print("Error: No result images found in the results directory.")
        return

    for result_image in result_images:
        result_image_path = os.path.join(results_dir, result_image)
        output_image = cv2.imread(result_image_path)

        if output_image is None:
            print(f"Error: Unable to load image '{result_image}'.")
            continue

        # Convert BGR to RGB for Matplotlib and display the image
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.axis('off')
        plt.show()

def zip_folder():
    # Define the folder to zip and its destination
    folder_to_zip = "yolov5/content/detection_results"
    
    # Ensure the folder exists before zipping
    if not os.path.exists(folder_to_zip):
        print(f"Error: Folder '{folder_to_zip}' does not exist.")
        return None

    zip_name = "yolov5/content/detection_results"

    # Create a ZIP archive of the folder
    zip_path = shutil.make_archive(zip_name, 'zip', folder_to_zip)
    print(f"Folder '{folder_to_zip}' has been zipped as '{zip_path}'")
    return zip_path

def delete_folder():
    # Define the folder to delete
    folder_to_delete = "yolov5/content/detection_results"
    
    # Ensure the folder exists before deleting
    if not os.path.exists(folder_to_delete):
        print(f"Error: Folder '{folder_to_delete}' does not exist.")
        return

    # Delete the folder and its contents
    shutil.rmtree(folder_to_delete, ignore_errors=True)
    print(f"Folder '{folder_to_delete}' has been deleted.")

# Main process
if __name__ == "__main__":
    # Step 1: Clone and set up YOLOv5 if not already done
    clone_and_setup_yolov5()
    
    # Step 2: Train the model
    train_model()
    
    # Step 3: Perform object detection
    detect_objects()
    
    # Step 4: View the results using OpenCV and Matplotlib
    view_results()
    
    # Step 5: Zip and delete the results folder
    zip_path = zip_folder()
    if zip_path:
        delete_folder()
