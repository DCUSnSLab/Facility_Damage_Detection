import os
import shutil

def move_matching_images(img1_file, img2_folders, result_folder):
    facilities_path = os.path.join(path, facilities)
    img1_folder_path = os.path.join(facilities_path, img1_file)
    result_folder_path = os.path.join(facilities_path, result_folder)

    img1_list = os.listdir(img1_folder_path)
    img2_lists = [os.path.join(facilities_path, folder) for folder in img2_folders]

    for file_name in img1_list:
        for img2_folder_path in img2_lists:
            if file_name in os.listdir(img2_folder_path):
                source_file_path = os.path.join(img2_folder_path, file_name)
                destination_file_path = os.path.join(result_folder_path, file_name)

                shutil.move(source_file_path, destination_file_path)
                print(f"Moved '{file_name}' from {img2_folder_path} to {result_folder_path}")
                break  # 이미지를 찾았으므로 다음 이미지로 이동합니다.

if __name__ == '__main__':
    path = "/mnt/home/jo/Facility_Damage_Detection/Dataset/camera_dataset"
    facilities = 'prevention'
    img1_file = 'diagonal/train/images'
    img2_folders = ['front_sheer/valid/images', 'front_sheer/test/images', 'front_sheer/train/images']
    result_folder = 'overlap/train/images'

    move_matching_images(img1_file, img2_folders, result_folder)
