import os
from sklearn.model_selection import train_test_split
import json

class BirdsDS(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
        self.class_counts = {}

        for subfolder in self.subfolders:
            self.files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.wav')]

        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.all_set = []

        for subfolder in self.subfolders:
            dir_name = os.path.join(*subfolder.split('/')[-2:])
            class_name = subfolder.split('/')[-1].lower()
            self.class_counts[class_name] = len(os.listdir(subfolder))

            files = os.listdir(subfolder)
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

            files = [os.path.join(dir_name, f) for f in files]
            train_files = [os.path.join(dir_name, f) for f in train_files]
            val_files = [os.path.join(dir_name, f) for f in val_files]
            test_files = [os.path.join(dir_name, f) for f in test_files]

            self.all_set.extend(files)
            self.train_set.extend(train_files)
            self.val_set.extend(val_files)
            self.test_set.extend(test_files)

        self.save_sets_to_files(os.path.dirname(os.path.dirname(self.root_path)) + '/../meta-v02')

    def save_sets_to_files(self, root_path):
        # 创建一个以 root_path 为根的文件夹
        save_folder = os.path.join(root_path, self.root_path[-1])
        os.makedirs(save_folder, exist_ok=True)

        # 将文件保存到该文件夹中
        with open(os.path.join(save_folder, 'all_set.txt'), 'w') as f:
            f.write('\n'.join(self.all_set))

        with open(os.path.join(save_folder, 'train_set.txt'), 'w') as f:
            f.write('\n'.join(self.train_set))

        with open(os.path.join(save_folder, 'val_set.txt'), 'w') as f:
            f.write('\n'.join(self.val_set))

        with open(os.path.join(save_folder, 'test_set.txt'), 'w') as f:
            f.write('\n'.join(self.test_set))

        with open(os.path.join(save_folder, 'class_counts.txt'), 'w') as f:
            for class_name, count in self.class_counts.items():
                f.write(f'{class_name}: {count}\n')


def process_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    if 'recordings' not in data:
        print(f"Error: 'recordings' key not found in {json_file}.")
        return

    recordings = data['recordings']

    if not isinstance(recordings, list):
        print(f"Error: 'recordings' should be a list in {json_file}.")
        return

    if not recordings:
        print(f"No recordings found in {json_file}.")
        return

    # Extract lng, lat, and name from each recording
    recording_info = [(rec['lng'], rec['lat'], rec['file-name']) for rec in recordings]

    if not all(isinstance(info, tuple) and len(info) == 3 for info in recording_info):
        print(f"Error: Each recording should have 'lng', 'lat', and 'name' keys in {json_file}.")
        return

    return recording_info

def merge_and_analyze_folder(folder_path, filename):
    min_lng_info = None
    max_lng_info = None

    # Iterate over subfolders in the main folder
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Check if the item in the main folder is a directory
        if os.path.isdir(subfolder_path):
            json1_path = os.path.join(subfolder_path, filename)

            # Check if json1.txt exists in the current subfolder
            if os.path.exists(json1_path):
                # Process json1.txt and get recording information
                recording_info = process_json(json1_path)

                if recording_info:
                    # Update min_lng_info and max_lng_info based on the current subfolder
                    if min_lng_info is None or min_lng_info[0] > min(recording_info, key=lambda x: x[0])[0]:
                        min_lng_info = min(recording_info, key=lambda x: x[0])

                    if max_lng_info is None or max_lng_info[0] < max(recording_info, key=lambda x: x[0])[0]:
                        max_lng_info = max(recording_info, key=lambda x: x[0])

    if min_lng_info and max_lng_info:
        print(f"Minimum Longitude: {min_lng_info[0]}, Filename: {min_lng_info[2]}")
        print(f"Maximum Longitude: {max_lng_info[0]}, Filename: {max_lng_info[2]}")
    
if __name__ == '__main__':
    root_path = 'D:/AnimalSound/Code/db3v_bird_vocalization/Data/dataset/3'
    
    audio_dataset = BirdsDS(root_path)
    
    # folder_path = '/nas/staff/data_work/Xin/birds-xie/3areas-10birds/json/'  

    # for id in range(1, 4):
    #     filename = f'json{id}.txt'
    #     print(f'Region {id}')
    #     merge_and_analyze_folder(folder_path, filename)   