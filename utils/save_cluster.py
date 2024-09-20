import os
import shutil

def convert_to_img_path(data_path):
    for idx in range(len(data_path)):
        data_path_li = data_path[idx].split("/")
        data_path_li[-2] = "images"
        data_path_li[-1] = data_path_li[-1][:-4] + ".jpg"
        data_path[idx] = rf"/".join(data_path_li)
    
    return data_path

def save_cluster(label, data_path, dataset_name, n_clusters):
    save_path = f"result/{dataset_name}/{n_clusters:02d}_clusters_expr"
    
    labels_num = ["01st", "02nd", "03rd", "04th", "05th", "06th", "07th", "08th", "09th", "10th"]
    labels_path = []
    
    for n in labels_num[:n_clusters]:
        os.makedirs(f"{save_path}/{n}_cluster", exist_ok=True)
        labels_path.append(f"{save_path}/{n}_cluster")
    
    # label = (n, )
    if dataset_name != "image":
        data_path = convert_to_img_path(data_path)
        
    for i, path in zip(label, data_path):
        to_path = f"{labels_path[i]}/{path.split('/')[-1]}"
        from_path = path
        
        shutil.copy(from_path, to_path)