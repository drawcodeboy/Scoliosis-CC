import os

def count_cluster():
    expr_li = ['seq', 'mask', 'image']
    
    for expr in expr_li:
        base_path = rf"result/{expr}"
        for folder_name in os.listdir(base_path):
            expr_path = rf"{base_path}/{folder_name}" # txt file here
            
            with open(rf"{expr_path}/sample_count.txt", 'w+') as f:
                for idx, cluster_name in enumerate(os.listdir(expr_path)):
                    if cluster_name[-3:] == 'txt':
                        continue
                    cluster_path = rf"{expr_path}/{cluster_name}"
                    # print(cluster_path, len(os.listdir(cluster_path)))
                    
                    if idx < len(os.listdir(expr_path)) - 1:
                        f.write(rf"{cluster_name}: {len(os.listdir(cluster_path)):04d} samples" + '\n')
                    else:
                        f.write(rf"{cluster_name}: {len(os.listdir(cluster_path)):04d} samples")
    
count_cluster()