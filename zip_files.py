import zipfile
import os

#filepath =  "./ss_fewsome/outputs"
filepath =  "./ss_fewsome/downloads"
#save_path = "./ss_fewsome/outputs_zipped.zip"
save_path = "./ss_fewsome/downloads_zipped.zip"

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, arcname=rel_path)
    print(f"Folder '{folder_path}' zipped successfully into '{output_path}'.")

if __name__ == "__main__":
    zip_folder(filepath, save_path)
