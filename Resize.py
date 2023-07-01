from PIL import Image
import os

# direktori dengan file gambar
dir_path = 'dataset/Muhamad Fajar Yudhistira Herjanto'

# loop untuk setiap file di dalam direktori
for filename in os.listdir(dir_path):
    # cek apakah file merupakan file gambar
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # buka file gambar
        image = Image.open(os.path.join(dir_path, filename))

        # resize gambar
        image = image.resize((250, 250))

        # simpan gambar yang sudah di-resize
        image.save(os.path.join(dir_path, filename.split('.')[0] + '_250.jpg'))