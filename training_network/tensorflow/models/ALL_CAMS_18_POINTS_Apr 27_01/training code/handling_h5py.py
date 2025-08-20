import matplotlib
matplotlib.use("TkAgg")  # Use an interactive backend
import h5py
import os
import matplotlib.pyplot as plt


path = "/cs/labs/tsevi/lior.kotlar/amitai-s-thesis/inference_datasets/movie_full/movie_1_10_4410_ds_3tc_7tj.h5"
if not os.path.exists(path):
    print("file doesen't exist")
    exit(1)

def read_database_file_for_training():
    with h5py.File(path, "r") as f:
        print("Keys: %s" %list(f.keys()))
        box = f['box']
        print(box.shape)
        oneinstance = box[0]
        print(oneinstance.shape)
        onecamera = oneinstance[3]
        print(onecamera.shape)
        onetimepoint = onecamera[:,:,1]
        print(onetimepoint.shape)
        plt.imshow(onetimepoint, cmap="gray", vmin=0, vmax=1)  # vmin and vmax ensure correct grayscale display
        plt.axis("off")  # Hide axes
        plt.show()


def read_data_file_for_inference():
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print("Keys: %s" %list(f.keys()))
        for key in keys:
            print(f'{key}: {f[key].shape}')


def crop_dataset():
    cropped_file_path = "C:\\Users\\lior.kotlar\\Documents\\Lior Studies\\Lab\\amitai-s-thesis\\inference_datasets\\cropped_dataset.h5"
    crop_size = 8
    with h5py.File(path, "r") as originalfile:
        with h5py.File(cropped_file_path, "w") as cropped_file:
            # for key in originalfile.keys():
            #     print(f'cropping {key}, size: {originalfile[key].shape}')
            #     subsetdata = originalfile[key][:crop_size]
            #     print(subsetdata.shape)
            #     cropped_file.create_dataset(key, subsetdata)
            sub1 = originalfile['best_frames_mov_idx'][:, :crop_size]
            print(sub1.shape)
            cropped_file.create_dataset('best_frames_mov_idx', data=sub1)
            sub2 = originalfile['box'][:crop_size]
            cropped_file.create_dataset('box', data=sub2)
            sub3 = originalfile['cropzone'][:crop_size]
            cropped_file.create_dataset('cropzone', data=sub3)
            sub4 = originalfile['frameInds'][:crop_size]
            cropped_file.create_dataset('frameInds', data=sub4)


read_data_file_for_inference()
crop_dataset()