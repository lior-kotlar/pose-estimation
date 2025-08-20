import matplotlib
matplotlib.use("TkAgg")  # Use an interactive backend
import h5py
import os
import matplotlib.pyplot as plt


path = "C:\\Users\\lior.kotlar\\Documents\\trainset_random_16_pnts.h5"
if not os.path.exists(path):
    print("file doesen't exist")
    exit(1)


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