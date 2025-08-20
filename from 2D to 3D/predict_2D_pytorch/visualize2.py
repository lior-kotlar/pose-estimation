import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.widgets import Slider, Button
from skimage import morphology
from scipy.spatial import ConvexHull
import h5py
import plotly.graph_objects as go
import plotly.io as pio

class Visualizer:
    @staticmethod
    def show_predictions_one_cam(box, points_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_all_cams(box, points_2D):
        movie = Visualizer.get_display_box(box)
        # movie = box[..., [1, 1, 1]]
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, i])))  # create a color array
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='o')  #
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprojections_one_cam(box, points_2D, reprojections_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        reprojections = reprojections_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            ax.scatter(*reprojections[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='d')  # scatter points on image
            for point, point_reprojected in zip(points[frame, camera[0]], reprojections[frame, camera[0]]):
                ax.plot(*zip(point, point_reprojected), color='yellow')
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprejections_all_cams(box, points_2D, points_2D_reprojected):
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        points_reprojected = points_2D_reprojected[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, :])))  # create a color array for points_2D
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='o')  # display points_2D
                ax.scatter(*points_reprojected[frame, i].T, edgecolors=colors, facecolors='none', marker='d')
                # Add a line between every corresponding point in points and points_reprojected
                for point, point_reprojected in zip(points[frame, i], points_reprojected[frame, i]):
                    ax.plot(*zip(point, point_reprojected), color='yellow')

            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_points_in_3D(points):
        # Assuming points is your (N, M, 3) array
        # Calculate the limits of the plot
        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        # connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
        #                (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 7),
        #                (14, 15)]
        connections = [(0,1), (1,2), (2,3),   (3,4),   (4,5),   (5,6),  (0,6),
                        (8,9), (9,10),(10,11), (11,12), (12,13), (13,14), (8,14),
                        (7, 15),
                        (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            points_to_plot = points[[frame], :, :]
            for i in range(num_points):
                ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')
            # Reset the limits of the plot
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Function to handle keyboard events
        def handle_key_event(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        fig.canvas.mpl_connect('key_press_event', handle_key_event)

        # Initial plot
        update(0)
        plt.show()

    @staticmethod
    def show_points_in_3D_projections(points):
        # Assuming points is your (N, M, 3) array
        # Calculate the limits of the plot
        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points

        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (7, 15),
                       (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        # Create the play button
        axplay = plt.axes([0.85, 0.95, 0.1, 0.03])
        button = Button(axplay, 'Play', color='lightgoldenrodyellow', hovercolor='0.975')

        # Create the speed control
        axspeed = plt.axes([0.05, 0.95, 0.1, 0.03], facecolor='lightgoldenrodyellow')
        speed = Slider(axspeed, 'Speed', 0.5, 100, valinit=10, valstep=0.5)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            points_to_plot = points[frame, :, :]
            for i in range(num_points):
                ax.scatter(points_to_plot[i, 0], points_to_plot[i, 1], points_to_plot[i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], points_to_plot[[i, j], 1], points_to_plot[[i, j], 2], c='k')
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            fig.canvas.draw_idle()

            # Plot the projections on the walls
            # XY plane
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], z_min, c=color_array, alpha=0.2)
            # XZ plane
            ax.scatter(points_to_plot[:, 0], y_min, points_to_plot[:, 2], c=color_array, alpha=0.2)
            # YZ plane
            ax.scatter(x_min, points_to_plot[:, 1], points_to_plot[:, 2], c=color_array, alpha=0.2)

            # Plot the connections on the projections
            # XY plane
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], points_to_plot[[i, j], 1], [z_min, z_min], c='b', alpha=0.5)
            # XZ plane
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], [y_min, y_min], points_to_plot[[i, j], 2], c='b', alpha=0.5)
            # YZ plane
            for i, j in connections:
                ax.plot([x_min, x_min], points_to_plot[[i, j], 1], points_to_plot[[i, j], 2], c='b', alpha=0.5)
            fig.canvas.draw_idle()

        def play(event):
            # Play the animation by updating the slider value
            nonlocal slider
            for i in range(slider.valmin, slider.valmax + 1):
                slider.set_val(i)
                # Adjust the speed according to the slider value
                plt.pause(1 / speed.val)

        # Connect the update functions to the widgets
        slider.on_changed(update)
        button.on_clicked(play)
        speed.on_changed(update)

        # Function to handle keyboard events
        def handle_key_event(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        fig.canvas.mpl_connect('key_press_event', handle_key_event)

        # Initial plot
        update(0)
        plt.show()

    @staticmethod
    def show_points_and_wing_planes_3D(points, planes):
        # planes is an array (num_frames, num_planes, 4)
        # Assuming points is your (num_frames, num_joints, 3)
        # Calculate the limits of the plot

        num_planes = planes.shape[1]
        POINTS_2_PLANES = [[0, 1, 2, 3, 4, 5, 6],
                           [8, 9, 10, 11, 12, 13, 14]]
        POINTS_4_PLANES = [[0, 1, 2, 3],   [3, 4, 5, 6],
                           [8, 9, 10, 11], [11, 12, 13, 14]]
        # POINTS = POINTS_2_PLANES if num_planes == 2 else if  POINTS_4_PLANES
        if num_planes == 2:
            POINTS = POINTS_2_PLANES
        elif num_planes == 4:
            POINTS = POINTS_4_PLANES
        else:
            POINTS = [np.arange(18)]

        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        connections = [(0,1), (1,2), (2,3),   (3,4),   (4,5),   (5,6),  (0,6),
                        (8,9), (9,10),(10,11), (11,12), (12,13), (13,14), (8,14),
                        (7, 15),
                        (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            for i in range(num_points):
                ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')
            # Reset the limits of the plot

            # add plane
            for plane_num in range(num_planes):
                wing_points = points[frame, POINTS[plane_num], :]
                min_x, min_y, min_z = np.amin(wing_points[[7, 15, -2, -1], :], axis=0)
                max_x, max_y, max_z = np.amax(wing_points[[7, 15, -2, -1], :], axis=0)
                a, b, c, d = planes[frame, plane_num, :]
                xx, yy = np.meshgrid(np.linspace(min_x, max_x, 10), np.linspace(min_y, max_y, 10))
                zz = (-a*xx - b*yy - d)/c
                zz[zz > max_z] = max_z
                zz[zz < min_z] = min_z
                ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Plane')

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Function to handle keyboard events
        def handle_key_event(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        fig.canvas.mpl_connect('key_press_event', handle_key_event)

        # Initial plot
        update(0)

        plt.show()

    @staticmethod
    def get_display_box(box):
        masks = box[..., -2:]
        num_frames, num_cams, _, _, num_masks = masks.shape
        for frame in range(num_frames):
            for cam in range(num_cams):
                for wing in range(num_masks):
                    mask = masks[frame, cam, :, :, wing]
                    dilated = morphology.binary_dilation(mask)
                    eroded = morphology.binary_erosion(mask)
                    perimeters = dilated ^ eroded
                    masks[frame, cam, :, :, wing] = perimeters
        box[..., -2:] = masks
        box[..., -2:] += box[..., [1, 1]]
        movie = box[..., [1, 3, 4]]
        movie[movie > 1] = 1
        return movie

    @staticmethod
    def display_movie_from_box(box):
        movie = box
        movie = box[..., [1, 3, 4]]
        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def display_movie_from_path(path):
        movie = Visualizer.get_box(path)

        # movie = box[..., [1, 1, 1]]
        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def get_box(path):
        box = h5py.File(path, "r")["/box"][:]
        box = np.transpose(box, (0, 3, 2, 1))
        x1 = np.expand_dims(box[:, :, :, 0:3], axis=1)
        x2 = np.expand_dims(box[:, :, :, 3:6], axis=1)
        x3 = np.expand_dims(box[:, :, :, 6:9], axis=1)
        x4 = np.expand_dims(box[:, :, :, 9:12], axis=1)
        box = np.concatenate((x1, x2, x3, x4), axis=1)
        return box

    @staticmethod
    def create_movie_plot(com, x_body, y_body, points_3D, start_frame, save_path):
        left_tip = points_3D[:, 3, :]
        right_tip = points_3D[:, 3 + 8, :]
        frame0 = 340 - start_frame
        frame_40_ms = frame0 + 40 * 16
        interval = 70
        # Create traces
        marker_size = 2
        mode = 'lines'
        trace_com = go.Scatter3d(x=com[:, 0], y=com[:, 1], z=com[:, 2], mode='markers', name='Center of Mass',
                                 marker=dict(size=marker_size), line=dict(width=5))
        trace_left_tip = go.Scatter3d(x=left_tip[:, 0], y=left_tip[:, 1], z=left_tip[:, 2], mode=mode,
                                      name='left tip', marker=dict(size=marker_size), line=dict(width=5))
        trace_right_tip = go.Scatter3d(x=right_tip[:, 0], y=right_tip[:, 1], z=right_tip[:, 2], mode=mode,
                                       name='right tip', marker=dict(size=marker_size), line=dict(width=5))
        # Create markers
        marker_start = go.Scatter3d(x=[com[0, 0]], y=[com[0, 1]], z=[com[0, 2]], mode='markers',
                                    marker=dict(size=8, color='red'), name='start')
        marker_start_dark = go.Scatter3d(x=[com[frame0, 0]], y=[com[frame0, 1]], z=[com[frame0, 2]], mode='markers',
                                         marker=dict(size=8, color='orange'), name='start dark')
        marker_40_ms = go.Scatter3d(x=[com[frame_40_ms, 0]], y=[com[frame_40_ms, 1]], z=[com[frame_40_ms, 2]],
                                    mode='markers', marker=dict(size=8, color='yellow'), name='+ 40 ms')
        # Create quiver plot
        name = 'body yaw pitch'
        size = 0.003

        quiver_x_body, qx_points = Visualizer.get_orientation_scatter(com, interval, 'x body', size, x_body,
                                                                      points_color=['black', 'yellow'], width=2)
        quiver_y_body, qy_points = Visualizer.get_orientation_scatter(com, interval, 'y body', size, y_body,
                                                                      points_color=['green', 'blue'])
        # Create a figure and add the traces
        fig = go.Figure(
            data=[trace_com, trace_left_tip, trace_right_tip, marker_start, marker_start_dark, marker_40_ms,
                  quiver_x_body, quiver_y_body, qx_points, qy_points])
        # Update the scene
        scene = dict(camera=dict(eye=dict(x=1., y=1, z=1)),
                     xaxis=dict(nticks=10),
                     yaxis=dict(nticks=10),
                     zaxis=dict(nticks=10),
                     aspectmode='data'  # This makes the axes equal
                     )
        fig.update_scenes(scene)
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.01,
                font=dict(
                    size=16,  # This enlarges the labels
                    color="black"
                )
            )
        )
        # Write the figure to an HTML file
        pio.write_html(fig, save_path)

    @staticmethod
    def get_orientation_scatter(com, interval, name, size, x_body, points_color=['orange', 'blue'], width=5):
        x_quiver = []
        y_quiver = []
        z_quiver = []
        x_points = []
        y_points = []
        z_points = []
        marker_colors = []  # Initialize an empty list for individual marker colors

        for i in range(0, len(com), interval):
            start = [com[i, j] - size / 6 * x_body[i, j] for j in range(3)]
            end = [com[i, j] + size / 6 * x_body[i, j] for j in range(3)]
            x_quiver.extend([start[0], end[0], None])
            y_quiver.extend([start[1], end[1], None])
            z_quiver.extend([start[2], end[2], None])

            x_points.extend([start[0], end[0]])
            y_points.extend([start[1], end[1]])
            z_points.extend([start[2], end[2]])

            # Specify colors for each point: start point in orange, end point in blue
            marker_colors.extend(points_color)

        points = go.Scatter3d(x=x_points, y=y_points, z=z_points, mode='markers',
                              marker=dict(color=marker_colors, size=5),  # Apply the color array here
                              name=name + ' points')

        quiver_x_body = go.Scatter3d(x=x_quiver, y=y_quiver, z=z_quiver, mode='lines',
                                     line=dict(color='black', width=width),
                                     name=name)

        return quiver_x_body, points


if __name__ == '__main__':
    # display movie

    # movie_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies\mov62\movie_62_160_1888_ds_3tc_7tj.h5"
    # Visualizer.display_movie_from_path(movie_path)

    # display 3D points
    # points_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies\mov6\movie_6_100_498_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Feb 13\points_3D.npy"
    # points_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies\mov6\points_3D_smoothed_ensemble.npy"
    # points_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies\mov10\movie_10_130_1666_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Feb 09\points_3D.npy"
    # points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\points_3D_ensemble.npy"
    # points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_ensemble.npy"
    # points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_smoothed_ensemble.npy"
    # points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_ensemble.npy"
    # points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov61_d\points_3D_ensemble.npy"
    # points_3D = np.load(points_path)
    # Visualizer.show_points_in_3D(points_3D[:200])

    # display box and 2D predictions
    path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\movie_10_130_1666_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Mar 05_01\predicted_points_and_box_reprojected.h5"
    box = h5py.File(path, "r")["/box"][:100]
    points_2D = h5py.File(path, "r")["/positions_pred"][:100]
    Visualizer.show_predictions_all_cams(box, points_2D)
