"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import functools
from collections import OrderedDict
import napari
import vedo
import numpy as np
import os
import inspect
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

import scipy
from scipy import ndimage as ndi

from skimage.feature import blob_log
from skimage import filters
from skimage.exposure import histogram
from skimage.segmentation import watershed
from skimage import measure, morphology, segmentation
from skimage.measure import regionprops_table
from skimage.filters import threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_otsu, \
    threshold_triangle, threshold_yen

from qtpy.QtWidgets import QMainWindow, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QLineEdit, QCheckBox
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QFileDialog

from napari_particle_detection_analysis.widget.checkable_combox import CheckableComboBox

#if TYPE_CHECKING:
#    import napari

'''
try:
    from numpy.typing import NDArray
    NDArrayA = NDArray[Any]
except (ImportError, TypeError):
    NDArray = np.ndarray  # type: ignore[misc]
    NDArrayA = np.ndarray  # type: ignore[misc]
'''

'''
class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
'''


class ParticleCellAnalysis(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.box_contents = QFormLayout()
        _layers = [
            layer for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        _labels = [
            layer for layer in self.viewer.layers
            if isinstance(layer, napari.layers.labels.Labels)
        ]
        _spots = [
            layer for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Points)
        ]
        self.segGroupBox = QGroupBox("Particle Cell Analysis")

        # Particle label layer selection
        self.particle_layer_cb = CheckableComboBox()
        for layer in _layers:
            self.particle_layer_cb.addItem(layer.name)
        # Particle label layer(s) selection
        self.cell_label_layer_cb = QComboBox()
        for label in _labels:
            self.cell_label_layer_cb.addItem(label.name)
        # Measure Colocalization if 2 particle label selected
        self.do_colocalization_measurement_cb = QCheckBox()
        self.selected_edit = QLineEdit("file selected")
        self.file_select_button = QPushButton("Select")
        self.file_select_button.resize(150, 50)
        self.file_select_button.clicked.connect(self.selectFileBox)

        self.do_density_map_cb = QCheckBox()
        self.spot_layer_cb = CheckableComboBox()
        for layer in _spots:
            self.spot_layer_cb.addItem(layer.name)
        self.density_radius = QLineEdit()
        self.density_radius.setText("50")

        apply_button = QPushButton("Analyze")
        apply_button.clicked.connect(self._on_click)

        # Layout
        self.box_contents.addRow("Particle(s) Layer", self.particle_layer_cb)
        self.box_contents.addRow("Cell Label Layer", self.cell_label_layer_cb)
        self.box_contents.addRow("Do Particle Co-localization", self.do_colocalization_measurement_cb)
        layout_file_selection = QFormLayout()
        layout_file_selection.addRow(self.selected_edit, self.file_select_button)
        self.box_contents.addRow("Select Original File", layout_file_selection)

        self.box_contents.addRow("Do Spot Density Map", self.do_density_map_cb)
        self.box_contents.addRow("Spot Layer", self.spot_layer_cb)
        self.box_contents.addRow("Density Radius", self.density_radius)

        self.box_contents.addWidget(apply_button)

        self.segGroupBox.setLayout(self.box_contents)
        box_layout = QVBoxLayout()
        # setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None
        box_layout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(box_layout)
        self.layout().addWidget(self.segGroupBox)

    def _on_click(self):
        process_analysis_cells_particles_fun(self.viewer, self.particle_layer_cb.currentData(),
                                             self.cell_label_layer_cb.currentText(),
                                             self.do_colocalization_measurement_cb.isChecked(),
                                             self.selected_edit.text(),
                                             self.do_density_map_cb.isChecked(),
                                             self.spot_layer_cb.currentData(),
                                             self.density_radius.text())

    def selectFileBox(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select image source', "./", "All files (*)")
        # fname = QFileDialog(self, "Select FIle", "./", "All files (*)")
        if fname:
            self.selected_edit.setText(str(fname))


class ParticleSegmentationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.box_contents = QFormLayout()
        _layers = [
            layer for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

        self.segGroupBox = QGroupBox("Particle Segmentation")
        thresholds = ['Li', 'Mean', 'Minimum', 'Otsu', 'Triangle''Yen']

        self.particle_layer_cb = CheckableComboBox()
        for layer in _layers:
            self.particle_layer_cb.addItem(layer.name)

        # self.cell_label_layer_cb = QComboBox()
        # self.cell_label_layer_cb.addItem('None')
        # for layer in layers:
        #    self.cell_label_layer_cb.addItem(layer.name)
        # self.nuclei_layer_cb.currentIndexChanged.connect(self.selectionchange)
        # Spot min sigma
        only_int_validator = QIntValidator()
        only_double_validator = QDoubleValidator()
        only_int_validator.setRange(0, 20)
        only_int_validator_large = QIntValidator()
        only_int_validator_large.setRange(0, 65500)
        self.spot_min_gaussian = QLineEdit()
        self.spot_min_gaussian.setToolTip("The minimum standard deviation for Gaussian "
                                          "kernel. Keep this low to detect smaller blobs.")
        self.spot_min_gaussian.setValidator(only_int_validator)
        self.spot_min_gaussian.setText("2")
        # Spot max sigma
        self.spot_max_gaussian = QLineEdit()
        self.spot_max_gaussian.setToolTip("The maximum standard deviation for Gaussian "
                                          "kernel. Keep this high to detect larger blobs.")
        self.spot_max_gaussian.setValidator(only_int_validator)
        self.spot_max_gaussian.setText("6")
        # Spot num sigma
        self.spot_num_sigma = QLineEdit()
        self.spot_num_sigma.setToolTip("The number of intermediate values of "
                                       "standard deviations to consider between "
                                       "`min_sigma` and `max_sigma`.")
        self.spot_num_sigma.setValidator(only_int_validator)
        self.spot_num_sigma.setText("1")
        # Spot threshold
        # self.spot_threshold = QLineEdit()
        # self.spot_threshold.setValidator(only_int_validator)
        # self.spot_threshold.setText("2")
        # Spot threshold rel
        self.spot_threshold_rel = QLineEdit()
        self.spot_threshold_rel.setToolTip("Minimum intensity of peaks, calculated as "
                                           "``max(log_space) * threshold_rel``, "
                                           "where ``log_space`` refers to the stack of "
                                           "Laplacian-of-Gaussian (LoG) images computed "
                                           "internally. This should have a value between "
                                           "0 and 1")
        self.spot_threshold_rel.setValidator(only_double_validator)
        self.spot_threshold_rel.setText("0.110")

        # self.spot_color = ColorButton(self, color='red')

        # Spot threshold rel
        self.high_pass_sigma = QLineEdit()
        self.high_pass_sigma.setValidator(only_int_validator)
        self.high_pass_sigma.setText("20")

        # Spot threshold value to define the mask for the particle watersheed segmentation
        self.threshold_value = QLineEdit()
        self.threshold_value.setToolTip("Value above which the pixel will be considered as particle")
        self.threshold_value.setValidator(only_int_validator_large)
        self.threshold_value.setText("3000")
        apply_button = QPushButton("Segment")
        apply_button.clicked.connect(self._on_click)

        # Layout
        self.box_contents.addRow("Particle Layer", self.particle_layer_cb)
        self.box_contents.addRow("Spot min gaussian", self.spot_min_gaussian)
        self.box_contents.addRow("Spot max gaussian", self.spot_max_gaussian)
        self.box_contents.addRow("Spot num sigma", self.spot_num_sigma)
        self.box_contents.addRow("Spot Threshold Rel", self.spot_threshold_rel)
        self.box_contents.addRow("High Pass sigma", self.high_pass_sigma)
        self.box_contents.addRow("Particle Threshold Value", self.threshold_value)
        self.box_contents.addWidget(apply_button)

        self.segGroupBox.setLayout(self.box_contents)
        box_layout = QVBoxLayout()
        # setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None
        box_layout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(box_layout)
        self.layout().addWidget(self.segGroupBox)

    def _on_click(self):
        process_particle_segmentation_fun(self.viewer, self.particle_layer_cb.currentData(),
                                          self.spot_min_gaussian.text(), self.spot_max_gaussian.text(),
                                          self.spot_num_sigma.text(), self.spot_threshold_rel.text(),
                                          self.high_pass_sigma.text(), self.threshold_value.text())


class SegmentationWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.box_contents = QFormLayout()
        _layers = [
            layer for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

        self.segGroupBox = QGroupBox("Cell Segmentation")
        # self.segGroupBox.setStyleSheet('QGroupBox:title{font:bold 20px}')
        '''
        self.segGroupBox.setStyleSheet('QGroupBox:title {'
                                    'subcontrol-origin: margin;'
                                    'subcontrol-position: top center;'
                                    'padding: 0 3px; }'
                                    )
        '''
        thresholds = ['Li', 'Mean', 'Minimum', 'Otsu', 'Triangle''Yen']
        self.nuclei_layer_cb = QComboBox()
        for layer in _layers:
            self.nuclei_layer_cb.addItem(layer.name)
        # self.nuclei_layer_cb.currentIndexChanged.connect(self.selectionchange)
        # Cyto layer selection
        self.cyto_layer_cb = QComboBox()
        for layer in _layers:
            self.cyto_layer_cb.addItem(layer.name)
        # Nuclei Threshold selection
        self.nuclei_threshold_cb = QComboBox()
        for threshold in thresholds:
            self.nuclei_threshold_cb.addItem(threshold)
        self.nuclei_threshold_cb.setCurrentIndex(thresholds.index('Otsu'))
        # Cyto Threshold selection
        self.cyto_threshold_cb = QComboBox()
        for threshold in thresholds:
            self.cyto_threshold_cb.addItem(threshold)
        self.cyto_threshold_cb.setCurrentIndex(thresholds.index('Mean'))
        # Nuclei gaussian sigma
        only_int_validator = QIntValidator()
        only_int_validator.setRange(0, 10)
        self.nuclei_gaussian_le = QLineEdit()
        self.nuclei_gaussian_le.setValidator(only_int_validator)
        self.nuclei_gaussian_le.setText("2")
        # Cyto gaussian sigma
        self.cyto_gaussian_le = QLineEdit()
        self.cyto_gaussian_le.setValidator(only_int_validator)
        self.cyto_gaussian_le.setText("2")
        # Min cell area
        self.min_cell_volume_le = QLineEdit()
        self.min_cell_volume_le.setValidator(only_int_validator)
        self.min_cell_volume_le.setText("10000")

        self.do_remove_cell_at_border_cb = QCheckBox()
        # self.cyto_layer_cb.currentIndexChanged.connect(self.selectionchange)
        # OK/Cancel button
        # button_box = QDialogButtonBox(QDialogButtonBox.Ok, self)
        # button_box.accepted.connect(self.accept)
        # button_box.rejected.connect(self.reject)
        apply_button = QPushButton("Segment")

        # self.apply_button.clicked.connect()

        apply_button.clicked.connect(self._on_click)

        self.box_contents.addRow("Nuclei Layer", self.nuclei_layer_cb)
        self.box_contents.addRow("Cyto Layer", self.cyto_layer_cb)
        self.box_contents.addRow("Nuclei Threshold", self.nuclei_threshold_cb)
        self.box_contents.addRow("Cyto Threshold", self.cyto_threshold_cb)
        self.box_contents.addRow("Nuclei sigma gaussian blur", self.nuclei_gaussian_le)
        self.box_contents.addRow("Cyto sigma gaussian blur", self.cyto_gaussian_le)
        self.box_contents.addRow("Min cell volume (px)", self.min_cell_volume_le)
        self.box_contents.addRow("Remove cell touching border in X/Y axes", self.do_remove_cell_at_border_cb)
        self.box_contents.addWidget(apply_button)

        self.segGroupBox.setLayout(self.box_contents)
        box_layout = QVBoxLayout()
        # setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None
        box_layout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(box_layout)
        self.layout().addWidget(self.segGroupBox)

        # self.particleSegGroupBox = QGroupBox("Particle Segmentation")
        # self.layout().addWidget(self.particleSegGroupBox)


    def _on_click(self):
        process_segmentation_fun(self.viewer, self.nuclei_layer_cb.currentText(),
                                 self.cyto_layer_cb.currentText(),
                                 self.nuclei_threshold_cb.currentText(),
                                 self.cyto_threshold_cb.currentText(),
                                 self.nuclei_gaussian_le.text(),
                                 self.cyto_gaussian_le.text(),
                                 self.min_cell_volume_le.text(),
                                 self.do_remove_cell_at_border_cb.isChecked())
        # print("napari has", len(self.viewer.layers), "layers")


'''
#@magic_factory
#def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
'''

'''
def _l_function(distances: NDArrayA, support: NDArrayA, n: int, area: float) -> tuple[NDArrayA, NDArrayA]:
    n_pairs_less_than_d = (distances < support.reshape(-1, 1)).sum(axis=1)  # type: ignore[attr-defined]
    intensity = n / area
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    l_estimate = np.sqrt(k_estimate / np.pi)
    return support, l_estimate
'''

# To add your own extra properties/features, define some functions
# e.g. Median Intensity
def median_intensity(region_mask, intensity_image):
    return np.median(intensity_image[region_mask > 0])


def spot_measurement_per_cell_measurement(viewer, _cell_label_layer_name, _particle_layer_name_list,
                                          _do_colocalization_measurement):
    cell_label_data = viewer.layers[_cell_label_layer_name].data
    number_of_cells = np.amax(cell_label_data)
    df_particle_details_list = []
    _points_spot_coordinate_df_list = []
    _df_result = pd.DataFrame()

    print('  Cell Label Name:' + _cell_label_layer_name)

    for pln in _particle_layer_name_list:
        print('    Process particle ' + pln)
        properties = ['max_intensity', 'mean_intensity', 'min_intensity']
        particle_data = viewer.layers[pln].data
        particle_label_data = viewer.layers['particle label ' + pln].data
        particle_spots_coordinates = viewer.layers['Spots ' + pln].data
        number_of_particle = np.amax(particle_label_data)
        coordinate_number = particle_spots_coordinates.shape[0]
        # msgBox = QMessageBox()
        print('    coordinate number : ' + str(coordinate_number))
        print('    particle number : ' + str(number_of_particle))
        # msgBox.setText('coordinate number : '+str(coordinate_number)+'\n '+'particle number : '+str(number_of_particle));
        # msgBox.exec();

        # TODO Do it per particle and not per spot, because sometimes the spot number is bigger than the particle number
        cell_id_colum = np.zeros((coordinate_number, 1))
        particle_id_column = np.zeros((coordinate_number, 1))
        spot_coordinates_with_cell_id = np.hstack((particle_spots_coordinates, cell_id_colum))
        spot_coordinates_with_cell_id = np.hstack((spot_coordinates_with_cell_id, particle_id_column))
        points_per_value = np.zeros((number_of_cells + 1), dtype=int)

        # skipped_coordinates = np.empty((0, 3))

        # Detect cell id by particle
        particle_cell_id_table = regionprops_table(np.asarray(particle_label_data).astype(int),
                                                   intensity_image=cell_label_data, properties=['max_intensity'])
        df_particle_cell_id = pd.DataFrame(particle_cell_id_table)
        df_particle_cell_id.rename(columns={'max_intensity': 'Cell Id'}, inplace=True)

        particle_intensity_features_table = regionprops_table(np.asarray(particle_label_data).astype(int),
                                                              intensity_image=particle_data,
                                                              properties=['area', 'max_intensity', 'mean_intensity',
                                                                          'min_intensity'],
                                                              extra_properties=(median_intensity,))
        df_particle_intensity_features = pd.DataFrame(particle_intensity_features_table)

        # Concat feature intensity per particle and cell ID per particule
        df_particle_intensity_features = pd.concat([df_particle_intensity_features, df_particle_cell_id], axis=1)

        # Measure the overlap with the other particle to check
        # For each of the other particle label, convert them to mask and measure the ratio of overlapping
        # single_df_result_particle = pd.DataFrame()

        if _do_colocalization_measurement:
            for other_pln in _particle_layer_name_list:
                if other_pln != pln:
                    other_particle_label_data = viewer.layers['particle label ' + other_pln].data
                    other_particle_binary = other_particle_label_data > 0
                    other_particle_binary_mean_table = regionprops_table(np.asarray(particle_label_data).astype(int),
                                                                         intensity_image=other_particle_binary,
                                                                         properties=['mean_intensity'])
                    single_df_result_particle = pd.DataFrame(other_particle_binary_mean_table)

                    single_df_result_particle[' overlap area with ' + other_pln] = df_particle_intensity_features[
                                                                                       'area'] * \
                                                                                   single_df_result_particle[
                                                                                       'mean_intensity']
                    single_df_result_particle[' overlap % area with ' + other_pln] = single_df_result_particle[
                                                                                         'mean_intensity'] * 100
                    single_df_result_particle.drop(columns={'mean_intensity'})

                    # Concat feature intensity per particle and cell ID per particule with the overlap area with other particle detected
                    df_particle_intensity_features = pd.concat(
                        [df_particle_intensity_features, single_df_result_particle], axis=1)

        # print(df_particle_intensity_features.to_string()) #WORK
        df_particle_details_list.append(df_particle_intensity_features)

        for i in range(coordinate_number):
            x = particle_spots_coordinates[i, 2]
            y = particle_spots_coordinates[i, 1]
            z = particle_spots_coordinates[i, 0]
            cell_id = cell_label_data[int(z), int(y), int(x)]

            # Check if the spot is also a particle, otherwise skip it

            particle_label_id = particle_label_data[int(z), int(y), int(x)]
            if particle_label_id > 0:
                spot_coordinates_with_cell_id[i, 3] = cell_id
                spot_coordinates_with_cell_id[i, 4] = particle_label_id
                points_per_value[cell_id] = points_per_value[cell_id] + 1
        '''
            else:
                #print('Skip label ' + str(i))
                skipped_coordinates = np.vstack((skipped_coordinates, particle_spots_coordinates[i]))

        viewer.add_points(skipped_coordinates,
                          ndim=3,
                          size=4,
                          face_color=viewer.layers['Spots ' + pln].face_color,
                          edge_color='white',
                          name='Spots skipped' + pln,
                          scale=viewer.layers['Spots ' + pln].scale)
        '''
        points_per_value_df = pd.DataFrame(points_per_value)
        # Remove the label 0 which is background
        points_per_value_df = points_per_value_df.iloc[1:, :]
        # Compute intensity features for this channel
        table_intensity = regionprops_table(np.asarray(cell_label_data).astype(int),
                                            intensity_image=particle_data,
                                            properties=properties, extra_properties=(median_intensity,))
        df_intensity = pd.DataFrame(table_intensity)
        # Concatenate horizontally the label/area dataframe and the intensity dataframe per cell measurement
        # if df_result.empty:
        #    df_result = df_intensity
        # else:
        _df_result = pd.concat([_df_result, df_intensity], axis=1)
        # Rename the column to add Channel information for the intensity features
        _df_result.rename(columns={
            'max_intensity': pln + ' Max Intensity',
            'min_intensity': pln + ' Min Intensity',
            'mean_intensity': pln + ' Mean Intensity',
            'median_intensity': pln + ' Median Intensity'}, inplace=True)
        # Add the number of spot
        _df_result[pln + ' # Spots'] = points_per_value_df.values

        # df_result.to_excel(directory + '/' + os.path.splitext(basename)[0] + '_result.xlsx', float_format='%.4f')
        # df_result.to_csv(directory + '/' + os.path.splitext(basename)[0] + '_result.csv', float_format='%.4f',
        #                  index=False, sep=',')
        # result_list_per_image.append(df_result)
        points_spot_coordinate_df = pd.DataFrame(spot_coordinates_with_cell_id)
        points_spot_coordinate_df.columns = ["Z", "Y", "X", "Cell ID", "Particle ID"]
        # Re-Order the columns by names to put
        points_spot_coordinate_df = points_spot_coordinate_df[['Cell ID', 'Particle ID', 'X', 'Y', 'Z']]
        _points_spot_coordinate_df_list.append(points_spot_coordinate_df)
        # Save the image as excel
        # points_details_per_value_df.to_excel(directory + '/' + os.path.splitext(basename)[0] + '_c_'+str(p['c'])+ '_result.xlsx', float_format='%.4f')
        # particle_channel_index = particle_channel_index + 1

    print('Done')
    return _df_result, df_particle_details_list, _points_spot_coordinate_df_list

def create_spot_density_map(points, radius:float = 50, image_dimension=None):

    pointcloud = vedo.pointcloud.Points(points)
    if image_dimension is None:
      image_dimension = np.max(points, axis=0).astype(int)
    vol = pointcloud.density(radius=radius,dims=image_dimension)
                             #dims=np.max(points, axis=0).astype(int))
    ndims = points.shape[1]

    # Somehow vedo returns a 3D volume with XYZ for 2D data
    if ndims == 2:
        vol = vol.tonumpy()[..., 0]
    else:
        vol = vol.tonumpy()

    return vol

def create_3d_border_mask(image_3d_array):
    array_shape = image_3d_array.shape
    bool_3d_array = np.full((array_shape[0], array_shape[1] - 2, array_shape[2] - 2), True)
    bool_3d_array_mask = np.pad(array=bool_3d_array, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant',
                                constant_values=False)
    return bool_3d_array_mask


def thresh(func):
    """
    A wrapper function to return a thresholded image.
    """

    def wrapper(im):
        return im > func(im)

    try:
        wrapper.__orifunc__ = func.__orifunc__
    except AttributeError:
        wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
    return


def segment_channel(gaussian_sigma, gaussian_mode, img, threshold_method_name, do_fill_hole):
    methods = OrderedDict({'Li': threshold_li,
                           'Mean': threshold_mean,
                           'Minimum': threshold_minimum,
                           'Otsu': threshold_otsu,
                           'Triangle': threshold_triangle,
                           'Yen': threshold_yen})
    filtered = filters.gaussian(img, sigma=gaussian_sigma, mode=gaussian_mode,
                                preserve_range=True)

    # Compute the image histogram for better performances
    nbins = 256  # Default in threshold functions
    hist = histogram(img.reshape(-1), nbins, source_range='image')
    threshold_func = methods[threshold_method_name]
    sig = inspect.signature(threshold_func)
    _kwargs = dict(hist=hist) if 'hist' in sig.parameters else {}

    # threshold_value = filters.threshold_mean(filtered)
    threshold_value = threshold_func(filtered, **_kwargs)
    thresholded = filtered > threshold_value
    if do_fill_hole:
        filled_hole = np.zeros_like(thresholded)
        z_number = thresholded.shape[0]
        for i in range(z_number):
            filled_hole[i, :, :] = scipy.ndimage.binary_fill_holes(thresholded[i, :, :])  # .astype(int)
        return filled_hole
    else:
        return thresholded


def gaussian_high_pass(image: np.ndarray, sigma: float = 2):
    """Apply a gaussian high pass filter to an image.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma (width) of the gaussian filter to be applied.
        The default value is 2.

    Returns
    -------
    high_passed_im : np.ndarray
        The image with the high pass filter applied
    """
    # low_pass = ndi.gaussian_filter(image, sigma) #Give strange result
    low_pass = filters.gaussian(image, sigma=sigma, preserve_range=True)
    high_passed_im = image - low_pass

    return high_passed_im


def seeded_watershed_segmentation(mask, seed_mask):
    # create the membrane staining from the cell mask
    distance = ndi.distance_transform_edt(mask)
    particle_labels = measure.label(seed_mask)
    return watershed(-distance, markers=particle_labels, mask=mask)


def detect_particle(img,
                    spot_min_sigma, spot_max_sigma, spot_num_sigma, spot_threshold, spot_threshold_rel, spot_spot_color,
                    high_pass_sigma, threshold_value):
    blobs_log = blob_log(img,
                         max_sigma=[spot_max_sigma, spot_max_sigma, spot_max_sigma],
                         min_sigma=[spot_min_sigma, spot_min_sigma, spot_min_sigma],
                         num_sigma=spot_num_sigma,
                         threshold=spot_threshold,
                         # detect nothing on low intensity  #threshold=.001) detect too many particle
                         threshold_rel=spot_threshold_rel)

    spot_coordinates = blobs_log[:, :3].astype(int)
    number_of_spots = np.shape(spot_coordinates)[0]
    # spot_coordinate_list.append(spot_coordinates)
    print('Number of spots:' + str(number_of_spots))

    filtered_particle_channel = gaussian_high_pass(img, high_pass_sigma)
    # filtered_channel_list.append(filtered_particle_channel)

    # threshold_value = filters.threshold_mean(filtered)
    # threshold_value = threshold_value  # threshold_func(filtered, **_kwargs)
    thresholded = filtered_particle_channel > threshold_value

    particle_seed_mask = np.zeros(shape=img.shape, dtype=np.uint16)
    for row in spot_coordinates:
        particle_seed_mask[row[0], row[1], row[2]] = 1
    particle_segmented = seeded_watershed_segmentation(thresholded, particle_seed_mask)

    #Remove all coordinate that doesn't match particle
    preserved_coordinates = np.empty((0, 3))
    skipped_coordinates = np.empty((0, 3))
    for i in range(spot_coordinates.shape[0]):
        x = spot_coordinates[i, 2]
        y = spot_coordinates[i, 1]
        z = spot_coordinates[i, 0]

        # Check if the spot is also a particle, otherwise skip it
        particle_label_id = particle_segmented[int(z), int(y), int(x)]
        if particle_label_id == 0:
            skipped_coordinates = np.vstack((preserved_coordinates, spot_coordinates[i]))
        else:
            preserved_coordinates = np.vstack((preserved_coordinates, spot_coordinates[i]))


    # particle_segmented_list.append(particle_segmented)

    particle_mask = particle_segmented > 0

    return preserved_coordinates, skipped_coordinates, particle_segmented, particle_mask


def process_segmentation_fun(viewer, nuclei_layer_name, cyto_layer_name, nuclei_threshold_method, cyto_threshold_method,
                             nuclei_gaussian, cyto_gaussian, min_cell_vol, do_remove_cell_at_border):
    gaussian_mode = "reflect"  # ["reflect", "constant", "nearest", "mirror", "wrap"]
    image_layers = [
        layer for layer in viewer.layers
        if isinstance(layer, napari.layers.Image)
    ]
    img_data = image_layers[0].data
    scale_factor = image_layers[0].scale
    # Get the image data from the selected layers
    nuclei_data = viewer.layers[nuclei_layer_name].data  # image_layers[1].data
    cyto_data = viewer.layers[cyto_layer_name].data
    # Segment the nuclei
    nuclei_mask = segment_channel(int(nuclei_gaussian),
                                  gaussian_mode, nuclei_data,
                                  nuclei_threshold_method,  # 'Otsu',
                                  True)
    # Segment the cytoplasm
    cyto_mask = segment_channel(int(nuclei_gaussian),
                                gaussian_mode,
                                cyto_data, cyto_threshold_method,  # 'Mean',
                                False)
    # Merge the mask
    cell_mask = functools.reduce(lambda m0, m1: np.where(m1 == 0, m0, m1), [cyto_mask, nuclei_mask])
    # create the membrane staining from the cell mask
    distance = ndi.distance_transform_edt(cell_mask)
    nuclei_labels = measure.label(nuclei_mask)
    predicted_instances = watershed(-distance, markers=nuclei_labels,
                                    mask=cell_mask)
    # Remove the small cells
    morphology.remove_small_objects(predicted_instances,
                                    min_size=int(min_cell_vol),
                                    in_place=True)
    # clear cell touching the border
    predicted_border_mask_3d = create_3d_border_mask(predicted_instances)
    if do_remove_cell_at_border:
        segmentation.clear_border(predicted_instances, mask=predicted_border_mask_3d, out=predicted_instances)
    predicted_instances, _, _ = segmentation.relabel_sequential(predicted_instances, offset=1)

    # Add the label to the Napari Layer
    label_layer1 = viewer.add_labels(predicted_instances, name='cell label',
                                     scale=scale_factor)


def process_particle_segmentation_fun(_viewer, particle_layer_name_list,
                                      spot_min_gaussian, spot_max_gaussian,
                                      spot_num_sigma, spot_threshold_rel,
                                      high_pass_sigma, threshold_value):
    print('Particle segmentation')
    image_layers = [
        layer for layer in _viewer.layers
        if isinstance(layer, napari.layers.Image)
    ]
    img_data = image_layers[0].data
    scale_factor = image_layers[0].scale

    for particle_layer_name in particle_layer_name_list:
        # Grab the color from the color map
        spot_color = _viewer.layers[particle_layer_name].colormap.name
        # Get the image data from the selected layers
        particle_data = _viewer.layers[particle_layer_name].data  # image_layers[1].data

        scale_factor = _viewer.layers[particle_layer_name].scale

        preserved_coordinates, skipped_coordinates, particle_segmented, particle_mask = detect_particle(particle_data,
                                                                              int(spot_min_gaussian),
                                                                              int(spot_max_gaussian),
                                                                              int(spot_num_sigma),
                                                                              None, float(spot_threshold_rel),
                                                                              spot_color,
                                                                              int(high_pass_sigma),
                                                                              int(threshold_value))

        # Add the label to the Napari Layer
        label_layer1 = _viewer.add_labels(particle_segmented, name='particle label ' + particle_layer_name,
                                         scale=scale_factor)

        '''
        # Check which spot has not been kept as a particle, du to the particle threshold, and display them as a layer
        # for investigation
        skipped_coordinates = np.empty((0, 3))
        for i in range(preserved_coordinates.shape[0]):
            x = preserved_coordinates[i, 2]
            y = preserved_coordinates[i, 1]
            z = preserved_coordinates[i, 0]

            # Check if the spot is also a particle, otherwise skip it
            particle_label_id = particle_segmented[int(z), int(y), int(x)]
            if particle_label_id == 0:
                skipped_coordinates = np.vstack((skipped_coordinates, preserved_coordinates[i]))
        '''
        _viewer.add_points(skipped_coordinates,
                          ndim=3,
                          size=4,
                          face_color=spot_color,
                          edge_color='white',
                          name='Spots skipped' + particle_layer_name,
                          scale=scale_factor)

        # Plot the spot on Napari
        # points_layer = _viewer.add_points(spot_coordinates,
        _viewer.add_points(preserved_coordinates,
                          ndim=3,
                          size=4,
                          face_color=spot_color,
                          edge_color='white',
                          name='Spots ' + particle_layer_name,
                          scale=scale_factor)


def process_analysis_cells_particles_fun(_viewer, particle_layer_name_list, cell_label_layer_name,
                                         do_colocalization_measurement, original_image_file_path,
                                         do_density_map, _spot_layer_name_list, density_radius):
    print('Process Analysis Cells/Particles')
    df_result, points_details_per_value_list, points_spot_coordinate_df_list = spot_measurement_per_cell_measurement(_viewer, cell_label_layer_name,
                                                                                     particle_layer_name_list,
                                                                                     do_colocalization_measurement)

    # result_path = 'C:\\Users\\u0094799\\Documents\\Projects\\Ghent\\Maxime_Roes\\2022_10_dataset\\2022_09_29 HeLa LSM880'
    result_path = os.path.dirname(original_image_file_path)
    filename = os.path.basename(original_image_file_path)  # Path(original_image_file_path).name
    file_name_without_extension = os.path.splitext(filename)[0]
    filename = 'Image 5 cropped'
    # df_result.to_csv(result_path + '/' + filename + '.csv', float_format='%.4f', index=False, sep='\t')
    df_result.to_excel(result_path + '/' + file_name_without_extension + '_cell_result.xlsx', float_format='%.4f')

    index = 0
    for data_frame_result in points_details_per_value_list:
        particle_name = particle_layer_name_list[index]
        print('Processed ' + particle_name)
        data_frame_result.to_excel(
            result_path + '/' + file_name_without_extension + '_particle_' + str((index + 1)) + '_result.xlsx',
            float_format='%.4f')
        index = index + 1

    index = 0
    for data_frame_spots in points_spot_coordinate_df_list:
        particle_name = particle_layer_name_list[index]
        print('Processed ' + particle_name)
        data_frame_spots.to_excel(
            result_path + '/' + file_name_without_extension + '_spots_' + str((index + 1)) + '_result.xlsx',
            float_format='%.4f')
        index = index + 1

    image_dimension = _viewer.layers[cell_label_layer_name].data.shape
    if do_density_map:
        for spot_layer_name in _spot_layer_name_list:
            #spot_data = viewer.layers[spot_layer_name].data
            image_data = create_spot_density_map(points=_viewer.layers[spot_layer_name].data,
                                                 radius=int(density_radius), image_dimension=list(image_dimension))
            density_layer = _viewer.add_image(image_data, name='Density '+spot_layer_name,
                                              scale=_viewer.layers[spot_layer_name].scale, colormap='inferno')

            '''
            import matplotlib.pyplot as plt
            import seaborn as sns
            n_steps = 50
            metric: str = "euclidean"
            N = viewer.layers[spot_layer_name].data.shape[0]
            hull = ConvexHull(viewer.layers[spot_layer_name].data)
            area = hull.volume
            max_dist = None
            if max_dist is None:
                max_dist = (area / 2) ** 0.5
            support = np.linspace(0, max_dist, n_steps)
            distances = pdist(viewer.layers[spot_layer_name].data, metric=metric)
            bins, obs_stats = _l_function(distances, support, N, area)

            figsize = [1024,1024]
            dpi=72
            _hue="coherence"
            _hue_order = None
            palette = "flare"
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig = ax.figure
            sns.lineplot(
                y="stats",
                x="bins",
                hue=_hue,
                data=obs_stats,
                hue_order=_hue_order,
                palette=palette,
                ax=ax,
            )
            '''


    print('Done')

#To debug only
if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()
