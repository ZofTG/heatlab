# HEATLAB

A practical image segmentation tool for python 3.11+.

<br>
<br>

## PURPOSE

<br>

This package provides a simple user interface for image segmentation over multiple frames.

The basic usage consists in generating a Qt QApplication that allows to import images or videos and to add geometrical segmenters. Then, the segmentation masks can be saved into a *".h5"* file containing:

- *masks*:

  a 4D numpy array of dtype *bool* with shape *(frames, height, width, segmenter)*. Here each *segmenter* corresponds to a specific segmentation object.

- *labels*:

  a dict containing the labels of the segmentation masks as keys and their indices along the last dimension of *masks* as value.


<br>
<br>

## INSTALLATION

*heatlab* package can be installed via *pip* trough the following code:

    pip install "heatlab @ https://github.com/ZofTG/heatlab.git"

<br>
<br>

## RUN THE APPLICATION

simpliest use of this package consists on calling the "run" method.

    import heatlab

    if __name__ == "__main__":
        heatlab.run()

This will launch a GUI application with basic commands to be used such as add video/images, include or remove segmenters, surf over the images or video frames and save the included segmenters.

## READ THE SAVED DATA

After having saved the some acquisitions, they can be retrieved via:

    segmentation_file = "segmented_data.h5"
    masks, labels = heatlab.read_segmentation_masks(segmentation_file)
