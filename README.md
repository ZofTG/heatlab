# HEATLAB

<br>
<br>

A practical image segmentation tool for python 3.11+.

<br>
<br>

## PURPOSE

<br>

This package provides a simple user interface for image segmentation over multiple frames.

The basic usage consists in generating a Qt QApplication that allows to import one or more images and to add geometrical segmenters. Then, the segmentation masks can be saved into a *".h5"* file containing:

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

This will launch a GUI application with basic commands to be used:



#### FROM COMMAND LINE

    cd <PYLABELLER_PATH>
    python labeller.py

#### FROM CODE

    # import
    from pylabeller import Labeller

    # show the labeller
    app = QApplication(sys.argv)
    labeller = Labeller()
    labeller.show()
    sys.exit(app.exec_())

#### WITH CUSTOM SEGMENTERS

The Labeller can be generated with a pre-made set of labellers. In any case, these can be then edited directly from the GUI.

    # import
    from pylabeller import Labeller, EllipseSegmenter, TriangleSegmenter

    # prepare the segmenter_list
    segmenter_list = [
        EllipseSegmenter(
            name="ELLIPSE",
            color=(255, 0, 0, 255),
            linewidth=2,
            fontsize=4,
        ),
        TriangleSegmenter(
            name="TRIANGLE",
            color=(0, 255, 0, 255),
            linewidth=3,
            fontsize=2,
        )
    ]

    # show the labeller
    app = QApplication(sys.argv)
    labeller = Labeller(segmenters=segmenter_list)
    labeller.show()
    sys.exit(app.exec_())

#### WITH CUSTOM DATA READERS

This section describes how to provide an arbitrary amount of custom functions that will allow to import custom data.

This example shows how to use the *formats* argument of the *Labeller* class to tell the object how to import images stored on *.h5* files.

    # import
    from pylabeller import Labeller
    import h5py

    # prepare the importing function
    def from_h5(file:str) -> NDArray:
        with h5py.File(file, "r") as obj:
            images = obj["images"][:]
        return images

    extra_formats = {'h5': from_h5}

    # show the labeller
    app = QApplication(sys.argv)
    labeller = Labeller(**extra_formats)
    labeller.show()
    sys.exit(app.exec_())

<br>

### SETUP THE IMAGE(s) TO BE SEGMENTED

#TODO

<br>

### EDIT A SEGMENTER (OPTIONAL)

All options are available by moving with the mouse over one segmenter. The following section describes how to use the GUI to edit the properties of one specific segmenter.

#### *RENAME*

#TODO

#### *CHANGE COLOR*

#TODO

#### *CHANGE FONTSIZE*

#TODO

#### *CHANGE LINEWIDTH*

#TODO

#### *SHOW/HIDE TEXT*

#TODO

#### *DELETE*

#TODO

<br>

### ADD A NEW THE SEGMENTER (OPTIONAL)

#TODO

<br>

### SEGMENTATION

This section describes how to use the segmenters on each frame.

#### *SEGMENT AN AREA*

#TODO

#### *ROTATE*

#TODO

#### *REMOVE A SEGMENTER AREA*

#TODO

#### *MOVE OVER MULTIPLE FRAMES*

#TODO

<br>

### SAVE THE SEGMENTED DATA

#TODO

<br>

<br>

### SETUP CUSTOM DATA READER

<br>




## INSTALLATION
