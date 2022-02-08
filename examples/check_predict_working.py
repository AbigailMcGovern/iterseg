import napari
import numpy as np
import zarr

arr = zarr.open('/Users/amcg0011/GitRepos/iterseg/src/iterseg/data/example.zarr')

viewer = napari.view_image(arr)
qtwidget, widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'U Net Predict Widget'
)

widget.predict_widget()

napari.run()
