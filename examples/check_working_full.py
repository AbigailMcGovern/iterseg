import napari
import numpy as np

arr = np.random.random((3, 20, 256, 256))

viewer = napari.view_image(arr)
qtwidget, widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'U Net Predict Widget'
)

widget.predict_widget(input_volume_layer=viewer.layers[2])

napari.run()
