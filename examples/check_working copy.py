import napari
import numpy as np


viewer = napari.Viewer()
layers = viewer.open('/home/abigail/data/C2a/C2aKD/210210_IVMTR89_C2aKD_Inj3_exp3.nd2')
qtwidget, widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'U Net Predict Widget'
)

widget.predict_widget(input_volume_layer=viewer.layers[2])

napari.run()
