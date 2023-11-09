from iterseg.segmentation import affinity_unet_watershed
from iterseg._dock_widgets import _load_data
import napari

#p = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/230816/230816_215421_train-unet/kidneynet/231708_045139_unet_kidneynet.pt'
#od = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/230816'
#ip = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/images'
#ip = '/Users/abigailmcgovern/Data/iterseg/kidney_development/intial_test'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/image_frames'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/segmented_stacks'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/human/zarrs/201118_hTR4_DMSO_3000s_.zarr'
#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/human/zarrs/201118_hTR4_21335_600_.zarr'
#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/20201015_Ms2_DMSO_1800.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/human/segmented_timeseries'
#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/20201015_Ms2_DMSO_1800.zarr'
#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/201104_MsDT87_CMFDA10_600_exp3.zarr'
#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/200917_Ms1_DMSO_600.zarr'

# --------------
# Paths & config
# --------------2200923_Ms1_DMSO_600.zarr

ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/2200923_Ms4_Fas100_600.zarr'
od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
save_name = '230920_MxV600_is_Fas100_plateseg2c.zarr'

# -------
# Compute
# -------

v = napari.Viewer()

_load_data(v, directory=ip, data_type='individual frames', 
           layer_name='images', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0), split_channels=True)


#affinity_unet_watershed(v, v.layers['images'], save_dir=od, name='230816_kidneynet_test-data', unet_or_config_file=p,)
#affinity_unet_watershed(v, v.layers['images'], save_dir=od, name='230824_plateseg-1', debug=True)

#p = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/training/unet_training/230821_135330_train-unet/plateseg_lr0.01_e4_ne50/232108_173448_unet_plateseg_lr0.01_e4_ne50.pt'
p = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/training/unet_training/230822_122923_train-unet/plateseg_lr0.01_e4_ne50_centerness/232208_161159_unet_plateseg_lr0.01_e4_ne50_centerness.pt'
affinity_unet_watershed(v, v.layers['images [2]'], save_dir=od, name=save_name, unet_or_config_file=p)

napari.run()

# -----------
# Old options
# -----------

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/201007_Ms2_DMSO_1800.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '230831_MxV1800is_2_plateseg2c'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/20201015_Ms3_Fas100_1800.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '20201015_MxV1800is_Fas100_plateseg2c'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/200910_ms2_hir_1800.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '200910_MxV1800is_hir_plateseg2c'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/200917_Ms1_DMSO_600.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '230917_MxV600_DMSO_is_plateseg2c.zarr'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/200910_ms4_hir_600.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '230918_MxV600_hir_is_plateseg2c.zarr'

#ip = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/zarrs/2200923_Ms1_DMSO_600.zarr'
#od = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/segmented_timeseries'
#save_name = '230919_MxV600_DMSO_2_is_plateseg2c.zarr'