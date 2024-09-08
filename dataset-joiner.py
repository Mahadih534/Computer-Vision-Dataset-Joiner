import supervision as sv
import cv2
import roboflow

# download a roboflow dataset directly
# rf = roboflow(api_key="your_api_key")
# project = rf.workspace("securitysystem").project("f-s-p")
# version = project.version(1)
# ds1 = version.download("yolov8")


# import your local dataset
ds2_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_dataset/train/images",
    annotations_directory_path=f"your_dataset/train/labels",
    data_yaml_path=f"your_dataset/data.yaml",
    force_masks=False
   )

ds2_val = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_dataset/valid/images",
    annotations_directory_path=f"your_dataset/valid/labels",
    data_yaml_path=f"your_dataset/data.yaml",
    force_masks=False
   )


ds2_test = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_dataset/test/images",
    annotations_directory_path=f"your_dataset/test/labels",
    data_yaml_path=f"your_dataset/data.yaml",
    force_masks=False
   )


print(f"initial dataset status {len(ds2_train)} and {len(ds2_val)} and {len(ds2_test)} and classes {ds2_train.classes}")



ds4_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_another_dataset/train/images",
    annotations_directory_path=f"your_another_dataset/train/labels",
    data_yaml_path=f"your_another_dataset/data.yaml",
    force_masks=False
   )
ds4_val = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_another_dataset/valid/images",
    annotations_directory_path=f"your_another_dataset/valid/labels",
    data_yaml_path=f"your_another_dataset/data.yaml",
    force_masks=False
   )

ds4_test = sv.DetectionDataset.from_yolo(
    images_directory_path=f"your_another_dataset/test/images",
    annotations_directory_path=f"your_another_dataset/test/labels",
    data_yaml_path=f"your_another_dataset/data.yaml",
    force_masks=False
   )

print(f"new dataset status {len(ds4_train)} and {len(ds4_val)} and {len(ds4_test)} and classes {ds4_train.classes}")


merged_dataset_train = sv.DetectionDataset.merge([ds2_train, ds4_train])
merged_dataset_val = sv.DetectionDataset.merge([ds2_val, ds4_val])
merged_dataset_test = sv.DetectionDataset.merge([ds2_test, ds4_test])



print(f"merged_dataset merged dataset status {len(merged_dataset_train)} and {len(merged_dataset_val)} and {len(merged_dataset_test)} and classes {merged_dataset_train.classes}")



# export merged dataset

merged_dataset_train.as_yolo(images_directory_path=f"merged_dataset/train/images",
                       annotations_directory_path=f"merged_dataset/train/labels",
                       data_yaml_path=f"merged_dataset/data.yaml")

merged_dataset_val.as_yolo(images_directory_path=f"merged_dataset/valid/images",
                       annotations_directory_path=f"merged_dataset/valid/labels"
                    )

merged_dataset_test.as_yolo(images_directory_path=f"merged_dataset/test/images",
                       annotations_directory_path=f"merged_dataset/test/labels"
                       )

# If you have only a single training directory, split the directory to generate separate test and validation directories.
# ds_train, ds = merged_dataset.split(split_ratio=0.8, shuffle=True)
# ds_valid, ds_test = ds.split(split_ratio=0.5, shuffle=True)

