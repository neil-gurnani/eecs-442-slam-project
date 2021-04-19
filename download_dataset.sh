#!/bin/bash

urls=(
	"https://www.eth3d.net/data/slam/datasets/kidnap_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/large_loop_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_4_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_5_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_7_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_face_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_face_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_face_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/mannequin_head_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/planar_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/planar_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_4_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_5_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_scene_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_scene_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/plant_scene_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/reflective_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/repetitive_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sfm_bench_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sfm_garden_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sfm_house_loop_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sofa_shake_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/table_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/table_4_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/table_7_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sfm_lab_room_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sfm_lab_room_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sofa_1_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sofa_2_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sofa_3_mono.zip"
	"https://www.eth3d.net/data/slam/datasets/sofa_4_mono.zip"
)

# Create the data directory if it doesn't already exist
# It should already exist, because 
if [ ! -d "./data" ]
then
	mkdir data
fi

cd ./data

pwd

for url in ${urls[@]}; do
	wget $url
done

unzip -q "*.zip"
rm *.zip