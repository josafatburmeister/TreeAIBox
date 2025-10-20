import os
from pathlib import Path
import sys
from typing import Any, Dict, Literal, Optional

from fire import Fire
import numpy as np
import numpy.typing as npt
from pointtorch import read, PointCloud


def checkModelExistence(model_path: str, model_name: str) -> bool:
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_name}")
        return False
    return True


def filter_point_cloud(
    point_cloud: PointCloud,
    src_folder: str,
    model_name: str,
    model_folder: str,
    use_gpu: bool,
    if_bottom_only: bool = False,
    subfolder: str = "filter",
) -> Optional[npt.NDArray]:
    """
    Apply component filtering (semantic segmentation) to the selected point cloud using 3D deep learning.

    Args:
        point_cloud: Point cloud to filter.
        src_folder: Path of the folder containing the TreeAIBox source code.
        model_name: Name of the deep learning model to apply.
        model_folder: Path of the folder containing the model checkpoints.
        use_gpu: Whether a GPU is available and should be used.
        if_bottom_only: Whether to only consider the xy-coordinates for point cloud tiling. Defaults to :code:`False`.
        subfolder: Name of the subfolder in the source folder containing the model configuration files.

    Returns:
        Semantic segmentation class indices if processing.
    """

    sys.path.insert(0, src_folder)

    from modules.filter.componentFilter import filterPoints

    sys.path.remove(src_folder)

    print(f"Apply component filtering called with use_gpu={use_gpu}")
    print(f"Currently selected model: {model_name}")

    config_file = os.path.join(src_folder, f"modules/{subfolder}/{model_name}.json")
    model_path = os.path.join(model_folder, f"{model_name}.pth")

    if not checkModelExistence(model_path, model_name):
        raise ValueError(f"Model {model_name} does not exist.")

    pcd = point_cloud.xyz()

    return filterPoints(
        config_file,
        pcd,
        model_path,
        if_bottom_only=if_bottom_only,
        use_efficient="esegformer" in model_name,
        use_cuda=use_gpu,
    )


def apply_noise_clean(
    point_cloud: PointCloud, src_folder: str, max_gap: float = 3.0, min_pts: int = 100
) -> npt.NDArray:
    """
    Performs connected component analysis and removes small isolated clusters. If the point cloud contains a column
    named "stemcls", the filtering is applied to the stem points only.

    Args:
        point_cloud: Point cloud to process.
        src_folder: Path of the folder containing the TreeAIBox source code.
        max_gap: Maximum gap between points to assign them to the same component.
        min_pts: Minimum number of points per component.

    Returns:
        Filtered labels.
    """
    sys.path.insert(0, src_folder)

    from modules.treeisonet.cleanSmallerClusters import applySmallClusterClean

    sys.path.remove(src_folder)

    pcd0 = point_cloud.xyz()
    pcd = pcd0[:, :3] - np.min(pcd0[:, :3], 0)

    if "stemcls" in point_cloud:
        stemcls = point_cloud["stemcls"].to_numpy()
        stemind = stemcls.astype(np.int32) > 1
        pcd_stem = pcd[stemind]
        conn_labels = applySmallClusterClean(pcd_stem, max_gap, min_pts)
        removal_ind = conn_labels == 0
        conn_labels[removal_ind] = 1.0
        conn_labels[~removal_ind] = 2.0
        conn_cls = np.ones(len(point_cloud))
        conn_cls[stemind] = conn_labels

        return conn_cls

    return applySmallClusterClean(pcd, max_gap, min_pts)


def get_tree_locations(
    point_cloud: PointCloud,
    src_folder: str,
    model_name: str,
    model_folder: str,
    use_gpu: bool,
    if_stem: bool,
    cutoff_thresh: float,
    conf_thresh: float,
    min_rad: float,
    max_gap: float,
    nms_thresh: float,
    custom_voxel_res_xy: float,
    custom_voxel_res_z: bool,
) -> bool:
    """
    Predict stem locations using deep learning.

    Args:
        point_cloud: Point cloud to process.
        src_folder: Path of the folder containing the TreeAIBox source code.
        model_name: Name of the deep learning model to apply.
        model_folder: Path of the folder containing the model checkpoints.
        use_gpu: Whether a GPU is available and should be used.
        if_stem: Whether tree stems are visible in the point cloud and should be used for tree location detection.
            Should be set to :code:`True` for dense TLS and UAV point clouds and to :code:`False` for sparse ALS point
            clouds.
        cutoff_thresh: Threshold for filtering the input points based on point height. The minimum and maximum
            z-coordinate of the input points are computed. Only points whose z-coordinate is below (z_max - z_min) *
            threshold are used as input. Only used :code:`if_stem` is :code:`True`.
        conf_thresh: The minimum predicted probability of a point being a tree location required for that point to be
            considered for tree location retrieval. Only used if :code:`if_stem` is :code:`False`. For the case that
            :code:`if_stem` is :code:`True`, the threshold is hard-coded to 0.1.
        min_rad: Minimum tree crown radius. Only used if code:`if_stem` is :code:`False`.
        max_gap: If :code:`if_stem` is :code:`False`, the points whose predicted probability is above
            :code:`conf_thresh`, are clustered using a connected component analysis. This parameter defines the maximum
            distance between points in the same cluster. Only used if code:`if_stem` is :code:`False`.
        nms_thresh: If :code:`if_stem` is :code:`False`, a non-maximum suppression is applied to the predicted tree
            locations based on the predicted crown radii. This parameter defines the overlap threshold for this
            non-maximum suppression. Only used if code:`if_stem` is :code:`False`.
        custom_voxel_res_xy: Custom voxel resolution in the xy-dimensions.
        custom_voxel_res_z: Custom voxel resolution in the z-dimensions.

    Returns:
        Tuple of three elements:
            - | A point cloud representing the tree stem locations (xyz if :code:`if_stem` is :code:`True` and xyz +
              | crown radius otherwise).
            - | An array containing the predicted probabilities for each input point of being a tree location. If
              | :code:`if_stem` is :code:`True`, :code:`None` is returned instead.
            - | An array containing the predicted crown radius for each input point. If :code:`if_stem` is :code:`True`,
              | :code:`None` is returned instead.
    """

    sys.path.insert(0, src_folder)

    from modules.treeisonet.treeLoc import treeLoc
    from modules.treeisonet.treeLoc import postPeakExtraction

    sys.path.remove(src_folder)

    print(f"Apply TreeLoc extraction with use_gpu={use_gpu}")
    print(f"Selected model: {model_name}")

    config_file = os.path.join(src_folder, f"modules/treeisonet/{model_name}.json")
    model_path = os.path.join(model_folder, f"{model_name}.pth")

    if not checkModelExistence(model_path, model_name):
        raise ValueError(f"Model {model_name} does not exist.")

    xyz = point_cloud.xyz()

    if "treefilter" in point_cloud:
        treefilter = point_cloud["treefilter"].to_numpy().astype(np.int32)
        treefilter_ind = treefilter > 1.0
        xyz_abg = xyz[treefilter_ind]
    else:
        xyz_abg = xyz
        treefilter_ind = None

    if if_stem:
        if "stemcls" not in point_cloud:
            raise ValueError(
                "When if_stem is set to True, the input point cloud must contain a column named 'stemcls'."
            )
        stemcls = point_cloud["stemcls"].to_numpy().astype(np.int32)
        if "treefilter" in point_cloud:
            stemcls = stemcls[treefilter_ind]
        xyz_abg = xyz_abg[stemcls > 1]

    # if if_stem is True, the treeLoc method returns the readily-processed 3D tree locations
    # otherwise, it returns five values for each input point, namely the xyz-coordinates the probability of a tree top
    # being at this location and the predicted crown radius for that location
    preds = treeLoc(
        config_file,
        xyz_abg,
        model_path,
        use_cuda=use_gpu,
        if_stem=if_stem,
        cutoff_thresh=cutoff_thresh,
        custom_resolution=np.array([custom_voxel_res_xy, custom_voxel_res_xy, custom_voxel_res_z]),
    )

    n_pts = len(point_cloud)

    if if_stem:
        pred_treeloc_tops = preds
        treeloc_radius = None
        treeloc_conf = None
    else:
        if treefilter_ind is not None:
            pred_treeloc_conf_rads = np.zeros([n_pts, preds.shape[1]], dtype=np.float32)
            pred_treeloc_conf_rads[treefilter_ind, :] = preds
        else:
            pred_treeloc_conf_rads = preds

        # Post-processing to extract tree locations
        pred_treeloc_tops = postPeakExtraction(
            pred_treeloc_conf_rads[pred_treeloc_conf_rads[:, -2] > conf_thresh],
            K=5,
            max_gap=max_gap,
            min_rad=min_rad,
            nms_thresh=nms_thresh,
        )

        if pred_treeloc_conf_rads is not None:
            treeloc_radius = pred_treeloc_conf_rads[:, -1]
            treeloc_conf = pred_treeloc_conf_rads[:, -2]

    # Create location point cloud
    loc_pcd = PointCloud(pred_treeloc_tops, columns=["x", "y", "z"] if if_stem else ["x", "y", "z", "crown_radius"])
    print(f"Number of tree locations extracted: {str(len(pred_treeloc_tops))}")

    return loc_pcd, treeloc_conf, treeloc_radius


def get_tree_offsets(
    point_cloud: PointCloud,
    tree_locations: PointCloud,
    src_folder: str,
    model_name: str,
    model_folder: str,
    use_gpu: bool,
    custom_voxel_res_xy: float,
    custom_voxel_res_z: float,
):
    """
    Assign tree points to individual trees by predicting offsets towards the tree locations for each point and assigning
    each offset point to the closest tree location.

    Args:
        point_cloud: Point cloud to process.
        tree_locations: Point cloud representing the predicted tree locations.
        src_folder: Path of the folder containing the TreeAIBox source code.
        model_name: Name of the deep learning model to apply.
        model_folder: Path of the folder containing the model checkpoints.
        use_gpu: Whether a GPU is available and should be used.
        custom_voxel_res_xy: Custom voxel resolution in the xy-dimensions.
        custom_voxel_res_z: Custom voxel resolution in the z-dimensions.

    Returns:
        Tree instance labels for each point.
    """

    sys.path.insert(0, src_folder)

    from modules.treeisonet.treeOff import treeOff

    sys.path.remove(src_folder)

    print(f"Apply TreeOff segmentation with use_gpu={use_gpu}")
    print(f"Currently selected model: {model_name}")

    config_file = os.path.join(src_folder, f"modules/treeisonet/{model_name}.json")
    model_path = os.path.join(model_folder, f"{model_name}.pth")

    if not checkModelExistence(model_path, model_name):
        raise ValueError(f"Model {model_name} does not exist.")

    pcd = point_cloud.xyz()

    treeloc = tree_locations.xyz()

    if "treefilter" in point_cloud:
        treefilter = point_cloud["treefilter"].to_numpy().astype(np.int32)
        treefilter_ind = treefilter > 1.0
        pcd_abg = pcd[treefilter_ind]
    else:
        pcd_abg = pcd
        treefilter_ind = None

    tree_instance_ids = treeOff(
        config_file,
        pcd_abg,
        treeloc,
        model_path,
        use_cuda=use_gpu,
        custom_resolution=np.array([custom_voxel_res_xy, custom_voxel_res_xy, custom_voxel_res_z]),
    )

    pred_treeitc = np.zeros(len(point_cloud), dtype=np.int32)
    if treefilter_ind is not None:
        pred_treeitc[treefilter_ind] = tree_instance_ids
    else:
        pred_treeitc = tree_instance_ids

    return pred_treeitc


def get_stem_clusters_shortest_path(
    point_cloud: PointCloud, stem_locations: PointCloud, src_folder: str, resolution: float = 0.06, max_gap: float = 0.3
) -> npt.NDArray:
    """
    Retrieves stem clusters of a point cloud based on connected component analysis and shortest-path analysis.

    Args:
        point_cloud: Point cloud to process.
        stem_locations: Stem locations.
        src_folder: Path of the folder containing the TreeAIBox source code.
        resolution: Voxel size for point cloud downsampling. Defaults to 0.06 m.
        max_gap: Maximum distance for connected component analysis.

    Returns:
        Stem cluster labels.
    """

    sys.path.insert(0, src_folder)

    from modules.treeisonet.stemCluster import shortestpath3D

    sys.path.remove(src_folder)

    print("Apply stemClusterSP")

    pcd = point_cloud.xyz()

    if "stemcls" not in point_cloud:
        raise ValueError("Input point cloud must contain a column named 'stemcls'.")

    stemcls = point_cloud["stemcls"].to_numpy().astype(np.int32)

    stembase = stem_locations.xyz()
    if len(stembase) == 0:
        raise ValueError("List of stem locations is empty.")

    stem_cluster_labels = shortestpath3D(pcd, stemcls, stembase, min_res=resolution, max_isolated_distance=max_gap)

    return stem_cluster_labels


def get_crown_clusters(
    point_cloud: PointCloud, src_folder: str, model_name: str, model_folder: str, use_gpu: bool
) -> npt.NDArray:
    """
    Clusters tree crown points into individual instances based on offset vectors predicted using 3D deep learning.

    Args:
        point_cloud: Point cloud to process.
        src_folder: Path of the folder containing the TreeAIBox source code.
        model_name: Name of the deep learning model to apply.
        model_folder: Path of the folder containing the model checkpoints.
        use_gpu: Whether a GPU is available and should be used.

    Returns:
        Tree instance labels for each point.
    """

    sys.path.insert(0, src_folder)

    from modules.treeisonet.crownOff import crownOff

    sys.path.remove(src_folder)

    print(f"Apply crownoff with use_gpu={use_gpu}")
    print(f"Currently selected model: {model_name}")

    config_file = os.path.join(src_folder, f"modules/treeisonet/{model_name}.json")
    model_path = os.path.join(model_folder, f"{model_name}.pth")

    if not checkModelExistence(model_path, model_name):
        raise ValueError(f"Model {model_name} does not exist.")

    pcd = point_cloud.xyz()

    if "stemoff" not in point_cloud:
        raise ValueError("Input point cloud must contain a column named 'stemoff'.")

    stem_id = point_cloud["stemoff"].to_numpy().astype(np.int32)

    if "treefilter" in point_cloud:
        treefilter = point_cloud["treefilter"].to_numpy().astype(np.int32)
        treefilter_ind = treefilter > 1.0
        pcd_abg = pcd[treefilter_ind]
        stem_id = stem_id[treefilter_ind]
    else:
        pcd_abg = pcd
        treefilter_ind = None

    preds = crownOff(
        config_file,
        pcd_abg,
        stem_id,
        model_path,
        use_cuda=use_gpu,
    )
    pred_treeitc = np.zeros(len(point_cloud), dtype=np.int32)
    if treefilter_ind is not None:
        pred_treeitc[treefilter_ind] = preds
    else:
        pred_treeitc = preds

    return pred_treeitc


def select_model(
    model_dict: Dict[str, Any], task: str, scanner_type: str, scene_type: str, resolution: Optional[int] = None
):
    """
    Selects a model for the given task based on the parameters.

    Args:
        model_dict: Dictionary defining the available model checkpoints.
        task: Name of the task.
        scanner_type: Scanner type that was used to collect the input point cloud: :code:`"tls"` | :code:`"uav"` |
            :code:`"als"`.
        scene_type: Scene type of the input point cloud: :code:`"boreal"` | :code:`"mixedwood"` | :code:`"reclamation"`.
        resolution: Input resolution (voxel size) of the model. Defaults to :code:`None`, which means that the highest
            available resolution is used.
    """

    if task == "stem_classification":
        available_models = model_dict[task].get(scanner_type, {}).get(scene_type, [])
    else:
        available_models = model_dict[task].get(scanner_type, {})
    selected_model = None
    max_resolution = float("inf")

    if resolution is not None:
        for model in available_models:
            if f"_{resolution}cm_" in model:
                selected_model = model
                break
    else:
        for model in available_models:
            resolution = int(model.split("_")[-2].removesuffix("cm"))
            if resolution < max_resolution:
                selected_model = model
                max_resolution = resolution

    return selected_model


def segment_point_cloud(
    point_cloud_path: str,
    output_folder: str,
    src_folder: str,
    model_folder: str,
    scanner_type: Literal["tls", "uav", "als"],
    scene_type: Literal["boreal", "mixedwood", "reclamation"],
    use_gpu: bool = True,
    clean_stem_noise: bool = False,
):
    """
    Applies the treeisoNet pipeline to a given point cloud.

    Args:
        point_cloud_path: Path of the point cloud to process.
        output_folder: Path of the folder in which to store the results.
        src_folder: Path of the folder containing the TreeAIBox source code.
        model_folder: Path of the folder containing the model checkpoints.
        scanner_type: Scanner type that was used to collect the input point cloud: :code:`"tls"` | :code:`"uav"` |
            :code:`"als"`.
        scene_type: Scene type of the input point cloud: :code:`"boreal"` | :code:`"mixedwood"` | :code:`"reclamation"`.
        use_gpu: Whether a GPU is available and should be used. Defaults to :code:`True`.
        clean_stem_noise: Whether small stem clusters should be filtered out using connected component analysis.
            Defaults to :code:`False`.
    """

    if (
        scanner_type == "tls"
        and scene_type != "boreal"
        or scanner_type == "uav"
        and scene_type != "mixedwood"
        or scanner_type == "als"
        and scene_type != "reclamation"
    ):
        raise ValueError(f"Scene type {scene_type} not supported for {scanner_type} data.")

    point_cloud = read(point_cloud_path)

    models = {
        "tree_filtering": {
            "tls": ["treefiltering_tls_esegformer3D_128_8cm(GPU3GB)"],
            "uav": [
                "treefiltering_uav_esegformer3D_128_12cm(GPU3GB)",
                "treefiltering_uav_esegformer3D_128_15cm(GPU3GB)",
            ],
            "als": [
                "treefiltering_als_esegformer3D_128_15cm(GPU3GB)",
                "treefiltering_als_esegformer3D_128_50cm(GPU3GB)",
                "treefiltering_als_esegformer3D_128_80cm(GPU3GB)",
            ],
        },
        "stem_classification": {
            "tls": {
                "boreal": [
                    "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
                    "treeisonet_tls_boreal_stemcls_esegformer3D_128_8cm(GPU4GB)"
                    "treeisonet_tls_boreal_stemcls_esegformer3D_128_10cm(GPU3GB)",
                ]
            },
            "uav": {
                "mixedwood": [
                    "treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)",
                ]
            },
        },
        "tree_location": {
            "tls": ["treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)"],
            "uav": ["treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)"],
            "als": ["treeisonet_als_reclamation_treeloc_esegformer3D_128_10cm(GPU4GB)"],
        },
        "tree_offset": {"als": ["treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)"]},
        "crown_offset": {
            "tls": ["treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)"],
            "uav": ["treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)"],
        },
    }

    tree_filtering_model = select_model(models, "tree_filtering", scanner_type, scene_type)

    tree_filtering = filter_point_cloud(
        point_cloud,
        src_folder,
        tree_filtering_model,
        model_folder,
        use_gpu=use_gpu,
        if_bottom_only=scanner_type != "als",
        subfolder="filter",
    )
    point_cloud["treefilter"] = tree_filtering

    if scanner_type != "als":
        stem_classification_model = select_model(models, "stem_classification", scanner_type, scene_type)

        stem_classification = filter_point_cloud(
            point_cloud, src_folder, stem_classification_model, model_folder, use_gpu=use_gpu, subfolder="treeisonet"
        )
        point_cloud["stemcls"] = stem_classification

        if clean_stem_noise:
            point_cloud["stemcls"] = apply_noise_clean(point_cloud, src_folder)

    tree_location_model = select_model(models, "tree_location", scanner_type, scene_type)

    if_stem = scanner_type in ["tls", "uav"]
    tree_loc_cutoff_thresh = 0.3
    tree_loc_conf_thresh = 0.3
    tree_loc_min_rad = 0.2
    tree_loc_max_gap = 0.3
    tree_loc_nms_thresh = 0.5
    custom_voxel_res_xy = 0
    custom_voxel_res_z = 0

    tree_locations, tree_location_conf, treeloc_radius = get_tree_locations(
        point_cloud,
        src_folder,
        tree_location_model,
        model_folder,
        use_gpu,
        if_stem,
        tree_loc_cutoff_thresh,
        tree_loc_conf_thresh,
        tree_loc_min_rad,
        tree_loc_max_gap,
        tree_loc_nms_thresh,
        custom_voxel_res_xy,
        custom_voxel_res_z,
    )

    if scanner_type == "als":
        point_cloud["treeloc_conf"] = tree_location_conf
        point_cloud["treeloc_radius"] = treeloc_radius
        tree_offset_model = select_model(models, "tree_offset", scanner_type, scene_type)
        point_cloud["itc"] = get_tree_offsets(
            point_cloud,
            tree_locations,
            src_folder,
            tree_offset_model,
            model_folder,
            use_gpu,
            custom_voxel_res_xy,
            custom_voxel_res_z,
        )
    else:
        stemoff_resolution = 0.06
        stemoff_max_gap = 0.3
        point_cloud["stemoff"] = get_stem_clusters_shortest_path(
            point_cloud, tree_locations, src_folder, resolution=stemoff_resolution, max_gap=stemoff_max_gap
        )

        crown_offset_model = select_model(models, "crown_offset", scanner_type, scene_type)

        point_cloud["itc"] = get_crown_clusters(point_cloud, src_folder, crown_offset_model, model_folder, use_gpu)

    file_name = Path(point_cloud_path).name
    file_stem = Path(point_cloud_path).stem
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True, parents=True)
    point_cloud.to(output_folder_path / file_name)

    tree_locations.to(output_folder_path / f"{file_stem}_tree_locations.laz")


if __name__ == "__main__":
    Fire(
        {
            "segment_point_cloud": segment_point_cloud,
        }
    )
