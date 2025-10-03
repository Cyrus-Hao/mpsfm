from collections import defaultdict

import numpy as np
import pycolmap

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.estimators import AbsolutePose, RelativePose
from mpsfm.utils.geometry import calculate_triangulation_angle, has_point_positive_depth


class MpsfmRegistration(BaseClass):
    """MP-SfM Registration class. This class is used to register images and triangulate points."""

    default_conf = {
        "lifted_registration": True,  # important for ablation but can be removed
        "absolute_pose": {},
        "relative_pose": {},
        "reduce_min_inliers_at_failure": 6,  # release
        # dev
        "parallax_thresh": 1.5,  # exploration
        "combined_triangle_thresh": 1.5,
        "robust_triangles": 1,
        "resample_bunlde": False,  # exploration,
        "colmap_options": "<--->",
        "verbose": 0,
        # 先验外参配置
        "use_prior_poses": False,  # 是否使用先验外参
        "pose_config_path": None,  # 外参配置文件路径
    }

    def _init(self, mpsfm_rec, correspondences, triangulator, **kwargs):
        self.mpsfm_rec = mpsfm_rec
        self.correspondences = correspondences
        self.triangulator = triangulator
        self.relative_pose_estimator = RelativePose(self.conf.relative_pose)
        self.absolute_pose_estimator = AbsolutePose(self.conf.absolute_pose)

        self.half_ap_min_inliers = 0
        self.registration_cache = defaultdict(dict)
        
        # 初始化先验外参
        self.prior_poses = {}
        if self.conf.use_prior_poses and self.conf.pose_config_path:
            self._load_prior_poses()
    
    def _load_prior_poses(self):
        """加载先验相机外参."""
        try:
            import yaml
            from pathlib import Path
            
            pose_config_path = Path(self.conf.pose_config_path)
            if not pose_config_path.exists():
                print(f"Warning: Pose config file not found: {pose_config_path}")
                return
            
            with open(pose_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            camera_poses = config.get('camera_poses', {})
            
            # 解析外参矩阵
            for image_name, pose_data in camera_poses.items():
                transform_matrix = np.array(pose_data['transform_matrix'])
                
                # 转换为pycolmap格式
                # 提取左上角3x3子矩阵（旋转矩阵）
                rotation_matrix = transform_matrix[:3, :3]
                # 提取第4列的前3个元素（平移向量）
                translation = transform_matrix[:3, 3]
                
                # 确保旋转矩阵是正交的
                U, _, Vt = np.linalg.svd(rotation_matrix)
                rotation_matrix = U @ Vt
                
                # 创建pycolmap Rigid3d对象
                # 直接从旋转矩阵创建Rotation3d
                rotation = pycolmap.Rotation3d(rotation_matrix)
                rigid_pose = pycolmap.Rigid3d(rotation, translation)
                
                self.prior_poses[image_name] = rigid_pose
            
            print(f"Loaded {len(self.prior_poses)} prior poses from {pose_config_path}")
                
        except Exception as e:
            print(f"Warning: Failed to load prior poses: {e}")
    
    def has_prior_pose(self, image_name: str) -> bool:
        """检查图像是否有先验外参."""
        return image_name in self.prior_poses
    
    def get_prior_pose(self, image_name: str):
        """获取先验外参."""
        return self.prior_poses.get(image_name, None)

    @staticmethod
    def _candidate_points3D_for_init(
        cam_from_world1, cam_from_world2, matches, image1, image2, camera1, camera2, inliers=None
    ):
        candidate_points = defaultdict(list)
        if inliers is None:
            inliers = slice(None)
        for match in matches[inliers]:
            pt2d_id_1, pt2d_id_2 = match
            kp1_i = image1.points2D[pt2d_id_1].xy
            kp2_i = image2.points2D[pt2d_id_2].xy
            pointdata = np.array([kp1_i, kp2_i], dtype=np.float64)
            out = pycolmap.estimate_triangulation(pointdata, [cam_from_world1, cam_from_world2], [camera1, camera2])
            if out is None:
                continue

            projection_center1 = cam_from_world1.rotation.inverse() * -cam_from_world1.translation
            projection_center2 = cam_from_world2.rotation.inverse() * -cam_from_world2.translation
            tri_angle = calculate_triangulation_angle(projection_center1, projection_center2, out["xyz"])
            posdepth1 = has_point_positive_depth(cam_from_world1.matrix(), out["xyz"])
            posdepth2 = has_point_positive_depth(cam_from_world2.matrix(), out["xyz"])

            candidate_points["pt2d_id_1"].append(pt2d_id_1)
            candidate_points["pt2d_id_2"].append(pt2d_id_2)
            candidate_points["tri_angle"].append(np.rad2deg(tri_angle))
            candidate_points["posdepth1"].append(posdepth1)
            candidate_points["posdepth2"].append(posdepth2)
            candidate_points["xyz"].append(out["xyz"])
        return candidate_points

    def _find_2D3D_pairs(self, im_ref_id, imid, image_ref, image, pair2D3D):
        corr = self.correspondences.matches(im_ref_id, imid)
        if im_ref_id in self.mpsfm_rec.images[image.imid].ignore_matches_AP:
            keep_matches = ~self.mpsfm_rec.images[image.imid].ignore_matches_AP[im_ref_id]
            corr = corr[keep_matches]
        if len(corr) == 0:
            pair2D3D["2d"] = np.zeros((0, 2))
            pair2D3D["3d"] = np.zeros((0, 3))
            pair2D3D["lifted"] = np.zeros(0, dtype=bool)
            return

        pts2d_ids_ref, pts2d_ids_qry = corr.T

        use_3d = np.array([image_ref.points2D[pt].has_point3D() for pt in pts2d_ids_ref])
        point3D_ids = np.array(
            [image_ref.points2D[pt].point3D_id for pt, has_3d in zip(pts2d_ids_ref, use_3d) if has_3d]
        )
        self._collect_pairs(
            im_ref_id,
            image_ref,
            image,
            pts2d_ids_ref,
            pts2d_ids_qry,
            use_3d,
            point3D_ids,
            pair2D3D,
        )

    def register_and_triangulate_init_pair(self, imid1, imid2):
        """Register initial image pair and triangulate it's points."""
        matches = self.correspondences.matches(imid1, imid2)
        kps1 = self.mpsfm_rec.keypoints(imid1)
        kps2 = self.mpsfm_rec.keypoints(imid2)
        camera1 = self.mpsfm_rec.camera(imid1)
        camera2 = self.mpsfm_rec.camera(imid2)

        # 先验优先：T2 的优先级为 prior2 > AP > E
        name1 = self.mpsfm_rec.images[imid1].name
        name2 = self.mpsfm_rec.images[imid2].name
        prior1 = self.get_prior_pose(name1)
        prior2 = self.get_prior_pose(name2)

        T_c1w = prior1 if prior1 is not None else pycolmap.Rigid3d()

        matches_used = None
        T_c2w = None
        rescale = 1
        unproj_cam_cached = None
        valid_lifted_cached = None

        if prior2 is not None:
            # 完全跳过 AP 和 E，直接使用先验
            T_c2w = prior2
            # 用先验位姿直接三角化，估计尺度用于lifted重标定（可选，存在三角化成功时）
            pts_tri_prior = self._candidate_points3D_for_init(
                T_c1w, T_c2w, matches, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
            )
            if len(pts_tri_prior["xyz"]) > 0:
                tri_world = np.vstack(pts_tri_prior["xyz"])  # (N,3) 世界
                tri_cam1 = T_c1w * tri_world
                z = tri_cam1[:, -1]
                mask = np.array(list(pts_tri_prior["pt2d_id_1"]))
                d = self.mpsfm_rec.images[imid1].depth.data_prior_at_kps(kps1[mask])
                rescale = np.median(z / d)
            matches_used = matches
        else:
            # 先尝试 AP（若有 prior1 则在其世界系下，否则 cam1 世界系为单位）
            unproj_cam0, valid_lifted0 = self._lift_points_for_init(imid1, kps1, camera1)
            valid_matches0 = matches[valid_lifted0[matches[:, 0]]]
            unproj_world0 = T_c1w.inverse() * unproj_cam0
            AP_info = self.absolute_pose_estimator(
                kps2[valid_matches0[:, 1]], unproj_world0[valid_matches0[:, 0]], camera2
            )
            ap_min_num_inliers = self.conf.colmap_options.abs_pose_min_num_inliers
            ap_sufficient = (AP_info is not None) and (AP_info["num_inliers"] >= ap_min_num_inliers)

            if ap_sufficient:
                # AP 充分：进行高/低视差判定（需一次 E 以获得三角角）
                E_info = self.relative_pose_estimator(
                    kps1[matches[:, 0]], kps2[matches[:, 1]], camera1, camera2
                )
                inlier_matches_e = matches[E_info["inlier_mask"]]
                T_c2w_e = E_info["cam2_from_cam1"] * T_c1w
                pts_tri_e = self._candidate_points3D_for_init(
                    T_c1w, T_c2w_e, inlier_matches_e, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
                )
                triangles = np.array(pts_tri_e["tri_angle"]) if len(pts_tri_e["tri_angle"]) > 0 else np.array([])
                high_parallax = (triangles > self.conf.parallax_thresh).sum() > AP_info["num_inliers"]

                if high_parallax:
                    # 高视差：采用 E 位姿与 E 内点，并用 E 三角化估计尺度
                    T_c2w = T_c2w_e
                    matches_used = inlier_matches_e
                    if len(pts_tri_e["xyz"]) > 0:
                        tri_world = np.vstack(pts_tri_e["xyz"])  # (N,3)
                        tri_cam1 = T_c1w * tri_world
                        z = tri_cam1[:, -1]
                        mask = np.array(list(pts_tri_e["pt2d_id_1"]))
                        d = self.mpsfm_rec.images[imid1].depth.data_prior_at_kps(kps1[mask])
                        rescale = np.median(z / d)
                else:
                    # 低视差：采用 AP 位姿与 AP 内点，并复用此前 lift
                    T_c2w = AP_info["cam_from_world"]
                    matches_used = valid_matches0[AP_info["inlier_mask"]]
                    rescale = 1
                    unproj_cam_cached = unproj_cam0
                    valid_lifted_cached = valid_lifted0
            else:
                # 回退到 E
                E_info = self.relative_pose_estimator(
                    kps1[matches[:, 0]], kps2[matches[:, 1]], camera1, camera2
                )
                inlier_matches = matches[E_info["inlier_mask"]]
                T_c2w = E_info["cam2_from_cam1"] * T_c1w
                matches_used = inlier_matches
                # 用 E 位姿进行三角化，估计尺度
                pts_tri_e = self._candidate_points3D_for_init(
                    T_c1w, T_c2w, inlier_matches, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
                )
                if len(pts_tri_e["xyz"]) > 0:
                    tri_world = np.vstack(pts_tri_e["xyz"])  # (N,3) 世界
                    tri_cam1 = T_c1w * tri_world
                    z = tri_cam1[:, -1]
                    mask = np.array(list(pts_tri_e["pt2d_id_1"]))
                    d = self.mpsfm_rec.images[imid1].depth.data_prior_at_kps(kps1[mask])
                    rescale = np.median(z / d)

        # 统一候选生成（若无需重标定且已有缓存，则复用 lift）
        if rescale == 1 and unproj_cam_cached is not None:
            unproj_cam, valid_lifted = unproj_cam_cached, valid_lifted_cached
        else:
            unproj_cam, valid_lifted = self._lift_points_for_init(imid1, kps1, camera1, rescale=rescale)
        unproj_world = T_c1w.inverse() * unproj_cam
        pts_lift = self._candidate_lift_for_init(
            T_c1w, T_c2w, matches_used[valid_lifted[matches_used[:, 0]]], unproj_world
        )
        pts_tri = self._candidate_points3D_for_init(
            T_c1w, T_c2w, matches_used, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
        )

        # 合并
        cand = {}
        ids1, ids2 = pts_lift["pt2d_id_1"], pts_tri["pt2d_id_1"]
        set1, set2 = set(ids1), set(ids2)
        idx1 = [i for i, x in enumerate(ids1) if x not in set2]
        idx2 = [i for i, x in enumerate(ids2) if x not in set1]
        common = list(set1 & set2)
        lift_com = {k: [v for v, i in zip(vals, ids1) if i in common] for k, vals in pts_lift.items()}
        tri_com = {k: [v for v, i in zip(vals, ids2) if i in common] for k, vals in pts_tri.items()}
        for k in pts_lift:
            cand[k] = [a if t < self.conf.combined_triangle_thresh else b for (a, b, t) in zip(lift_com[k], tri_com[k], pts_tri["tri_angle"])]
            cand[k] += [pts_lift[k][i] for i in idx1 if pts_lift["tri_angle"][i] < self.conf.combined_triangle_thresh]
            cand[k] += [pts_tri[k][i] for i in idx2 if pts_tri["tri_angle"][i] >= self.conf.combined_triangle_thresh]

        # 赋姿并注册
        self.mpsfm_rec.images[imid1].cam_from_world = T_c1w
        self.mpsfm_rec.images[imid2].cam_from_world = T_c2w
        
        self.mpsfm_rec.register_image(imid1)
        self.mpsfm_rec.register_image(imid2)
        if len(cand["xyz"]) < 3:
            print(f"Init pair {imid1} and {imid2} has less than 3 points to triangulate. Not registered")
            return False
        for i, xyz in enumerate(cand["xyz"]):
            track = pycolmap.Track()
            track.add_element(imid1, cand["pt2d_id_1"][i])
            track.add_element(imid2, cand["pt2d_id_2"][i])
            if (
                self.mpsfm_rec.images[imid1].points2D[cand["pt2d_id_1"][i]].has_point3D()
                or self.mpsfm_rec.images[imid2].points2D[cand["pt2d_id_2"][i]].has_point3D()
            ):
                continue
            if (
                self.conf.colmap_options.init_min_tri_angle < cand["tri_angle"][i]
                and cand["posdepth1"][i]
                and cand["posdepth2"][i]
            ):
                self.mpsfm_rec.obs.add_point3D(xyz, track)
        return not len(self.mpsfm_rec.points3D) < 3

    def register_next_image(self, imid, ref_imids=None, **kwargs):
        """Register next image and triangulate points."""
        image = self.mpsfm_rec.images[imid]
        camera = self.mpsfm_rec.rec.cameras[image.camera_id]

        # 检查是否有先验外参
        image_name = image.name
        prior_pose = self.get_prior_pose(image_name)
        if prior_pose is not None:
            image.cam_from_world = prior_pose
            self.mpsfm_rec.register_image(imid)
            return True

        if ref_imids is None:
            ref_imids = self.mpsfm_rec.registered_images.keys()

        self.registration_cache[imid]["store_matches"] = {}
        ref_imids = list(ref_imids)
        ap_min_num_inliers = self.conf.colmap_options.abs_pose_min_num_inliers
        if self.half_ap_min_inliers:
            ap_min_num_inliers = int(ap_min_num_inliers / (1.2**self.half_ap_min_inliers))
        force_registration = self.half_ap_min_inliers >= self.conf.reduce_min_inliers_at_failure

        while True:
            pair2D3D = defaultdict(dict)
            for im_ref_id in ref_imids:
                image_ref = self.mpsfm_rec.images[im_ref_id]
                self._find_2D3D_pairs(im_ref_id, imid, image_ref, image, pair2D3D[im_ref_id])

            points2D, points3D, stack_order, lifted_mask, ids3d = self._process_2D3D_pairs(pair2D3D)

            unique_ids3d, unique_indices, el_to_unique_index = np.unique(ids3d, return_index=True, return_inverse=True)
            triangpts3D = points3D[~lifted_mask][unique_indices]
            triangpts2D = points2D[~lifted_mask][unique_indices]
            if self.conf.lifted_registration:
                liftedpts2D, liftedpts3D = points2D[lifted_mask], points3D[lifted_mask]
            else:
                liftedpts2D, liftedpts3D = np.zeros((0, 2)), np.zeros((0, 3))
            points2D = np.concatenate([triangpts2D, liftedpts2D])
            points3D = np.concatenate([triangpts3D, liftedpts3D])

            if len(points2D) < 3:
                print(f"\nImage {imid} has less than 3 points to triangulate. Not registered")
                return False

            AP_info = self.absolute_pose_estimator(points2D, points3D, camera)
            if AP_info is None:
                print("\nAP estim No inliers found")
                return False

            if AP_info["num_inliers"] < ap_min_num_inliers and not force_registration:
                print(f"\nAP estim Not enough inliers: {ap_min_num_inliers}")
                return False

            inlier_mask = AP_info["inlier_mask"]
            ref_match_sizes = [len(pair2D3D[im_ref_id]["2d"]) for im_ref_id in stack_order]
            split_indices = np.cumsum(ref_match_sizes)[:-1]
            # mapping back masks to correspondences
            t_mask = inlier_mask[: len(triangpts3D)]
            l_mask = inlier_mask[len(triangpts3D) :]
            remapped_inl_mask = np.ones(len(lifted_mask), dtype=bool)
            remapped_inl_mask[lifted_mask] = l_mask
            remapped_inl_mask[~lifted_mask] = t_mask[el_to_unique_index]
            assert (
                set(np.unique(ids3d[t_mask[el_to_unique_index]])) - set(unique_ids3d[inlier_mask[: len(triangpts3D)]])
                == set()
            )

            split_mask = dict(zip(stack_order, np.split(remapped_inl_mask, split_indices)))
            best_id = self.mpsfm_rec.best_next_ref_imid
            self.mpsfm_rec.last_ap_inlier_masks = split_mask

            if self.conf.resample_bunlde:
                compare_ids = set(stack_order)
                compare_ids.remove(best_id)
                compare_ids = list(compare_ids)
                print()
                print("Best:", best_id, "other:", compare_ids)
                print("_+_+_+_+_+_+" * 7)
                print(
                    "TOTS:",
                    split_mask[best_id].sum(),
                    "vs",
                    [split_mask[im_ref_id].sum() for im_ref_id in compare_ids],
                )
                print(
                    "RATIOS:",
                    split_mask[best_id].sum() / len(split_mask[best_id]),
                    "vs",
                    [split_mask[im_ref_id].sum() / len(split_mask[im_ref_id]) for im_ref_id in compare_ids],
                )
                print("_+_+_+_+_+_+" * 7)
                if (
                    split_mask[best_id].sum() / len(split_mask[best_id]) < 0.1
                    and np.nanmax(
                        [split_mask[im_ref_id].sum() / len(split_mask[im_ref_id]) for im_ref_id in compare_ids]
                    )
                    > 0.2
                ):
                    for ref_id in split_mask:
                        if len(split_mask[ref_id]) > 0:

                            if ref_id in self.mpsfm_rec.images[imid].ignore_matches_AP:
                                used = ~self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id]
                                self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id][used] |= split_mask[ref_id]
                            else:
                                self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id] = split_mask[ref_id]

                    continue

            image.cam_from_world = AP_info["cam_from_world"]
            self.mpsfm_rec.register_image(imid)
            break

        return True

    def register_and_triangulate_next_image(self, imid, ref_imids=None):
        """Register next image and triangulate points."""
        if not self.register_next_image(imid, ref_imids=ref_imids):
            return False

        return self.triangulate_image(imid)

    def _init_pair_points_and_pose(self, imid1, imid2, kps1, kps2, matches, camera1, camera2):
        E_info = self.relative_pose_estimator(kps1[matches[:, 0]], kps2[matches[:, 1]], camera1, camera2)

        inlier_matches = matches[E_info["inlier_mask"]]
        points_triangulated = self._candidate_points3D_for_init(
            pycolmap.Rigid3d(),
            E_info["cam2_from_cam1"],
            inlier_matches,
            self.mpsfm_rec.images[imid1],
            self.mpsfm_rec.images[imid2],
            camera1,
            camera2,
        )

        unproj_3D1, valid_lifted = self._lift_points_for_init(imid1, kps1, camera1)
        valid_matches = matches[valid_lifted[matches[:, 0]]]
        AP_info = self.absolute_pose_estimator(kps2[valid_matches[:, 1]], unproj_3D1[valid_matches[:, 0]], camera2)
        triangles = np.array(points_triangulated["tri_angle"])
        if AP_info is None:
            high_parallax = True
        else:
            high_parallax = (triangles > self.conf.parallax_thresh).sum() > AP_info["num_inliers"]
        if self.conf.verbose > 1:
            print(" -- INIT INFO --")
            print(f"\t        num E inliers: {triangles.shape[0]}")
            print(f"\tnum E w/ triangle>1.5: {(triangles>1.5).sum()}")
            print(
                f"\tnum E w/ triangle>{self.conf.combined_triangle_thresh}: "
                f"\t{(triangles>self.conf.combined_triangle_thresh).sum()}"
            )
        if AP_info is not None and self.conf.verbose > 1:
            print(f"\t       num AP inliers: {AP_info['num_inliers']}")
        if high_parallax:
            cam_from_world2 = E_info["cam2_from_cam1"]

            # gathering lifted and triangulated points
            triangulated_z = np.vstack(points_triangulated["xyz"])[:, -1]
            mask = np.array(list(points_triangulated["pt2d_id_1"]))
            d = self.mpsfm_rec.images[imid1].depth.data_prior_at_kps(kps1[mask])
            rescale = np.median(triangulated_z / d)
            unproj_3D1, valid_lifted = self._lift_points_for_init(imid1, kps1, camera1, rescale=rescale)
            valid_matches = inlier_matches[valid_lifted[inlier_matches[:, 0]]]
            points_lifted = self._candidate_lift_for_init(
                pycolmap.Rigid3d(), cam_from_world2, valid_matches, unproj_3D1
            )

        else:
            cam_from_world2 = AP_info["cam_from_world"]
            points_lifted = self._candidate_lift_for_init(
                pycolmap.Rigid3d(), cam_from_world2, valid_matches, unproj_3D1, AP_info["inlier_mask"]
            )
            # gathering lifted and triangulated points
            points_triangulated = self._candidate_points3D_for_init(
                pycolmap.Rigid3d(),
                cam_from_world2,
                valid_matches[AP_info["inlier_mask"]],
                self.mpsfm_rec.images[imid1],
                self.mpsfm_rec.images[imid2],
                camera1,
                camera2,
            )

        # combining lifted and triangulated points
        candidate_points = {}
        ids1, ids2 = points_lifted["pt2d_id_1"], points_triangulated["pt2d_id_1"]
        setids1, setids2 = set(ids1), set(ids2)
        indices1 = [i for i, x in enumerate(ids1) if x not in setids2]
        indices2 = [i for i, x in enumerate(ids2) if x not in setids1]
        common_elements = list(setids1 & setids2)
        common_points_lifted = {
            k: [v for v, id in zip(values, ids1) if id in common_elements] for k, values in points_lifted.items()
        }
        common_points_triangulated = {
            k: [v for v, id in zip(values, ids2) if id in common_elements] for k, values in points_triangulated.items()
        }
        for k in points_lifted:
            candidate_points[k] = [
                a if tri_angle < self.conf.combined_triangle_thresh else b
                for (a, b, tri_angle) in zip(
                    common_points_lifted[k], common_points_triangulated[k], common_points_triangulated["tri_angle"]
                )
            ]
            candidate_points[k] += [
                points_lifted[k][i]
                for i in indices1
                if points_lifted["tri_angle"][i] < self.conf.combined_triangle_thresh
            ]
            candidate_points[k] += [
                points_triangulated[k][i]
                for i in indices2
                if points_triangulated["tri_angle"][i] >= self.conf.combined_triangle_thresh
            ]
        return candidate_points, cam_from_world2

    def _collect_pairs(
        self, im_ref_id, image_ref, image, pts2d_ids_ref, pts2d_ids_qry, use_3d, point3D_ids, pair2D3D, **kwrags
    ):
        used_matches = []

        pair2D3D["2d"] = np.array([image.points2D[pt].xy for pt in pts2d_ids_qry])
        pair2D3D["3d"] = np.ones((pts2d_ids_ref.shape[0], 3)) * -1
        if sum(use_3d) > 0:
            if self.conf.robust_triangles is not None and self.conf.lifted_registration:
                risky_mask = self.mpsfm_rec.find_points3D_with_small_triangulation_angle(
                    min_angle=self.conf.robust_triangles, point3D_ids=point3D_ids
                )
                use_3d[use_3d] &= ~risky_mask
                point3D_ids = point3D_ids[~risky_mask]
            if sum(use_3d) > 0:
                pair2D3D["3d"][use_3d] = np.array([self.mpsfm_rec.points3D[pt].xyz for pt in point3D_ids])
        if self.conf.lifted_registration:
            if (~use_3d).sum() > 0:
                pts2Dids_ref_lifted = pts2d_ids_ref[~use_3d]
                pair2D3D["3d"][~use_3d] = self._lift_points_to_3d(
                    im_ref_id,
                    image_ref,
                    [self.mpsfm_rec.images[im_ref_id].points2D[pt].xy for pt in pts2Dids_ref_lifted],
                )

            pair2D3D["lifted"] = ~use_3d
            if len(point3D_ids) != (~pair2D3D["lifted"]).sum():
                print("here")
        if not self.conf.lifted_registration:
            pair2D3D["3d"] = pair2D3D["3d"][use_3d]
            pair2D3D["2d"] = pair2D3D["2d"][use_3d]
        pair2D3D["3dids"] = point3D_ids
        return used_matches, pair2D3D

    def _lift_points_to_3d(self, im_ref_id, image_ref, liftref_2d):
        """Lift 2D points to 3D space using depth maps and camera transformations."""
        xy = np.array(liftref_2d)
        d = self.mpsfm_rec.images[im_ref_id].depth.data_at_kps(xy)[:, None]
        camera_ref = self.mpsfm_rec.rec.cameras[image_ref.camera_id]
        return image_ref.cam_from_world.inverse() * (
            np.concatenate([camera_ref.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d
        )

    def _lift_points_for_init(self, im_ref_id, liftref_2d, camera_ref, rescale=1):
        """Lift 2D points to 3D space using depth maps and camera transformations."""
        xy = np.array(liftref_2d)
        d = self.mpsfm_rec.images[im_ref_id].depth.data_prior_at_kps(xy)[:, None]
        if rescale != 1:
            d *= rescale
        valid = self.mpsfm_rec.images[im_ref_id].depth.valid_at_kps(xy)
        return (np.concatenate([camera_ref.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d), valid

    def _process_2D3D_pairs(self, pair2D3D):
        sorted_ref_ids = sorted(pair2D3D)
        pts_2D = np.concatenate([pair2D3D[ref_id]["2d"] for ref_id in sorted_ref_ids])
        pts_3D = np.concatenate([pair2D3D[ref_id]["3d"] for ref_id in sorted_ref_ids])
        if self.conf.lifted_registration:
            lifted = np.concatenate(
                [pair2D3D[ref_id]["lifted"] for ref_id in sorted_ref_ids if "lifted" in pair2D3D[ref_id]]
            )
        if not self.conf.lifted_registration or (~lifted).sum() > 0:
            if all(len(pair2D3D[ref_id]["3dids"]) == 0 for ref_id in sorted_ref_ids if "3dids" in pair2D3D[ref_id]):
                ids3d = np.zeros(0, dtype=int)
            else:
                ids3d = np.concatenate(
                    [pair2D3D[ref_id]["3dids"] for ref_id in sorted_ref_ids if "3dids" in pair2D3D[ref_id]]
                )
            if not self.conf.lifted_registration:
                lifted = np.zeros(len(ids3d), dtype=bool)
        else:
            ids3d = np.zeros(0, dtype=int)
        assert (~lifted).sum() == len(ids3d)
        return pts_2D, pts_3D, sorted_ref_ids, lifted, ids3d

    def triangulate_image(self, imid, **kwargs):
        """Triangulate points for the given image id."""
        return self.triangulator.triangulate_image(imid, **kwargs)

    def _candidate_lift_for_init(self, cam_from_world1, cam_from_world2, matches, lifted3D, inliers=None):
        """Collect candidate points for lifted 3D points."""
        if inliers is None:
            inliers = slice(None)

        candidate_points = defaultdict(list)

        for match in matches[inliers]:
            pt2d_id_1, pt2d_id_2 = match
            xyz = lifted3D[pt2d_id_1]

            projection_center1 = cam_from_world1.rotation.inverse() * -cam_from_world1.translation
            projection_center2 = cam_from_world2.rotation.inverse() * -cam_from_world2.translation
            tri_angle = calculate_triangulation_angle(projection_center1, projection_center2, xyz)
            posdepth1 = has_point_positive_depth(cam_from_world1.matrix(), xyz)
            posdepth2 = has_point_positive_depth(cam_from_world2.matrix(), xyz)
            candidate_points["pt2d_id_1"].append(pt2d_id_1)
            candidate_points["pt2d_id_2"].append(pt2d_id_2)
            candidate_points["tri_angle"].append(np.rad2deg(tri_angle))
            candidate_points["posdepth1"].append(posdepth1)
            candidate_points["posdepth2"].append(posdepth2)
            candidate_points["xyz"].append(xyz)
        return candidate_points
