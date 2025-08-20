function [predictions, box, body_parts_2D, seg_scores] = fix_wings_3d_per_frame(predictions, body_parts_2D, easyWandData, cropzone, box, seg_scores)
    %[cam1, cam2, cam3]
    which_to_flip = logical([[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]]);
    num_of_options = size(which_to_flip, 1);
    test_preds = predictions(:, :, :, :);
    num_frames = size(test_preds, 1);
    num_joints = size(test_preds, 3);
    im_size = size(box, 1);
    cam_inds=1:size(predictions,2);
    num_cams = size(predictions,2);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    inds = [left_inds; right_inds];
    aranged_predictions = predictions;
    masks = permute(squeeze(box(:, :, [2,3], :, :)), [5,4,1,2,3]); 
    %% fix right-left consistency
    for frame = 1:num_frames
       % [chosen_cam, cams_to_check, wings_sz] = find_best_wings_cam(squeeze(masks(frame,:,:,:,:)), queeze(seg_scores(frame, :,:)));
        chosen_cam = 1; cams_to_check = (2:4);
        if frame > 1   % fix let-right of the chosen camera
            prev_left_pts = squeeze(aranged_predictions(frame - 1, chosen_cam, left_inds, :));
            cur_left_pts = squeeze(aranged_predictions(frame, chosen_cam, left_inds, :));
            prev_right_pts = squeeze(aranged_predictions(frame - 1, chosen_cam, right_inds, :));
            cur_right_pts = squeeze(aranged_predictions(frame, chosen_cam, right_inds, :));
            l2l_dist = get_points_distance(cur_left_pts, prev_left_pts);
            r2r_dist = get_points_distance(cur_right_pts, prev_right_pts);
            l2r_dist = get_points_distance(cur_left_pts, prev_right_pts);
            r2l_dist = get_points_distance(cur_right_pts, prev_left_pts);
            do_switch = l2l_dist + r2r_dist >= l2r_dist + r2l_dist;
            if do_switch  % do switch to chosen camera
                % switch masks
                mask1 = masks(frame, chosen_cam, :, :, 1);
                mask2 = masks(frame, chosen_cam, :, :, 2);
                masks(frame, chosen_cam, :, :, 1) = mask2;
                masks(frame, chosen_cam, :, :, 2) = mask1;
                % switch predictions
                left_pnts = aranged_predictions(frame, chosen_cam, left_inds, :);
                right_pnts = aranged_predictions(frame, chosen_cam, right_inds, :);
                aranged_predictions(frame, chosen_cam, left_inds, :) = right_pnts;
                aranged_predictions(frame, chosen_cam, right_inds, :) = left_pnts;
                % switch scores
                score_1 = seg_scores(frame, chosen_cam, 1);
                score_2 = seg_scores(frame, chosen_cam, 2);
                seg_scores(frame, chosen_cam, 1) = score_2;
                seg_scores(frame, chosen_cam, 2) = score_1;

            end
        end
        frame_scores = zeros(num_of_options, 1);
        % loop on every option of other cameras, flip the cameras, and get the score 
        for option = 1:num_of_options  
            test_preds_i = aranged_predictions(frame, :, :, :);
            cams_to_flip_inds = which_to_flip(option, :);
            if option ~= 1  % 1 is [0,0,0] means don't flip any camera
                cams_to_flip = cams_to_check(cams_to_flip_inds);
                %% flip the relevant cameras
                 for cam=cams_to_check
                    if ismember(cam, cams_to_flip)
                        left_wings_preds = test_preds_i(:, cam, left_inds, :);
                        right_wings_preds = test_preds_i(:, cam, right_inds, :);
                        test_preds_i(:, cam, right_inds, :) = left_wings_preds;
                        test_preds_i(:, cam, left_inds, :) = right_wings_preds;
                    end
                 end 
            end
            crop = cropzone(:, :, frame);
            [~, ~ ,test_3d_pts] = get_3d_pts_rays_intersects(test_preds_i, easyWandData, crop, cam_inds);
            test_3d_pts = squeeze(test_3d_pts);
            total_boxes_volume = 0;
            for pnt=1:num_joints
                joint_pts = squeeze(test_3d_pts(pnt, :, :)); 
                cloud = pointCloud(joint_pts);
                x_limits = cloud.XLimits;
                y_limits = cloud.YLimits;
                z_limits = cloud.ZLimits;
                x_size = abs(x_limits(1) - x_limits(2)); 
                y_size = abs(y_limits(1) - y_limits(2));
                z_size = abs(z_limits(1) - z_limits(2));
                box_volume = x_size*y_size*z_size;
                total_boxes_volume = total_boxes_volume + box_volume;
            end
            frame_scores(option) = total_boxes_volume;
        end
        [M,I] = min(frame_scores,[],'all');
        winning_option = which_to_flip(I, :);
        if I == 1
            continue
        end
        cams_to_flip = cams_to_check(winning_option);
        for cam=cams_to_check
            if ismember(cam, cams_to_flip)
                % switch left and right prediction indexes
                mask1 = masks(frame, cam, :, :, 1);
                mask2 = masks(frame, cam, :, :, 2);
                masks(frame, cam, :, :, 1) = mask2;
                masks(frame, cam, :, :, 2) = mask1;
                % switch predictions
                left_pnts = aranged_predictions(frame, cam, left_inds, :);
                right_pnts = aranged_predictions(frame, cam, right_inds, :);
                aranged_predictions(frame, cam, left_inds, :) = right_pnts;
                aranged_predictions(frame, cam, right_inds, :) = left_pnts;
                % switch scores
                score_1 = seg_scores(frame, cam, 1);
                score_2 = seg_scores(frame, cam, 2);
                seg_scores(frame, cam, 1) = score_2;
                seg_scores(frame, cam, 2) = score_1;
            end
        end
    end
    predictions = aranged_predictions;
    %% find wing joints sides
    for frame=1:num_frames
        for cam=1:num_cams
             left_wj = squeeze(body_parts_2D(frame, cam, 1, :)); 
             right_wj = squeeze(body_parts_2D(frame, cam, 2, :)); 
             left_mask = squeeze(masks(frame, cam, :, :, 1));
             right_mask = squeeze(masks(frame, cam, :, :, 2));
             dist_r2r  = get_distance_from_mask_to_point(right_mask, right_wj);
             dist_r2l = get_distance_from_mask_to_point(right_mask, left_wj);
             dist_l2l  = get_distance_from_mask_to_point(left_mask, left_wj);
             dist_l2r = get_distance_from_mask_to_point(left_mask, right_wj);
             if dist_r2l < dist_r2r || dist_l2r < dist_l2l
                % switch body points location 
                body_parts_2D(frame, cam, 1, :) = right_wj;
                body_parts_2D(frame, cam, 2, :) = left_wj;
             end
        end
    end
    
    %% create new wings masks based on 3D back-projection
    add_wings_joints_to_masks = true;
    if add_wings_joints_to_masks
        aranged_predictions(:, :, (num_joints+1:num_joints+2), :) = body_parts_2D(:, :, [1,2], :);
        left_inds(num_joints/2+1) = num_joints+1;
        right_inds(num_joints/2+1) = num_joints+2;
        inds = [left_inds; right_inds];
    end
    [~, ~ ,all_pts_3D] = get_3d_pts_rays_intersects(aranged_predictions, easyWandData, cropzone, cam_inds);
    points_3D = get_avarage_points_3d(all_pts_3D, 1, 1.5);
    points_3D_sm = smooth_3d_points(points_3D, 3, 0.9999995);
    points_2D_proj = from_3D_pts_to_pixels(points_3D_sm, easyWandData, cropzone);  % get the bach-projections
    new_masks = zeros(size(masks));
    for frame=1:num_frames
        for cam=1:num_cams
            for wing=1:2
                wing_inds = inds(wing, :);
                wing_pnts = double(squeeze(points_2D_proj(frame, cam,wing_inds, :)));
                k = convhull(wing_pnts(:,1),wing_pnts(:,2));
                new_mask = poly2mask(wing_pnts(k,1),wing_pnts(k,2),im_size,im_size);
                se = strel('disk',5);
                % Dilate the binary image using the disk-shaped structuring element
                new_mask = imdilate(new_mask,se);
                new_masks(frame, cam, :, :, wing) = new_mask;
            end
        end
    end
    %%
    masks_per = permute(new_masks, [3,4,5,2,1]);
    box(:, :, [2,3], :, :) = masks_per;
    
end

function dist = get_points_distance(pnts_1, pnts_2)
    dist = mean(vecnorm(pnts_1 - pnts_2, 2,2));
end

function [chosen_cam, cameras_to_test, wings_sz] = find_best_wings_cam(all_4_masks, all_4_masks_scores)
        num_cams = size(all_4_masks, 1);
        wings_sz = zeros(4,4);  % size1, size2, size*size2, size*size2*scores
        for cam=1:num_cams
            wing1_size = nnz(squeeze(all_4_masks(cam, :, :, 1)));
            wing2_size = nnz(squeeze(all_4_masks(cam, :, :, 2)));
            combined_sz = wing1_size * wing2_size;
            wings_sz(cam,1) = wing1_size;
            wings_sz(cam,2) = wing2_size;
            score1 = all_4_masks_scores(cam, 1);
            score2 = all_4_masks_scores(cam, 2);
            wings_sz(cam,3) = combined_sz;
            wings_sz(cam,4) = combined_sz * score1 * score2;
        end
        [M, chosen_cam] = max(wings_sz(:, 3));
        all_cams = (1:num_cams);
        cameras_to_test = all_cams(all_cams ~= chosen_cam);
end

