function preds_3D_clowd = get_preds_3D_clowd(preds_3D_smoothed, dense_parm)
    [num_pnts, num_frames, ~] = size(preds_3D_smoothed);
    num_points_per_wing = (num_pnts - 4)/2;
    left_inds = 1:num_points_per_wing;
    right_inds =  (num_points_per_wing + 1):(2*num_points_per_wing);
    left_wing_pnts = preds_3D_smoothed(left_inds, :, :);
    right_wing_pnts = preds_3D_smoothed(right_inds, :, :);
    
    left_wing_pnts(num_points_per_wing, :, :) = left_wing_pnts(1, :, :);
    right_wing_pnts(num_points_per_wing, :, :) = right_wing_pnts(1, :, :);
    
    left_wing_clowd = get_spline_wing_clowd(num_frames, num_points_per_wing, left_wing_pnts, 100);
    right_wing_clowd = get_spline_wing_clowd(num_frames, num_points_per_wing, right_wing_pnts, 100);
    
    preds_3D_clowd = cat(2, left_wing_clowd, right_wing_clowd);
    a=0;
    
end

function wing_clowd_smoothed = get_spline_wing_clowd(num_frames, num_points_per_wing, wing_pnts, M)
    cut = ceil(M/10);
    wing_clowd_smoothed = nan(num_frames, M, 3);
    for frame=1:num_frames
        pnts = squeeze(wing_pnts(:, frame, :));
        smoothed_pnts = get_spline_2(pnts, M);
        
        %% fix end-start intersection
%         smoothed_pnts = smoothed_pnts(cut:(M-cut), :);
%         n = M - size(cut:(M-cut), 2);
%         p_cur = squeeze(smoothed_pnts(end, :));
%         p_next = squeeze(smoothed_pnts(1, :));
%         points_line = [linspace(p_cur(1),p_next(1),n); linspace(p_cur(2),p_next(2),n); linspace(p_cur(3),p_next(3),n)]';
%         %         figure; plot3(smoothed_pnts(:,1), smoothed_pnts(:,2), smoothed_pnts(:,3));
%         smoothed_pnts(end+1:M, :) = points_line;

        wing_clowd_smoothed(frame, : , :) = smoothed_pnts;
    end
end


function smoothed_pnts = get_spline_2(pnts, M)
        num_joints = size(pnts,1);
        smoothed_pnts = nan(M, 3);
        curve = cscvn(pnts'); % construct a closed cubic spline curve
        t = linspace(curve.breaks(1),curve.breaks(end),M); % define M parameter values
        samples = fnval(curve,t); % evaluate the curve at t
        smoothed_pnts = samples';
end



function smoothed_pnts = get_spline(pnts, M)
        x = squeeze(pnts(:, 1));
        y = squeeze(pnts(:, 2));
        z = squeeze(pnts(:, 3));
        d = [0; cumsum(sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2))];
        % Interpolate at M equally spaced locations from the start to end of your curve
        t = linspace(0,d(end),M);
        xx = ppval(spline(d,x),t);
        yy = ppval(spline(d,y),t);
        zz = ppval(spline(d,z),t);
        smoothed_pnts = [xx; yy; zz]';
end

function wing_clowd = get_wing_clowd(num_frames, num_points_per_wing, left_wing_pnts, dense_parm)
    m = 200;
    wing_clowd = nan(num_frames, num_points_per_wing, m, 3);
    wing_clowd_smoothed = nan(num_frames, num_points_per_wing* m, 3);
    for frame=1:num_frames
        for pnt=1:num_points_per_wing-1
            p_cur = squeeze(left_wing_pnts(pnt, frame, :));
            p_next = squeeze(left_wing_pnts(pnt+1, frame, :));
            dis = norm(p_cur - p_next);
            n = ceil(dis * dense_parm);
            points_line = [linspace(p_cur(1),p_next(1),n); linspace(p_cur(2),p_next(2),n); linspace(p_cur(3),p_next(3),n)]';
            wing_clowd(frame, pnt, 1:size(points_line, 1), :) = points_line;            
        end
        frame_clowd = reshape(wing_clowd(frame, :, :, :), [1, m*num_points_per_wing, 3]);
        wing_clowd_smoothed(frame, : , :) = frame_clowd;
    end
    sz = size(wing_clowd);
    wing_clowd = reshape(wing_clowd, [sz(1),sz(2)*sz(3), sz(4)]);
    a=0;
end