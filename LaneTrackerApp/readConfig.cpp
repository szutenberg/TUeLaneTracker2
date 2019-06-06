/*
 * readConfig.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#include "readConfig.h"
#include "opencv2/opencv.hpp"

using namespace cv;

int readConfig(string path, LaneTracker::Config* cfg)
{
	FileStorage fs(path, FileStorage::READ);

	if (!fs.isOpened())
	{
		cerr << "Failed to open config file " << path << "\n";
		return -1;
	}

	cfg->lane_avg_width = fs["lane_avg_width"];
	cfg->lane_std_width = fs["lane_std_width"];
	cfg->lane_min_width = fs["lane_min_width"];
	cfg->lane_max_width = fs["lane_max_width"];
	cfg->lane_marker_width = fs["lane_marker_width"];
	cfg->cam_res_v = fs["cam_res_v"];
	cfg->cam_res_h = fs["cam_res_h"];
	cfg->cam_fx = fs["cam_fx"];
	cfg->cam_fy = fs["cam_fy"];
	cfg->cam_cx = fs["cam_cx"];
	cfg->cam_cy = fs["cam_cy"];
	cfg->cam_pitch = fs["cam_pitch"];
	cfg->cam_yaw = fs["cam_yaw"];
	cfg->cam_height = fs["cam_height"];
	cfg->cam_lateral_offset = fs["cam_lateral_offset"];
	cfg->base_line_IBCS = fs["base_line_IBCS"];
	cfg->purview_line_IBCS = fs["purview_line_IBCS"];
	cfg->step_lane_filter_cm = fs["step_lane_filter_cm"];
	cfg->step_vp_filter = fs["step_vp_filter"];
	cfg->vp_range_ver = fs["vp_range_ver"];
	cfg->vp_range_hor = fs["vp_range_hor"];
	cfg->buffer_count = fs["buffer_count"];
	cfg->display_graphics = ((int)fs["display_graphics"]) != 0;
	cfg->print_json = ((int)fs["print_json"]) != 0;
	cfg->curve_detector = fs["curve_detector"];
	cfg->neural_network = fs["neural_network"];

	fs.release();

	return 0;
}
