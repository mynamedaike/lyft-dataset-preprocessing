import os

import numpy as np

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.data import MapAPI
from l5kit.data.map_api import InterpolationMethod

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "./sample"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./agent_motion_config.yaml")


class MapSearch():
    def __init__(self, dm = dm, cfg = cfg):
        self.mapAPI = MapAPI.from_cfg(dm, cfg)
        elements = self.mapAPI.get_bounds()
        self.lane_ids = elements['lanes']['ids']
        self.crosswalk_ids = elements['crosswalks']['ids']

    def _is_point_within_area(self, point, x, y, r):
        dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
        return dist < r

    def _is_line_within_area(self, line, x, y, r):
        return self._is_point_within_area(line[0], x, y, r) or self._is_point_within_area(line[-1], x, y, r)

    def _is_polygon_within_area(self, polygon, x, y, r):
        return self._is_point_within_area(polygon[0], x, y, r) or self._is_point_within_area(polygon[1], x, y, r) or self._is_point_within_area(polygon[2], x, y, r) or self._is_point_within_area(polygon[3], x, y, r)

    def _get_stopline_ids(self):
        stopline_ids = []
        for lane_id in self.lane_ids:
            for lane_traffic_control_id in self.mapAPI[lane_id].element.lane.traffic_controls:
                traffic_control_element = self.mapAPI[
                    lane_traffic_control_id.id.decode("utf-8")].element.traffic_control_element
                if traffic_control_element.HasField("pedestrian_crosswalk") and len(
                        traffic_control_element.pedestrian_crosswalk.yield_lines) > 0:
                    stoplines = traffic_control_element.pedestrian_crosswalk.yield_lines
                    for stopline in stoplines:
                        stopline_ids.append(stopline.id.decode("utf-8"))

        return stopline_ids

    def _convert_line_to_polygon(self, line, dist = 0.1):
        diff = np.diff(line, axis=0)

        k = diff[:, 1] / diff[:, 0]
        normal_vector = dist * np.array([k / ((k ** 2 + 1) ** 0.5), -1 / ((k ** 2 + 1) ** 0.5)])

        line0 = line - normal_vector
        line1 = line + normal_vector

        return np.concatenate((line0, line1))

    def search_lane_centerline(self, x, y, r):
        result = {}
        for lane_id in self.lane_ids:
            centerline = self.mapAPI.get_lane_as_interpolation(lane_id, 0.05, InterpolationMethod.INTER_METER)[
                             'xyz_midlane'][:, :2]

            if self._is_line_within_area(centerline, x, y, r):
                lane = {}
                lane['centerline'] = centerline
                lanes_ahead = self.mapAPI[lane_id].element.lane.lanes_ahead
                lane['successors'] = [lanes_ahead[i].id.decode('utf-8') for i in range(len(lanes_ahead))]
                result[lane_id] = lane

        return result

    def search_lane_boundary(self, x, y, r):
        result = []
        for lane_id in self.lane_ids:
            left_boundary = self.mapAPI.get_lane_as_interpolation(lane_id, 0.05, InterpolationMethod.INTER_METER)[
                                'xyz_left'][:, :2]
            right_boundary = self.mapAPI.get_lane_as_interpolation(lane_id, 0.05, InterpolationMethod.INTER_METER)[
                                 'xyz_right'][:, :2]

            if self._is_line_within_area(left_boundary, x, y, r) or self._is_line_within_area(right_boundary, x, y, r):
                result.append([left_boundary, right_boundary])

        return result

    def search_crosswalk(self, x, y, r):
        result = []
        for crosswalk_id in self.crosswalk_ids:
            crosswalk = self.mapAPI.get_crosswalk_coords(crosswalk_id)['xyz'][:, :2]
            if self._is_polygon_within_area(crosswalk, x, y, r):
                result.append(crosswalk)

        return result

    def search_stopline(self, x, y, r):
        result = []
        stopline_ids = self._get_stopline_ids()
        for stopline_id in stopline_ids:
            stopline = self.mapAPI.get_stopline_coords(stopline_id)['xyz'][:, :2]
            stopline = self._convert_line_to_polygon(stopline)
            if self._is_polygon_within_area(stopline, x, y, r):
                result.append(stopline)

        return result

    def search_static_map(self, x, y, r):
        result = {}

        lanes = self.search_lane_centerline(x, y, r)
        boundaries = self.search_lane_boundary(x, y, r)
        crosswalks = self.search_crosswalk(x, y, r)
        stoplines = self.search_stopline(x, y, r)

        result['lanes'] = lanes
        result['boundaries'] = boundaries
        result['crosswalks'] = crosswalks
        result['stop_lines'] = stoplines

        return result

mapSearch = MapSearch()

elements = mapSearch.search_static_map(740, -2010, 20)
print(elements['lanes'])