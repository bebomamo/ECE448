# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    shape = identify_shape(alien) # 1 is circle, 2 is horizontal sausage, 3 is vertical sausage
    center = alien.get_centroid()
    width = alien.get_width()
    length = alien.get_length()
    if(shape == 1):
        corners = [(center[0]-width, center[1]-width),(center[0]-width, center[1]+width),(center[0]+width, center[1]+width),(center[0]+width, center[1]-width)]
        segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
        for segment in segments:
            for wall in walls:
                wall_segment = ((wall[0], wall[1]),(wall[2], wall[3]))
                if(do_segments_intersect(segment, wall_segment)):
                    return True
    if(shape == 2):
        corners = [(center[0]-length/2-width, center[1]-width),(center[0]-length/2-width, center[1]+width),(center[0]+length/2+width, center[1]+width),(center[0]+length/2+width, center[1]-width)]
        segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
        for segment in segments:
            for wall in walls:
                wall_segment = ((wall[0], wall[1]),(wall[2], wall[3]))
                if(do_segments_intersect(segment, wall_segment)):
                    return True
    if(shape == 3):
        corners = [(center[0]-width, center[1]-length/2-width),(center[0]-width, center[1]+length/2+width),(center[0]+width, center[1]+length/2+width),(center[0]+width, center[1]-length/2-width)]
        segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
        for segment in segments:
            for wall in walls:
                wall_segment = ((wall[0], wall[1]),(wall[2], wall[3]))
                if(do_segments_intersect(segment, wall_segment)):
                    return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    vertices = ((0, window[1]), (0,0), (window[0], 0), window)
    center = alien.get_centroid()
    width = alien.get_width()
    length = alien.get_length()
    shape = identify_shape(alien) # 1 is circle, 2 is horizontal sausage, 3 is vertical sausage
    if(shape == 1):
        edges = [(center[0]-width, center[1]-width),(center[0]-width, center[1]+width),(center[0]+width, center[1]+width),(center[0]+width, center[1]-width)]
        # edges = [(center[0]-width, center[1]),(center[0], center[1]-width),(center[0]+width, center[1]),(center[0], center[1]+width)]
        for edge in edges:
            if(not is_point_in_polygon(edge, vertices)):
                return False
    if(shape == 2):
        edges = [(center[0]-length/2-width, center[1]-width),(center[0]-length/2-width, center[1]+width),(center[0]+length/2+width, center[1]+width),(center[0]+length/2+width, center[1]-width)]
        # edges = [(center[0]-length/2-width, center[1]),(center[0], center[1]-width),(center[0]+length/2+width, center[1]),(center[0], center[1]+width)]
        for edge in edges:
            if(not is_point_in_polygon(edge, vertices)):
                return False
    if(shape == 3):
        edges = [(center[0]-width, center[1]-length/2-width),(center[0]-width, center[1]+length/2+width),(center[0]+width, center[1]+length/2+width),(center[0]+width, center[1]-length/2-width)]
        # edges = [(center[0]-width, center[1]),(center[0], center[1]-length/2-width),(center[0]+width, center[1]),(center[0], center[1]+length/2+width)]
        for edge in edges:
            if(not is_point_in_polygon(edge, vertices)):
                return False

    return True

# identifies the aliens current shape
def identify_shape(alien: Alien):
    if(alien.is_circle()):
        shape = 1
    else:
        head_and_tail = alien.get_head_and_tail()
        if(head_and_tail[0][0]-head_and_tail[1][0] == 0):
            shape = 3
        else:
            shape = 2
    return shape


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    #create segments of the polygon
    s1,s2,s3,s4 = ((polygon[0], polygon[1]),(polygon[1], polygon[2]),(polygon[3], polygon[2]),(polygon[0], polygon[3]))

    #positive if on one side of line, negative if on the other side of line
    if(point_segment_distance(point, s1) < 0 and point_segment_distance(point, s2) < 0):
        if(point_segment_distance(point, s3) >= 0 and point_segment_distance(point, s4) >= 0):
            return True
    elif(point_segment_distance(point, s1) < 0 and point_segment_distance(point, s2) > 0):
        if(point_segment_distance(point, s3) >= 0 and point_segment_distance(point, s4) <= 0):
            return True
    elif(point_segment_distance(point, s1) > 0 and point_segment_distance(point, s2) < 0):
        if(point_segment_distance(point, s3) <= 0 and point_segment_distance(point, s4) >= 0):
            return True
    elif(point_segment_distance(point, s1) > 0 and point_segment_distance(point, s2) > 0):
        if(point_segment_distance(point, s3) <= 0 and point_segment_distance(point, s4) <= 0):
            return True
    elif(point_segment_distance(point, s1) == 0 or point_segment_distance(point, s2) == 0 or point_segment_distance(point, s3) == 0 or point_segment_distance(point, s4) == 0):
        return True
    return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    strait_line_path = (alien.get_centroid(), waypoint)
    for wall in walls:
        wall_segment = ((wall[0], wall[1]),(wall[2], wall[3]))
        if do_segments_intersect(strait_line_path, wall_segment):
            return True
    # shape = identify_shape(alien)
    # center = alien.get_centroid()
    # width = alien.get_width()
    # length = alien.get_length()
    # distance = np.sqrt((waypoint[1]-center[1])**2 + (waypoint[0]-center[0])**2)
    # sin_beta,cos_alpha = (waypoint[0]-center[0])/distance #.514
    # cos_beta,sin_alpha = (waypoint[1]-center[1])/distance #.857
    # if(waypoint[0] - center[0] == 0):
    #     #vertical movement
    #     return 0 # temp
    # elif(shape == 1): # Circle
    #     x_pos = center[0] - (cos_alpha*width)
    #     y_pos = center[1] - (sin_alpha*width)
    #     center_corner_one = (x_pos - (cos_beta*width), y_pos - (sin_beta*width))
    #     center_corner_two = (x_pos + (cos_beta*width), y_pos + (sin_beta*width))
    #     corners = [(center[0]-width, center[1]-width),(center[0]-width, center[1]+width),(center[0]+width, center[1]+width),(center[0]+width, center[1]-width)]
    #     path_polygon_segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
    # elif(shape == 2): # Horizontal
    #     corners = [(center[0]-length/2-width, center[1]-width),(center[0]-length/2-width, center[1]+width),(center[0]+length/2+width, center[1]+width),(center[0]+length/2+width, center[1]-width)]
    #     segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
    # elif(shape == 3): # Vertical
    #     corners = [(center[0]-width, center[1]-length/2-width),(center[0]-width, center[1]+length/2+width),(center[0]+width, center[1]+length/2+width),(center[0]+width, center[1]-length/2-width)]
    #     segments = ((corners[0],corners[1]),(corners[1],corners[2]),(corners[2],corners[3]),(corners[3],corners[0]))
    return False

# a and s are tuples ordered (x,y) representing vectors or points(euclidean)
def dot_product(a, s):
    return (a[0]*s[0] + a[1]*s[1])
def cross_product(a, s):
    return (a[0]*s[1] - a[1]*s[0])
def euclidean(a, s): 
    return np.sqrt((s[0]-a[0])**2 + (s[1]-a[1])**2)

def is_between(x, x1, x2):
    #we don't know how x1 and x2 are ordered
    if(x1 < x2):
        if(x >= x1 and x <= x2):
            return True
    elif(x2 < x1):
        if(x >= x2 and x <= x1):
            return True
    else:
        if(x == x1):
            return True
        else:
            return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    # In essence we are just doing calc 3 and trig to find the euclidean distance:
    # distance fromn p to closest point(q) = |p to start of s(a)|sin(angle from a created with s(alpha))
    # = |a|sin(alpha) = (a x s) / |s| = (|a||s|sin(alpha)) / |s| = |a|sin(alpha)... voila
    # and start of s to q = |a|cos(alpha) = (a . s) / |s| = (|a||s|cos(alpha)) / |s| = |a|cos(alpha) ... again voila
    s_length = euclidean(s[0], s[1])
    #create vectors normalized by the position of the base of s
    p_norm = (p[0] - s[0][0], p[1] - s[0][1])
    s_norm = (s[1][0] - s[0][0], s[1][1] - s[0][1])
    p_cross_s = cross_product(p_norm, s_norm)
    projection = dot_product(p_norm, s_norm) / s_length
    if (projection <= s_length and projection >= 0):
        return p_cross_s / s_length
    else: #its an endpoint
        scalar = -1 if p_cross_s < 0 else 1
        if (p_cross_s == 0):
            #might be collinear
            x1, x2, y1, y2 = s[0][0], s[1][0], s[0][1], s[1][1]
            m = 0
            m_vertical = False
            if(x2-x1 == 0):
                #then its a vertical line
                m_vertical = True
            else:
                m = (y2-y1)/(x2-x1)
            b = y1-m*x1
            y = m*p[0] + b
            if(is_between(y, y1, y2)):
                return 0
            return scalar*euclidean(p,s[0]) if (euclidean(p,s[0]) < euclidean(p,s[1])) else scalar*euclidean(p,s[1])
        return scalar*euclidean(p,s[0]) if (euclidean(p,s[0]) < euclidean(p,s[1])) else scalar*euclidean(p,s[1])


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # Tests for conflicting ccw between segments, as well as a quick check for collinearity

    #first line
    x1, x2, y1, y2 = s1[0][0], s1[1][0], s1[0][1], s1[1][1]
    m1 = 0
    m1_vertical = False
    if(x2-x1 == 0):
        #then its a vertical line
        m1_vertical = True
    else:
        m1 = (y2-y1)/(x2-x1)
    b1 = y1-m1*x1

    #second line
    x1, x2, y1, y2 = s2[0][0], s2[1][0], s2[0][1], s2[1][1]
    m2 = 0
    m2_vertical = False
    if(x2-x1 == 0):
        #then its a vertical line
        m2_vertical = True
    else:
        m2 = (y2-y1)/(x2-x1)
    b2 = y1-m2*x1

    # vertical line checks
    if(m1_vertical and m2_vertical): #both vertical
        if(s1[0][0] == s2[0][0]): #both on same vertical line
            if(is_between(s2[0][1], s1[0][1], s1[1][1]) or is_between(s2[1][1], s1[0][1], s1[1][1])):
                return True
        return False

    #collinearity but not vertical test
    if not (m1_vertical or m2_vertical):
        if(m1==m2):
            if(b1!=b2):
                return False
            elif(is_between(s2[0][1], s1[0][1], s1[1][1]) or is_between(s2[1][1], s1[0][1], s1[1][1]) or is_between(s2[0][0], s1[0][0], s1[1][0]) or is_between(s2[1][0], s1[0][0], s1[1][0])):
                return True

    return ccw(s1[0],s2[0],s2[1]) != ccw(s1[1],s2[0],s2[1]) and ccw(s1[0],s1[1],s2[0]) != ccw(s1[0],s1[1],s2[1])


# determines if a set of points is counterclockwise in order A, B, C
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if(do_segments_intersect(s1, s2)):
        return 0
    
    #get distances from each endpoint to the other segment
    distances = []
    distances.append(abs(point_segment_distance(s1[0], s2)))
    distances.append(abs(point_segment_distance(s1[1], s2)))
    distances.append(abs(point_segment_distance(s2[0], s1)))
    distances.append(abs(point_segment_distance(s2[1], s1)))

    min_dist = float('inf')
    for dist in distances:
        if dist < min_dist:
            min_dist = dist
    
    return min_dist


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
