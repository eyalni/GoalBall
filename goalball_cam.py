import os
import sys
import configparser
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imagesc as imagesc
import math
# import cv2
# from cv2_rolling_ball import subtract_background_rolling_ball
import ball_net as bn


# Definitions
MIN_NUM_OF_FRMAES = 10
NUM_OF_FRAMES_IN_BUFFER = 300
NUM_OF_FRAMES_FOR_BACKGROUND_CALC = 10

HEIGHT_INDEX = 0
WIDTH_INDEX = 1
COLOR_INDEX = 2

diff_threshold = [100,-60,-60]
ball_color_threshold = [[60,125], [60,100], [90,140]]
min_ball_vel = 3.0

##
cnt = 0

R = 60
EPS = 1e-6
EPS2 = 0.5

STATUS_INIT = 0
STATUS_STATIC = 1
STATUS_DIRECTED = 2


def pt_dist(x1, y1, x2, y2):
  dx = x1 - x2
  dy = y1 - y2
  return math.sqrt(dx * dx + dy * dy)

class Blob:
  cnt = 1
  def __init__(self, x, y, r, a, frame_ind):
    self.id = Blob.cnt
    Blob.cnt += 1
    self.pts = [[x, y]]
    self.pp = [[r, a]]
    self.status = STATUS_INIT
    self.v = None
    self.age = a
    self.nx = None
    self.ny = None
    self.frame_inds = [frame_ind]

  def fit(self, x, y, r):
    d = pt_dist(self.pts[-1][0], self.pts[-1][1], x, y)
    return d < R, d

  def add(self, x, y, r, a, frame_ind):
    self.pts.append([x, y])
    self.pp.append([r, a])
    self.frame_inds.append(frame_ind)
    self.age = a
    if len(self.pts) > 2:
      #if self.status == STATUS_DIRECTED and self.nx is not None:
      #  print("Predict", self.nx, self.ny, "vs", x, y)

      dx1 = self.pts[-2][0] - self.pts[-3][0]
      dy1 = self.pts[-2][1] - self.pts[-3][1]

      dx2 = x - self.pts[-2][0]
      dy2 = y - self.pts[-2][1]

      d1 = pt_dist(self.pts[-2][0], self.pts[-2][1], x, y)
      d2 = pt_dist(self.pts[-2][0], self.pts[-2][1], self.pts[-3][0], self.pts[-3][1])
      if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d1 > 5 and d2 > 5:
        self.status = STATUS_DIRECTED
        #print("Directed", self.pts)
        #self.predict()
      elif self.status != STATUS_DIRECTED:
        self.status = STATUS_STATIC

  def predict(self):
    npts = np.array(self.pts)
    l = len(self.pts) + 1
    idx = np.array(range(1, l))

    kx = np.polyfit(idx, npts[:,0], 1)
    fkx = np.poly1d(kx)

    ky = np.polyfit(idx, npts[:,1], 1)
    fky = np.poly1d(ky)

    self.nx = fkx(l)
    self.ny = fky(l)
    return self.nx, self.ny

##
B = []
bb = None
prev_bb = None


def get_ball_blob():
    return bb


def find_fblob(x, y, r):
    global B, cnt
    rbp = []
    sbp = []

    for b in B:
        ft, d = b.fit(x, y, r)
        if ft:
            if cnt - b.age < 4:
                rbp.append([b, d])
            elif b.status == STATUS_STATIC:
                sbp.append([b, d])

    if len(sbp) + len(rbp) == 0:
        return None
    rbp.sort(key=lambda e: e[1])
    if len(rbp) > 0:
        return rbp[0][0]

    sbp.sort(key=lambda e: e[1])
    return sbp[0][0]


def handle_blob(x, y, r, frame_ind):
    global B, cnt, bb
    b = find_fblob(x, y, r)
    if b is None:
        B.append(Blob(x, y, r, cnt, frame_ind))
        return
    b.add(x, y, r, cnt, frame_ind)
    if b.status == STATUS_DIRECTED:
        if bb is None:
            bb = b
        elif len(b.pts) > len(bb.pts):
            bb = b


def begin_gen():
    global bb, prev_bb
    prev_bb = bb
    bb = None


def end_gen():
    global cnt, bb
    cnt += 1

##
def calc_background_by_mean(frames , cur_frame_index , num_of_frames_for_background_calc = 10):
    frame_shape = frames.shape

    background_by_mean = np.zeros([frame_shape[HEIGHT_INDEX+1], frame_shape[WIDTH_INDEX+1], frame_shape[COLOR_INDEX+1]])

    if (num_of_frames_for_background_calc < 0):
        print ('num_of_frames_for_background_calc must be greater than 0 (%d)' % (num_of_frames_for_background_calc))
        return background_by_mean

    for i in range(cur_frame_index - num_of_frames_for_background_calc, cur_frame_index):
        temp_frame = frames[i, :, :, :]
        background_by_mean = background_by_mean + temp_frame
    background_by_mean = background_by_mean / num_of_frames_for_background_calc

    return  background_by_mean

##
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

##
def calc_image_from_rolling_ball_alg(img, randius= 2, light_background=False, use_paraboloid=False, do_presmooth=True):
    img_ret, background = subtract_background_rolling_ball(img, randius, light_background=light_background,
                                         use_paraboloid=use_paraboloid, do_presmooth=do_presmooth)
    img_ret_1, background_1 = subtract_background_rolling_ball(img, randius, light_background=light_background,
                                         use_paraboloid=True, do_presmooth=do_presmooth)
    img_ret_2, background_2 = subtract_background_rolling_ball(img, randius, light_background=light_background,
                                         use_paraboloid=True, do_presmooth=False)

    plt.figure(2)
    plt.imshow(img_ret)
    plt.title('img_ret')
    plt.figure(3)
    plt.imshow(img_ret_1)
    plt.title('img_ret_1 (parabolied = True)')
    plt.figure(4)
    plt.imshow(img_ret_2)
    plt.title('img_ret_2 (parabolied = True, presmooth = False)')


    return img_ret, background

##
def calc_diff_img_threshold(frame , back_ground_by_mean_uint8, diff_threshold = [0,-60,-60]):

        # rolling_ball_frame_ball, rolling_ball_frame_background = calc_image_from_rolling_ball_alg(frame_gray_uint8, 2)
        frame_float_64 = frame.astype('float64')
        diff_img = frame_float_64 - back_ground_by_mean_uint8.astype('float64')
        diff_img_int = diff_img.astype('int64')
        diff_uint8 = diff_img.astype('uint8')

        diff_channel_0_thresholded = diff_img[:,:,0] < diff_threshold[0]
        diff_channel_1_thresholded = diff_img[:,:,1] < diff_threshold[1]
        diff_channel_2_thresholded = diff_img[:,:,2] < diff_threshold[2]


        diff_img_int_thresholded = np.multiply(diff_channel_0_thresholded , diff_channel_1_thresholded)
        diff_img_int_thresholded = np.multiply(diff_img_int_thresholded , diff_channel_2_thresholded)
        return diff_img_int_thresholded

##
def find_match_color_img(frame, ball_color_threshold= [[60,125], [60,100], [90,140]]):
    MIN_INDEX = 0
    MAX_INDEX = 1

    # Convert the BRG image to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Convert the RGB image to HSV
    frame_hsv = cv.cvtColor(frame_rgb, cv.COLOR_RGB2HSV)

    c0 = np.multiply(frame[:, :, 0] > ball_color_threshold[0][MIN_INDEX], frame[:, :, 0] < ball_color_threshold[0][MAX_INDEX])
    c1 = np.multiply(frame[:, :, 1] > ball_color_threshold[1][MIN_INDEX], frame[:, :, 1] < ball_color_threshold[1][MAX_INDEX])
    c2 = np.multiply(frame[:, :, 2] > ball_color_threshold[2][MIN_INDEX], frame[:, :, 2] < ball_color_threshold[2][MAX_INDEX])

    combined_channels = np.multiply(c0,c1)
    match_ball_color_img = np.multiply(combined_channels,c2)

    return match_ball_color_img
##
def handle_blobs(mask, frame, frame_ind):
  bin_mask = mask.astype('uint8')
  mask = bin_mask
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  k = 0
  begin_gen()
  for i,c in enumerate(cnts):
    rx,ry,rw,rh  = cv.boundingRect(c)
    mn = min(rw, rh)
    mx = max(rw, rh)
    r = mx / mn
    # if mn < 10 or mx > 40 or r > 1.5:
    if mn < 2 or mx > 4 or r > 1.5:
        continue

    cut_m = mask[ry : ry + rh, rx : rx + rw]

    print ('blob %d\\%d: rx = %d - %d (%d), ry= %d - %d (%d)' % (i, len(cnts),
                                                        rx , rx + rw, rw ,
                                                        ry , ry + rh, rh))

    blob, nz = check_blob(cut_m, 0, 0, rw, rh)
    if not blob:
      continue
    pnz = nz / (rw * rh)
    if pnz < 0.5:
      continue

    cut_f = frame[ry : ry + rh, rx : rx + rw]
    cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)
    # if bn.check_pic(cut_c) != 0:
    #   continue
    ((x, y), r) = cv.minEnclosingCircle(c)
    handle_blob(int(x), int(y), int(r), frame_ind)
    k += 1


  end_gen()
##

def check_blob(pic, x, y, w, h):
  total_cnt_pic = cv.countNonZero(pic)
  total_size = w*h

  fill_percent = total_cnt_pic / total_size

  # Verify that the blob is in the court
  # blob_pt = np.float32([[x, y]])
  # court_pt = cv.perspectiveTransform(blob_pt, M)
  # if is_out_of_court(court_pt):
  #     fill_percent = 0

  return fill_percent > 0.5, total_cnt_pic

  # h_threshold = 0.15
  # v_threshold = 0.15
  # r_threshold = 1.7
  #
  # dy = int(h / 2)
  # y0 = y + 2 * dy
  # cut_h = pic[y0 : y0 + dy, x : x + w]
  #
  # dx = int(w / 2)
  # x0 = x + 2 * dx
  # cut_v = pic[y : y + h, x0 : x0 + dx]
  #
  # hnz = cv.countNonZero(cut_h)
  # vnz = cv.countNonZero(cut_v)
  # nz = cv.countNonZero(pic)
  # mn = min(hnz, vnz)
  # r = max(hnz, vnz) / mn if mn > 0 else 1000
  # h_res = hnz / nz > h_threshold
  # v_res = vnz / nz > v_threshold
  # r_res = r < r_threshold
  #
  # # return r < 1.5 and hnz / nz > 0.15 and vnz / nz > 0.15, nz
  # return r_res and h_res and v_res, nz

##
def get_blobs_to_report(B):
    report_now = False
    blob_inds_to_report = []

    for cur_b in B:
        if cur_b.age < 4:
            continue
        cur_pts = cur_b.pts
        x = [pt[0] for pt in cur_pts]
        y = [pt[1] for pt in cur_pts]
        z = np.polyfit(x, y, 2)

        p = np.poly1d(z)

        min_age = min(cur_b.age , 10)
        if ((np.diff(cur_b.frame_inds)==0).any()):
            continue
        mean_vel_x = np.mean(np.diff(x[0:min_age]) / np.diff(cur_b.frame_inds))
        mean_vel_y = np.mean(np.diff(y[0:min_age]) / np.diff(cur_b.frame_inds))

        vel_strength = np.sqrt(mean_vel_x ** 2 + mean_vel_y ** 2)
        vel_angle = np.arctan2(mean_vel_y, mean_vel_x)

        if (vel_strength < min_ball_vel):
            continue

        court_x = 200
        x_was_found_to_report = np.array([k<court_x for k in x]).any()

        exp_y = p(court_x)

        # cur_x , cur_y = cur_pts[-1]
        if x_was_found_to_report:
            report_now = True
            blob_inds_to_report.append(cur_b)

    return report_now , blob_inds_to_report


##

def calc_start_velocity_vec(b):
    start_vel_angle = 0
    start_vel_strength = 0

    if (len(b) < 2):
        return start_vel_angle, start_vel_strength

    b_0 = b.pts[0]
    b_1 = b.pts[1]
    b_f0 = b.frame_inds[0]
    b_f1 = b.frame_inds[1]
    diff_frames = b_f1 - b_f0
    diff_pt = b_1 - b_0

    vel_x = diff_pt[0] / diff_frames
    vel_y = diff_pt[0] / diff_frames

    start_vel_strength = np.sqrt(vel_x ** 2 + vel_y ** 2)
    start_vel_angle = np.arctan2(vel_y, vel_x)

    return start_vel_angle , start_vel_strength

##
##
class goal_ball:
  def __init__(self, h, w, c=1):
    self.frames = np.zeros([NUM_OF_FRAMES_IN_BUFFER , 1 , 1])
    self.frame_index = 0
    self.h = h
    self.w = w
    self.c = c

    self.num_of_frames_for_background_calc = 10

  ##
  def calc_background_by_mean(self):
    frame_shape = self.frames.shape

    background_by_mean = np.zeros([frame_shape[HEIGHT_INDEX+1], frame_shape[WIDTH_INDEX+1], frame_shape[COLOR_INDEX+1]])

    if (self.num_of_frames_for_background_calc < 0):
        print ('num_of_frames_for_background_calc must be greater than 0 (%d)' % (num_of_frames_for_background_calc))
        return background_by_mean

    min_start_frame_index = max(self.frame_index - num_of_frames_for_background_calc , 0)
    for i in range(min_start_frame_index, self.frame_index):
        temp_frame = self.frames[i, :, :, :]
        background_by_mean = background_by_mean + temp_frame
    background_by_mean = background_by_mean / self.num_of_frames_for_background_calc

    return  background_by_mean

  ##
    return diff_img_int_thresholded


##
  def find_match_color_img(self, frame, ball_color_threshold=[[60, 125], [60, 100], [90, 140]]):
    MIN_INDEX = 0
    MAX_INDEX = 1

    # # Convert the BRG image to RGB
    # frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #
    # # Convert the RGB image to HSV
    # frame_hsv = cv.cvtColor(frame_rgb, cv.COLOR_RGB2HSV)

    c0 = np.multiply(frame[:, :, 0] > ball_color_threshold[0][MIN_INDEX],
                     frame[:, :, 0] < ball_color_threshold[0][MAX_INDEX])
    c1 = np.multiply(frame[:, :, 1] > ball_color_threshold[1][MIN_INDEX],
                     frame[:, :, 1] < ball_color_threshold[1][MAX_INDEX])
    c2 = np.multiply(frame[:, :, 2] > ball_color_threshold[2][MIN_INDEX],
                     frame[:, :, 2] < ball_color_threshold[2][MAX_INDEX])

    combined_channels = np.multiply(c0, c1)
    match_ball_color_img = np.multiply(combined_channels, c2)

    return match_ball_color_img


##
  def handle_blobs(self, mask, frame, frame_ind):
    bin_mask = mask.astype('uint8')
    mask = bin_mask
    cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    k = 0
    self.begin_gen()
    for i, c in enumerate(cnts):
        rx, ry, rw, rh = cv.boundingRect(c)
        mn = min(rw, rh)
        mx = max(rw, rh)
        r = mx / mn
        # if mn < 10 or mx > 40 or r > 1.5:
        if mn < 2 or mx > 4 or r > 1.5:
            continue

        cut_m = mask[ry: ry + rh, rx: rx + rw]

        print('blob %d\\%d: rx = %d - %d (%d), ry= %d - %d (%d)' % (i, len(cnts),
                                                                    rx, rx + rw, rw,
                                                                    ry, ry + rh, rh))

        blob, nz = self.check_blob(cut_m, 0, 0, rw, rh)
        if not blob:
            continue
        pnz = nz / (rw * rh)
        if pnz < 0.5:
            continue

        cut_f = frame[ry: ry + rh, rx: rx + rw]
        cut_c = cv.bitwise_and(cut_f, cut_f, mask=cut_m)
        # if bn.check_pic(cut_c) != 0:
        #   continue
        ((x, y), r) = cv.minEnclosingCircle(c)
        self.handle_blob(int(x), int(y), int(r), frame_ind)
        k += 1

    self.end_gen()


##

  def check_blob(self, pic, x, y, w, h):
    total_cnt_pic = cv.countNonZero(pic)
    total_size = w * h

    fill_percent = total_cnt_pic / total_size

    # Verify that the blob is in the court
    # blob_pt = np.float32([[x, y]])
    # court_pt = cv.perspectiveTransform(blob_pt, M)
    # if is_out_of_court(court_pt):
    #     fill_percent = 0

    return fill_percent > 0.5, total_cnt_pic




##
  def get_blobs_to_report(self, B):
    report_now = False
    blob_inds_to_report = []

    for cur_b in B:
        if cur_b.age < 4:
            continue
        cur_pts = cur_b.pts
        x = [pt[0] for pt in cur_pts]
        y = [pt[1] for pt in cur_pts]
        z = np.polyfit(x, y, 2)

        p = np.poly1d(z)

        min_age = min(cur_b.age, 10)
        if ((np.diff(cur_b.frame_inds) == 0).any()):
            continue
        mean_vel_x = np.mean(np.diff(x[0:min_age]) / np.diff(cur_b.frame_inds))
        mean_vel_y = np.mean(np.diff(y[0:min_age]) / np.diff(cur_b.frame_inds))

        vel_strength = np.sqrt(mean_vel_x ** 2 + mean_vel_y ** 2)
        vel_angle = np.arctan2(mean_vel_y, mean_vel_x)

        if (vel_strength < min_ball_vel):
            continue

        court_x = 200
        x_was_found_to_report = np.array([k < court_x for k in x]).any()

        exp_y = p(court_x)

        # cur_x , cur_y = cur_pts[-1]
        if x_was_found_to_report:
            report_now = True
            blob_inds_to_report.append(cur_b)

    return report_now, blob_inds_to_report


##

  def calc_start_velocity_vec(self, b):
    start_vel_angle = 0
    start_vel_strength = 0

    if (len(b) < 2):
        return start_vel_angle, start_vel_strength

    b_0 = b.pts[0]
    b_1 = b.pts[1]
    b_f0 = b.frame_inds[0]
    b_f1 = b.frame_inds[1]
    diff_frames = b_f1 - b_f0
    diff_pt = b_1 - b_0

    vel_x = diff_pt[0] / diff_frames
    vel_y = diff_pt[0] / diff_frames

    start_vel_strength = np.sqrt(vel_x ** 2 + vel_y ** 2)
    start_vel_angle = np.arctan2(vel_y, vel_x)

    return start_vel_angle, start_vel_strength
##
##
##

  def process_frame(self, frame):
    self.frames[frame_index, :, :, :] = frame

    if (self.frame_index < MIN_NUM_OF_FRMAES):
        frame_index += 1
        return

    back_ground_by_mean = self.calc_background_by_mean(frames , frame_index , NUM_OF_FRAMES_FOR_BACKGROUND_CALC)
    back_ground_by_mean_uint8 = back_ground_by_mean.astype('uint8')

    diff_img_int_thresholded = calc_diff_img_threshold(frame, back_ground_by_mean_uint8, diff_threshold=diff_threshold)
    match_ball_color_img = find_match_color_img(frame, ball_color_threshold)

    potential_mask_img = np.multiply(diff_img_int_thresholded, match_ball_color_img)

    self.handle_blobs(potential_mask_img, frame, frame_index)

    self.frame_index += 1

    plt.figure(1)
    plt.imshow(frame)
    plt.title('%d' % (frame_index))

    report_now, blob_inds_to_report = self.get_blobs_to_report(B)
    if (report_now):
        for blob_ind in blob_inds_to_report:
            cur_blob_points = B[blob_ind].pts
            hit_point = calc_hit_point(cur_blob_points)
            start_vel_angle, start_vel_strength = calc_start_velocity_vec(B[blob_ind])


##
##

def test_clip(path, start_frame = 0):
    vs = cv.VideoCapture(path)

    vs.set(cv.CAP_PROP_POS_FRAMES, start_frame - 1)

    frames = np.zeros([NUM_OF_FRAMES_IN_BUFFER , 1 , 1])

    frame_index = 0
    while (True):
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        frame_shape = frame.shape
        h = frame_shape[HEIGHT_INDEX]
        w = frame_shape[WIDTH_INDEX]
        c = 1
        if (len(frame_shape) > 2):
            c = frame_shape[COLOR_INDEX]

        if (frame_index == 0):
            frames = np.zeros([NUM_OF_FRAMES_IN_BUFFER, h, w, c])

        frames[frame_index, : , : , :] = frame

        if (frame_index < MIN_NUM_OF_FRMAES):
            frame_index += 1
            continue

        back_ground_by_mean = calc_background_by_mean(frames , frame_index , NUM_OF_FRAMES_FOR_BACKGROUND_CALC)
        back_ground_by_mean_uint8 = back_ground_by_mean.astype('uint8')

        frame_gray = rgb2gray(frame)
        frame_gray_uint8 = frame_gray.astype('uint8')

        # plt.figure(2)
        # plt.imshow(frame)
        # plt.title('%d' % (frame_index))

        diff_img_int_thresholded = calc_diff_img_threshold(frame , back_ground_by_mean_uint8, diff_threshold=diff_threshold)
        match_ball_color_img = find_match_color_img(frame, ball_color_threshold)

        potential_mask_img = np.multiply(diff_img_int_thresholded , match_ball_color_img)

        handle_blobs(potential_mask_img, frame, frame_index)

        # rolling_ball_frame_ball, rolling_ball_frame_background = calc_image_from_rolling_ball_alg(frame_gray_uint8, 2)
        frame_float_64 = frame.astype('float64')
        diff_img = frame_float_64 - back_ground_by_mean_uint8.astype('float64')
        diff_img_int = diff_img.astype('int64')
        diff_uint8 = diff_img.astype('uint8')
        #
        # diff_channel_0_thresholded = diff_img[:,:,0]<-20
        # diff_channel_1_thresholded = diff_img[:,:,1]<-60
        # diff_channel_2_thresholded = diff_img[:,:,2]<-60
        #
        #
        # diff_img_int_thresholded = np.multiply(diff_channel_0_thresholded , diff_channel_1_thresholded)
        # diff_img_int_thresholded = np.multiply(diff_img_int_thresholded , diff_channel_2_thresholded)

        background_gray = rgb2gray(back_ground_by_mean_uint8)
        diff_img_gray = rgb2gray(diff_img)
        background_gray_uint8 = background_gray.astype('uint8')
        diff_img_gray_uint8 = diff_img_gray.astype('uint8')

        frame_index += 1

        ## For debug
        # cv.imshow('frame %d'% (frame_index),frame)
        # cv.waitKey()

        # ax1.imshow(frame)
        # ax1.imshow(frame_gray_uint8, cmap = "gray")
        # ax1.set_title('Org %d' % (frame_index))
        #
        # # ax2.imshow(back_ground_by_mean_uin8)
        # ax2.imshow(background_gray_uint8, cmap = "gray")
        # ax2.set_title('backgroud 10 (%d - %d)' % (frame_index - MIN_NUM_OF_FRMAES , frame_index))
        #
        # # ax3.imshow(diff_img_gray)
        # ax3.imshow(diff_img_gray_uint8, cmap = "gray")
        # ax3.set_title('Diff image')


        # plt.waitforbuttonpress()

        plt.figure(1)
        plt.imshow(frame)
        plt.title('%d' % (frame_index))

        report_now, blob_inds_to_report = get_blobs_to_report(B)
        if (report_now):
            for blob_ind in blob_inds_to_report:
                cur_blob_points = B[blob_ind].pts
                hit_point = calc_hit_point(cur_blob_points)
                start_vel_angle, start_vel_strength = calc_start_velocity_vec(B[blob_ind])

        k=0

##

if __name__ == '__main__':
    path = r'/Users/eyalni/Eyal/Sivan/data/Sivan_1.mp4'
    start_frame = 70
    test_clip(path, start_frame)