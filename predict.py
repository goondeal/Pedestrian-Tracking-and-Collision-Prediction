import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
import matplotlib.pyplot as plt
import cv2


class Predictor:
    CAM_FPS = 30
    STEP = 3
    TIME_SEC = STEP / CAM_FPS 
    MAX_AGE = 30
    SAFE_DISTANCE_THRESHOLD = 10

    # M = np.array([
    #     [2.78644210e-16,  2.29706390e-01, -1.74947611e+02],
    #     [-4.43293034e-02, -7.59930915e-01,  2.76868164e+02],
    #     [1.74645934e-18, -3.07042794e-03,  1.00000000e+00]
    # ])
    M = np.array([
      [-3.25696188e-01, -1.76543893e+00,  7.35773724e+02],
      [ 9.47390314e-15, -3.81113414e+00,  1.29643161e+03],
      [ 1.20366147e-17, -3.39507486e-03,  1.00000000e+00]
      ])

    M_INV = inv(M)

    CAR_POINTS = np.array([
        [186, 716, 1],
        [1137, 716, 1],
    ]).T

    history = {}

    valley_x1 = 413
    valley_x2 = 622
    valley_y = 680

    def _get_transformed_points(self, M, points):
        # den = point @ M[2]  # M[2, 0] * x + M[2, 1] * y + M[2, 2]
        # x_new = point @ M[0] / den  # (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / den
        # y_new = point @ M[1] / den  # (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / den
        # return x_new, y_new
        assert points.shape[0] == 3, 'wrong points input shape: points.shape[0] must = 3' 
        new_points = M @ points
        # new_points = new_points / new_points[2]
        # print(new_points)
        return new_points

    TRANSFORMED_CAR_POINTS = M @ CAR_POINTS

    def predict(self, fid, uid, bbox):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            xpos = (bbox[0] + bbox[2]) // 2
            ypos = bbox[3]
            foot_on_land_points = np.array([
                    [xpos],
                    [ypos],
                    [1.0],
                ])
            new_point = self._get_transformed_points(Predictor.M, foot_on_land_points)
            
            curr_point = new_point # (3, 1)
            
            velocity = 0
            if self.history.get(uid):
                last_pos = self.history.get(uid).get('positions')[-1 if len(self.history.get(uid).get('positions')) < Predictor.STEP else -Predictor.STEP]
                velocity = (curr_point - last_pos) / (Predictor.TIME_SEC if len(self.history.get(uid).get('positions')) < Predictor.STEP else Predictor.TIME_SEC * Predictor.STEP)
                self.history[uid]['last_detected'] = fid
                self.history[uid]['positions'].append(curr_point)
                self.history[uid]['velocities'].append(velocity)
                velocity = sum(self.history[uid]['velocities'][-Predictor.STEP:])/ Predictor.STEP
            else:
                # velocity = (curr_point - curr_point) / Predictor.TIME_SEC
                self.history[uid] = {
                    'last_detected': fid,
                    'positions': [curr_point],
                    'velocities': [0]
                }

            next_frame_point = curr_point + velocity * Predictor.TIME_SEC * 5

            if (next_frame_point[1][0]/ next_frame_point[2][0] >= Predictor.valley_y) and (Predictor.valley_x1 <= next_frame_point[0][0]/ next_frame_point[2][0] <= Predictor.valley_x2):
                person_status = 'Dead'
                color = (0,0,255)
            else:
                person_status = 'Not Dead yet'
                color = (0,255,0)
            

            next_original_point = self._get_transformed_points(Predictor.M_INV, next_frame_point)
            # print(next_original_point.shape)
            # print(next_original_point)
            
            x = next_original_point[0,0]
            y = next_original_point[1,0]

            return (x-w//2, y-h, x+w//2, y), person_status, color
            